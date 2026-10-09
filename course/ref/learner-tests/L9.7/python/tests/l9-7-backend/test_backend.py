"""My tests for L9.7 (rung R7). Oracles: HF's golden logits of the five tiny
checkpoints (TINYLLM_FIXTURES), the numpy backend (L7.9) on configs no
checkpoint has, and the invariances themselves (chunked == whole, packed ==
separate, batch row == alone). They import only the contract."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tinyllm.backend.c as cb
from tinyllm.ffi.libtinyllm import TlError, load
from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM

FIX = Path(os.environ["TINYLLM_FIXTURES"]) / "L7.9"
NAMES = [
    "tiny-llama-2l",
    "tiny-mistral-swa",
    "tiny-qwen2",
    "tiny-llama3-rope",
    "tiny-yarn",
]


def model(name):
    return LlamaForCausalLM.from_pretrained(str(FIX / name))


def close(got, want):
    want = np.asarray(want)
    assert np.abs(got - want).max() <= 1e-4 * np.abs(want).max()


def cfg(**kw):
    base = dict(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=20,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    base.update(kw)
    return LlamaConfig(**base)


def test_linear_bias_one_row_twice():
    x, W, b = (
        np.array([[1.0, 2.0]], np.float32),
        np.array([[1.0, 0.0], [3.0, 1.0]], np.float32),
        np.array([0.5, -1.0], np.float32),
    )
    assert cb.linear(x, W, b).tolist() == [[1.5, 4.0]]
    assert cb.linear(x, W, b).tolist() == [[1.5, 4.0]]
    assert b.tolist() == [0.5, -1.0]


@pytest.mark.parametrize("name", NAMES)
def test_golden_logits(name):
    f = np.load(FIX / "hf_logits.npz")
    close(cb.CBackend(model(name)).forward(f[name + ".ids"]), f[name + ".logits"])


@pytest.mark.parametrize("name", ["tiny-llama-2l", "tiny-qwen2", "tiny-llama3-rope"])
def test_chunks_equal_whole(name):
    ids = np.load(FIX / "hf_logits.npz")[name + ".ids"][0]
    be = cb.CBackend(model(name))
    cache = be.new_cache()
    parts = [
        be.forward(ids[a:b], cache=cache) for a, b in [(0, 1), (1, 4), (4, len(ids))]
    ]
    assert np.array_equal(np.concatenate(parts), be.forward(ids))


def test_varlen_and_batch():
    be = cb.CBackend(model("tiny-llama-2l"))
    seqs = [[3, 4, 5], [9, 8, 7, 6, 5, 4], [1]]
    for got, s in zip(be.forward_varlen(seqs), seqs):
        assert np.array_equal(got, be.forward(s))
    ids = np.array([[1, 2, 3], [4, 5, 6]])
    both = be.forward(ids)
    assert np.array_equal(both[1], be.forward(ids[1]))


@pytest.mark.parametrize(
    "kw",
    [
        dict(learned_sinks=True, sliding_window=2),
        dict(
            partial_rotary_factor=0.5, rope_interleaved=True, tie_word_embeddings=True
        ),
        dict(rms_norm_eps=0.1, mlp_bias=True),
    ],
)
def test_against_numpy_backend(kw):
    m = LlamaForCausalLM(cfg(**kw))
    sd = m.state_dict()
    for k in sd:
        if k.endswith("sinks"):
            sd[k] = np.linspace(-1.0, 2.0, sd[k].size).astype(np.float32)
    m.load_state_dict(sd)
    ids = np.array([2, 7, 1, 18, 0, 5, 5, 9])
    close(cb.CBackend(m).forward(ids), m(ids).data)


def test_greedy_tokens():
    doc = json.loads((FIX / "tiny_greedy_32.json").read_text())
    be = cb.CBackend(model("tiny-llama-2l"))
    cache = be.new_cache()
    row = be.forward(list(doc["prompts"][0].encode()), cache=cache)[-1]
    out = []
    for _ in range(32):
        out.append(cb.argmax(row))
        row = be.forward([out[-1]], cache=cache)[-1]
    assert out == doc["per_prompt"][0]


def test_check_passes_and_catches():
    m = model("tiny-llama-2l")
    assert all(r.ok for r in cb.CBackend(m, check=True).report)

    class Off:
        def __init__(self, lib):
            self.lib = lib

        def declare(self, *a):
            self.lib.declare(*a)

        def __getattr__(self, name):
            fn = getattr(self.lib, name)
            if name == "tl_add_f32":

                def bad(a, b, y, n):
                    fn(a, b, y, n)
                    y[0] = y[0] * 1.01

                return bad
            if name == "tl_flash_attn_fwd_f32":

                def stub(*args):
                    raise TlError(name, 9, "unimplemented")

                return stub
            return fn

    rep = {r.op: r for r in cb.check_ops(m, Off(load()))}
    assert not rep["add"].ok and not rep["attention"].ok and rep["matmul"].ok
    with pytest.raises(cb.OpCheckError):
        cb.CBackend(m, lib=Off(load()), check=True)


def test_rejections():
    be = cb.CBackend(model("tiny-llama-2l"))
    for bad in ([0, 256], [1.5, 2.0]):
        with pytest.raises(ValueError):
            be.forward(np.array(bad))
    with pytest.raises(ValueError):
        cb.embedding(np.zeros((3, 2), np.float32), [3])
    with pytest.raises(ValueError):
        be.forward_varlen([[1], []])
    for kw in (
        dict(hidden_act="gelu_tanh"),
        dict(sink_tokens=1, sliding_window=3),
        dict(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=8),
    ):
        with pytest.raises(ValueError):
            cb.CBackend(LlamaForCausalLM(cfg(**kw), init=False))
