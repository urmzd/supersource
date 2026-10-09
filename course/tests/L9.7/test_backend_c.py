"""Course tests for L9.7: the Python C backend (tinyllm/backend/c.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L9.7), and the chapter section it comes
from.

`ss check L9.7` builds libtinyllm from your units (L9.1, L9.3, L9.6, rt.01,
or their references with --ref-deps) and puts it in TINYLLM_LIB; your
backend reaches it through rt.01's load().

The worked example of the chapter (section 3): a Linear with a bias is one
matmul. For x = (1, 2), W = [[1, 0], [3, 1]], b = (0.5, -1), the output
buffer is filled with b, then tl_matmul_f32 adds x W^T with beta = 1:
(0.5 + 1, -1 + 5) = (1.5, 4). The M09.3 bound of that dot product
(K = 2) is gamma_2 (|x| |W|^T) = (2u / (1 - 2u)) (1, 5) with u = 2^-24, and
the C result is exact, so the op check's ratio is 0. RoPE at position 1
with inv_freq 1 turns the pair (1, 0) into (cos 1, sin 1) =
(0.5403023, 0.8414710). RMSNorm of (3, 4) with gain 1 and eps 0 divides by
sqrt((9 + 16) / 2) = 3.5355339: (0.8485281, 1.1313708).

The golden fixtures are L7.9's five tiny Hugging Face models and their
float32 logits (course/fixtures/L7.9/hf_logits.npz) and the greedy ids of
tiny-llama-2l on its margin-filtered prompts (tiny_greedy_32.json).
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
import tinyllm.backend.c as cb
from tinyllm.ffi.libtinyllm import TlError, load
from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L7.9"
MODELS = [
    "tiny-llama-2l",
    "tiny-mistral-swa",
    "tiny-qwen2",
    "tiny-llama3-rope",
    "tiny-yarn",
]
U = 2.0**-24

_MODELS: dict[str, LlamaForCausalLM] = {}


def tiny(name: str) -> LlamaForCausalLM:
    if name not in _MODELS:
        _MODELS[name] = LlamaForCausalLM.from_pretrained(str(FIX / name))
    return _MODELS[name]


def golden(name: str) -> tuple[np.ndarray, np.ndarray]:
    f = np.load(FIX / "hf_logits.npz")
    return f[name + ".ids"], f[name + ".logits"]


def logits_close(got, want, msg: str = "") -> None:
    # float32 logits of a random tiny model reach about 20: the bar is
    # relative to that range (1e-4 of it, ten times tighter than MS-L9's
    # 1e-3 so a lost epsilon or scale cannot hide in it).
    want = np.asarray(want)
    assert_close(got, want, rtol=1e-4, atol=1e-4 * float(np.abs(want).max()), msg=msg)


def small(**kw) -> LlamaConfig:
    base = dict(
        vocab_size=29,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-5,
        max_position_embeddings=64,
    )
    base.update(kw)
    return LlamaConfig(**base)


class Faulty:
    """A Lib stand-in: every call goes to the real library, except one
    function, which is wrapped (to perturb its output or raise)."""

    def __init__(self, lib, name: str, wrap) -> None:
        self._lib, self._name, self._wrap = lib, name, wrap

    def declare(self, name, restype, argtypes) -> None:
        self._lib.declare(name, restype, argtypes)

    def __getattr__(self, name):
        fn = getattr(self._lib, name)
        return self._wrap(fn) if name == self._name else fn


def scale_output(index: int, factor: float):
    """Wrap a C function so its output buffer (argument `index`, a float
    pointer, n elements in the last argument or rows * d) is scaled after
    the call: a kernel that is wrong by a relative `factor - 1`."""

    def wrap(fn):
        def call(*args):
            r = fn(*args)
            n = int(args[-1]) if isinstance(args[-1], int) else None
            if n is None:  # rmsnorm: (x, w, y, rows, d, eps)
                n = int(args[3]) * int(args[4])
            y = args[index]
            for i in range(n):
                y[i] = y[i] * factor
            return r

        return call

    return wrap


# --- the worked example -------------------------------------------------------------------


def test_hand_example_linear_bias_and_rope():
    # WHY: the chapter's worked example: a Linear with a bias is ONE matmul
    #      over a buffer pre-filled with the bias (beta = 1), the result is
    #      exact here, and calling it twice for a single row gives the same
    #      answer (the bias rows are a fresh buffer, never the weight
    #      itself). Then one rope pair and one RMSNorm row by hand.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L9.7 section 3, Worked example by hand
    x = np.array([[1.0, 2.0]], np.float32)
    W = np.array([[1.0, 0.0], [3.0, 1.0]], np.float32)
    b = np.array([0.5, -1.0], np.float32)
    for _ in range(2):
        assert cb.linear(x, W, b).tolist() == [[1.5, 4.0]]
    assert b.tolist() == [0.5, -1.0]
    assert cb.linear(x, W).tolist() == [[1.0, 5.0]]
    bound = (2 * U / (1 - 2 * U)) * np.array([1.0, 5.0])
    assert np.all(np.abs(cb.linear(x, W, b)[0] - [1.5, 4.0]) <= bound)
    y = cb.rope(np.array([[[1.0, 0.0]]]), [1], [1.0])
    assert_close(y[0, 0], [np.cos(1.0), np.sin(1.0)], rtol=4 * U, atol=0)
    n = cb.rmsnorm(np.array([[3.0, 4.0]]), np.ones(2), 0.0)
    assert_close(n[0], [3 / np.sqrt(12.5), 4 / np.sqrt(12.5)], rtol=8 * U, atol=0)


def test_signatures_match_the_headers():
    # WHY: ctypes trusts the table, not the header: an int64_t declared as
    #      c_int, or a void function given a restype, passes garbage with no
    #      error. The widths here are read off elementwise.h and attention.h.
    # KIND: unit
    # CHAPTER: L9.7 section 2.2, Declaring the kernels
    s = cb.signatures()
    f32p, i32p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
    i64 = ctypes.c_int64
    assert s["tl_rmsnorm_f32"] == (None, [f32p, f32p, f32p, i64, i64, ctypes.c_float])
    assert s["tl_rope_f32"] == (
        None,
        [f32p, i32p, i64, i64, i64, i64, f32p, ctypes.c_float, ctypes.c_int],
    )
    assert s["tl_embedding_f32"] == (None, [f32p, i32p, f32p, i64, i64])
    assert s["tl_silu_mul_f32"][0] is None and s["tl_add_f32"][0] is None
    assert s["tl_argmax_f32"] == (ctypes.c_int32, [f32p, i64])
    restype, args = s["tl_flash_attn_fwd_f32"]
    assert len(args) == 20 and args[5:11] == [i64] * 6 and args[11] == ctypes.c_float
    assert (
        args[12] == i64
        and args[13] == ctypes.c_int
        and args[14] == i64
        and args[15] == f32p
    )
    assert args[18] == ctypes.c_void_p and args[19] == ctypes.c_void_p
    assert restype not in (None, ctypes.c_int32)  # the loader's STATUS marker


# --- the op check -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_op_check_passes_on_every_tiny_model(name):
    # WHY: the load-time check runs every op once on the loaded model's own
    #      shapes and weights and divides the error by its M09.3 bound. On
    #      correct kernels every ratio is at most 1: a bound that is too tight
    #      (one term of a rotated pair missing, gamma_1 for a K-term dot) would
    #      refuse a correct library at load.
    # KIND: differential
    # CATCHES: s14, m06
    # CHAPTER: L9.7 section 2.4, The load-time op check
    be = cb.CBackend(tiny(name), check=True)
    assert [r.op for r in be.report] == list(cb.OPS)
    for r in be.report:
        assert r.ok and 0.0 <= r.ratio <= 1.0, (name, r)
    by = {r.op: r for r in be.report}
    assert by["embedding"].ratio == 0.0 and by["argmax"].ratio == 0.0
    assert (
        by["matmul"].calls == 8
    )  # q, k, v, o, gate, up, down of layer 0, and the head


@pytest.mark.parametrize(
    "fn,index,factor,op",
    [
        ("tl_add_f32", 2, 1.0 + 1e-3, "add"),
        ("tl_rmsnorm_f32", 2, 1.0 + 1e-5, "rmsnorm"),
        ("tl_silu_mul_f32", 2, 1.0 + 1e-5, "silu_mul"),
    ],
)
def test_op_check_names_a_wrong_kernel(fn, index, factor, op):
    # WHY: the point of the check: a kernel off by 1e-5 relative (far below
    #      what a logits comparison notices) is named, with its ratio, before
    #      a token is produced, and CBackend(check=True) refuses to load.
    #      Every other op still passes: the report says which kernel to fix.
    # KIND: fault
    # CATCHES: s13
    # CHAPTER: L9.7 section 2.4, The load-time op check
    lib = Faulty(load(), fn, scale_output(index, factor))
    m = tiny("tiny-llama-2l")
    report = cb.check_ops(m, lib)
    bad = [r.op for r in report if not r.ok]
    assert bad == [op], report
    with pytest.raises(cb.OpCheckError) as e:
        cb.CBackend(m, lib=Faulty(load(), fn, scale_output(index, factor)), check=True)
    assert op in str(e.value) and [r.op for r in e.value.report if not r.ok] == [op]


def test_op_check_reports_a_stubbed_kernel():
    # WHY: before L9.3 is written its stub returns TL_EUNSUPPORTED, which the
    #      loader raises as TlError. The check turns that into a failed row
    #      (ratio inf, the error text) instead of crashing, so `--check` lists
    #      every missing kernel at once.
    # KIND: fault
    # CATCHES: s15
    # CHAPTER: L9.7 section 5, Pitfalls

    def stub(_fn):
        def call(*_args):
            raise TlError("tl_flash_attn_fwd_f32", 9, "unimplemented: L9.3")

        return call

    report = cb.check_ops(
        tiny("tiny-llama-2l"), Faulty(load(), "tl_flash_attn_fwd_f32", stub)
    )
    by = {r.op: r for r in report}
    assert not by["attention"].ok and by["attention"].ratio == float("inf")
    assert "unimplemented: L9.3" in by["attention"].detail
    assert all(r.ok for r in report if r.op != "attention")


# --- against Hugging Face and the numpy backend -------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_tiny_hf_logits_golden(name):
    # WHY: the C forward of five tiny checkpoints gives Hugging Face's
    #      float32 logits: GQA and MQA, a sliding window (Mistral), q/k/v
    #      biases (Qwen2), Llama-3 and YaRN rope scaling (attention_scaling),
    #      eps 1e-5 (tiny-llama-2l). One side of every differential test is
    #      checked against an oracle, so two equally wrong backends fail.
    # KIND: golden
    # CATCHES: s02, s04, s06, s09, m05
    # CHAPTER: L9.7 section 4, What the tests check
    ids, want = golden(name)
    logits_close(cb.CBackend(tiny(name)).forward(ids), want, name)


def test_logits_match_the_numpy_backend():
    # WHY: the design bar of L9.7: the C backend's logits equal the numpy
    #      backend's (L7.9's forward) on tiny-llama-2l, for a batch of two.
    # KIND: differential
    # CHAPTER: L9.7 section 4, What the tests check
    m = tiny("tiny-llama-2l")
    ids, _ = golden("tiny-llama-2l")
    logits_close(cb.CBackend(m).forward(ids), m(ids).data)


@pytest.mark.parametrize(
    "kw",
    [
        dict(learned_sinks=True, sliding_window=3),
        dict(
            tie_word_embeddings=True,
            head_dim=8,
            partial_rotary_factor=0.5,
            rope_interleaved=True,
        ),
        dict(mlp_bias=True, qkv_bias=True, num_key_value_heads=1),
        dict(num_key_value_heads=4, rope_theta=500.0, rms_norm_eps=0.05),
    ],
    ids=["sinks-window", "tied-partial-interleaved", "biases-mqa", "mha-large-eps"],
)
def test_config_features_match_the_numpy_backend(kw):
    # WHY: the features no committed checkpoint has, on random tiny models:
    #      learned sinks with a window, tied heads, partial and interleaved
    #      rotary (4 of 8 dims, where the two layouts differ), MLP and q/k/v
    #      biases with one kv head, full MHA with an eps large enough to
    #      matter (the tiny checkpoints' activations make eps 1e-5 invisible).
    #      The numpy backend (whose L7.9 tests checked it against HF) is the
    #      oracle.
    # KIND: differential
    # CATCHES: s05, s07, s08, s10
    # CHAPTER: L9.7 section 2.3, One forward, op by op
    m = LlamaForCausalLM(small(**kw))
    if kw.get("learned_sinks"):
        sd = m.state_dict()
        g = PCG32(7)
        for k in sd:
            if k.endswith("sinks"):
                sd[k] = g.normal_array(sd[k].shape, 1.0).astype(np.float32)
        m.load_state_dict(sd)
    ids = np.array([[1, 5, 9, 2, 28, 0, 3, 3, 17, 4]])
    logits_close(cb.CBackend(m).forward(ids), m(ids).data, str(kw))


def test_greedy_matches_hf_tokens():
    # WHY: MS-L9's first step in miniature: greedy decoding through the C
    #      backend (a cache, one token per step, tl_argmax_f32 with ties to
    #      the lowest id) reproduces HF's 32 greedy ids on every
    #      margin-filtered prompt of tiny-llama-2l.
    # KIND: golden
    # CATCHES: s03, s16
    # CHAPTER: L9.7 section 4, What the tests check
    doc = json.loads((FIX / "tiny_greedy_32.json").read_text())
    be = cb.CBackend(tiny("tiny-llama-2l"))
    for prompt, want in zip(doc["prompts"], doc["per_prompt"]):
        cache = be.new_cache()
        row = be.forward(list(prompt.encode()), cache=cache)[-1]
        out = []
        for _ in range(len(want)):
            out.append(cb.argmax(row))
            row = be.forward([out[-1]], cache=cache)[-1]
        assert out == want, prompt


# --- invariance: the properties batched serving relies on -----------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_cached_decode_is_bitwise_the_full_forward(name):
    # WHY: chunk invariance end to end: running a sequence in chunks through
    #      the cache (1 token, then 4, then 1, then the rest) gives logits
    #      BITWISE equal to the whole sequence at once. This needs q_offset =
    #      the cache length, a cache that grows without losing keys, and a
    #      bias buffer that is not the bias itself (Qwen2's first chunk is one
    #      row).
    # KIND: property
    # CATCHES: s01, s03, s16, m01
    # CHAPTER: L9.7 section 2.5, Invariance
    ids, _ = golden(name)
    row = ids[0]
    be = cb.CBackend(tiny(name))
    full = be.forward(row)
    cache = be.new_cache()
    cuts = [0, 1, 5, 6, len(row)]
    parts = [be.forward(row[a:b], cache=cache) for a, b in zip(cuts, cuts[1:])]
    assert cache.seq_len(0) == len(row)
    assert np.array_equal(np.concatenate(parts), full)


def test_varlen_batch_equals_per_sequence_calls():
    # WHY: the design's property for L9.7: three sequences of lengths 3, 12,
    #      and 7 packed into one pass (every projection one call over 22
    #      rows; each sequence's positions start at 0) give logits bitwise
    #      equal to three separate calls. It is what lets an engine batch
    #      requests without changing anyone's output (L9.1 batch invariance).
    # KIND: property
    # CATCHES: s12
    # CHAPTER: L9.7 section 2.5, Invariance
    ids, _ = golden("tiny-llama-2l")
    seqs = [ids[0][:3], ids[0], ids[1][:7]]
    be = cb.CBackend(tiny("tiny-llama-2l"))
    got = be.forward_varlen(seqs)
    assert [g.shape for g in got] == [(3, 256), (12, 256), (7, 256)]
    for g, s in zip(got, seqs):
        assert np.array_equal(g, be.forward(s))


def test_batch_rows_equal_single_calls():
    # WHY: a [B, T] batch shares every matmul; each row's logits are bitwise
    #      the row run alone, which needs every row's positions to restart
    #      at 0 (or at the cache length), not to run on across rows.
    # KIND: property
    # CATCHES: s17
    # CHAPTER: L9.7 section 2.5, Invariance
    ids, _ = golden("tiny-mistral-swa")
    be = cb.CBackend(tiny("tiny-mistral-swa"))
    both = be.forward(ids)
    for i in range(ids.shape[0]):
        assert np.array_equal(both[i], be.forward(ids[i]))


# --- boundaries ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw,word",
    [
        (
            dict(
                attention="mla",
                kv_lora_rank=4,
                qk_nope_head_dim=4,
                qk_rope_head_dim=2,
                v_head_dim=4,
            ),
            "mla",
        ),
        (
            dict(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=8),
            "num_experts",
        ),
        (dict(hidden_act="gelu_tanh"), "hidden_act"),
        (dict(sink_tokens=2, sliding_window=4), "sink_tokens"),
    ],
    ids=["mla", "moe", "gelu", "sink-tokens"],
)
def test_rejects_unsupported_configs(kw, word):
    # WHY: libtinyllm has no latent attention, no expert routing, no GeLU
    #      MLP op, and no window exemption for sink tokens. A backend that ran
    #      such a model anyway would produce plausible wrong text; it must
    #      refuse at construction, naming the feature.
    # KIND: boundary
    # CATCHES: m02, m03
    # CHAPTER: L9.7 section 2.3, One forward, op by op
    m = LlamaForCausalLM(small(**kw), init=False)
    with pytest.raises(ValueError, match=word):
        cb.CBackend(m)


def test_rejects_bad_ids_and_empty_sequences():
    # WHY: tl_embedding_f32 trusts its ids (elementwise.h): id == vocab would
    #      read past the table. The backend checks ids before the C call, and
    #      refuses float ids, 3-D ids, and an empty sequence in a varlen batch.
    # KIND: boundary
    # CATCHES: s11, m04
    # CHAPTER: L9.7 section 5, Pitfalls
    be = cb.CBackend(tiny("tiny-llama-2l"))
    for bad in ([1, 256], [-1, 2], np.zeros((1, 2, 3), np.int64), np.array([1.0, 2.0])):
        with pytest.raises(ValueError):
            be.forward(bad)
    with pytest.raises(ValueError):
        cb.embedding(np.zeros((4, 2), np.float32), [4])
    with pytest.raises(ValueError):
        be.forward_varlen([[1, 2], []])


def test_cache_belongs_to_one_batch_size():
    # WHY: a cache filled for a batch of 2 holds two rows of keys per layer;
    #      appending one row would mix sequences. It is a ValueError.
    # KIND: boundary
    # CHAPTER: L9.7 section 4, The interface
    be = cb.CBackend(tiny("tiny-llama-2l"))
    cache = be.new_cache()
    be.forward(np.array([[1, 2], [3, 4]]), cache=cache)
    with pytest.raises(ValueError):
        be.forward(np.array([[5]]), cache=cache)


def test_argmax_ties_and_nan():
    # WHY: greedy decoding's tie rule (spec/sampling.md): the lowest index
    #      wins, NaN entries are skipped, all-NaN is -1. Python and Rust
    #      engines must agree on it token for token.
    # KIND: boundary
    # CHAPTER: L9.7 section 4, What the tests check
    assert cb.argmax([0.5, 2.0, 2.0, 1.0]) == 1
    assert cb.argmax([np.nan, 1.0, 3.0, 3.0]) == 2
    assert cb.argmax([np.nan, np.nan]) == -1
