"""Course tests for L7.9: the Llama-family model (tinyllm/modern/llama.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.9), and the chapter section it comes
from. The downloader's tests are in test_hf.py.

The worked example of the chapter (section 3): SmolLM2-135M-Instruct's
config.json has no head_dim, so d_h = 576 / 9 = 64, and 3 kv heads. One
layer holds q and o (576 x 576 each), k and v (576 x 192 each), the gated MLP
(3 x 576 x 1536), and two norms (2 x 576): 3,540,096. Thirty layers, the
embedding (49152 x 576 = 28,311,552, also the lm_head: tied), and the final
norm (576) make 134,515,008, Hugging Face's num_parameters().

The golden fixtures (course/fixtures/L7.9/) are five tiny random models
saved by transformers 5.19.0 save_pretrained (Llama with BF16 weights and
the byte tokenizer, Mistral with a sliding window, Qwen2 with q/k/v biases,
Llama-3 and YaRN rope scaling), their float32 logits (hf_logits.npz), and
HF's parameter counts (hf_params.json), from course/oracle/L7.9/llama_hf.py.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.accounting import param_count
from tinyllm.autograd import functional as F
from tinyllm.io.safetensors import save_safetensors
from tinyllm.modern.gqa import ConcatKVCache
from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM
from tinyllm.modern.mla import ConcatLatentCache
from tinyllm.modern.window import SinkWindowCache

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L7.9"
MODELS = [
    "tiny-llama-2l",
    "tiny-mistral-swa",
    "tiny-qwen2",
    "tiny-llama3-rope",
    "tiny-yarn",
]


def small(**kw) -> LlamaConfig:
    base = dict(
        vocab_size=23,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        max_position_embeddings=64,
    )
    base.update(kw)
    return LlamaConfig(**base)


def as_f64(m: LlamaForCausalLM) -> LlamaForCausalLM:
    for _, p in m.named_parameters():
        p.data = p.data.astype(np.float64)
    return m


def logits_close(got: np.ndarray, want: np.ndarray, msg: str = "") -> None:
    # Random tiny models have logits up to about 20: float32 rounding is
    # relative to that range, so the 1e-5 bound is scaled by it.
    assert_close(got, want, rtol=1e-5, atol=1e-5 * float(np.abs(want).max()), msg=msg)


# --- the worked example ------------------------------------------------------------------


def test_hand_example_smollm2_config():
    # WHY: the chapter's worked example: SmolLM2's config.json read into
    #      LlamaConfig (head_dim derived, 3 kv heads, rope_theta 1e5, tied),
    #      and the model built from it holds exactly 134,515,008 parameters,
    #      M05.1's count and HF's num_parameters(), without any weights.
    # KIND: unit
    # CATCHES: s01, s07
    # CHAPTER: L7.9 section 3, Worked example by hand
    cfg = LlamaConfig.from_hf(str(FIX / "smollm2-135m-instruct.config.json"))
    assert (
        cfg.head_dim,
        cfg.num_key_value_heads,
        cfg.rope_theta,
        cfg.tie_word_embeddings,
    ) == (64, 3, 100000.0, True)
    assert (
        cfg.rms_norm_eps == 1e-5
        and cfg.rope_scaling is None
        and cfg.rope_spec().layout == "half"
    )
    layer = 2 * 576 * 576 + 2 * 576 * 192 + 3 * 576 * 1536 + 2 * 576
    assert layer == 3_540_096
    total = 30 * layer + 49152 * 576 + 576
    assert total == 134_515_008 == param_count(cfg.model_config())["total"]
    assert LlamaForCausalLM(cfg, init=False).param_count() == total


# --- against Hugging Face ----------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_tiny_hf_logits_golden(name):
    # WHY: five tiny random checkpoints exactly as transformers saves them
    #      (config.json in 5.x form, safetensors in BF16 or F32) load by name
    #      and give HF's float32 logits: GQA and MQA, a sliding window,
    #      q/k/v biases, tied and untied heads, Llama-3 and YaRN rope scaling
    #      past the original context.
    # KIND: golden
    # CATCHES: s02, s04, s05, s06
    # CHAPTER: L7.9 section 2.2, From config.json to modules
    f = np.load(FIX / "hf_logits.npz")
    m = LlamaForCausalLM.from_pretrained(str(FIX / name))
    y = m(f[f"{name}.ids"])
    assert y.dtype == np.float32 and y.shape == f[f"{name}.logits"].shape
    logits_close(y.data, f[f"{name}.logits"], msg=name)


def test_param_counts_match_hf_and_m05_1():
    # WHY: `{tinyllm} info` reports the parameter count; it must be HF's
    #      num_parameters() for every checkpoint and M05.1's count of the
    #      same config, a tied embedding counted once.
    # KIND: property
    # CATCHES: s06, s07
    # CHAPTER: L7.9 section 2.4, Counting what you loaded
    hf = json.loads((FIX / "hf_params.json").read_text())
    for name in MODELS:
        m = LlamaForCausalLM.from_pretrained(str(FIX / name))
        assert (
            m.param_count() == hf[name] == param_count(m.config.model_config())["total"]
        ), name


def test_state_dict_keys_are_hf():
    # WHY: the checkpoint contract is HF's names in HF's registration order:
    #      model.embed_tokens, then per layer self_attn, mlp, the two norms,
    #      then model.norm; no lm_head.weight when tied, lm_head.weight last
    #      when not.
    # KIND: unit
    # CATCHES: m07
    # CHAPTER: L7.9 section 2.3, The checkpoint contract
    keys = list(LlamaForCausalLM(small(num_hidden_layers=1)).state_dict())
    layer = "model.layers.0."
    assert keys == ["model.embed_tokens.weight"] + [
        layer + k
        for k in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
    ] + ["model.norm.weight"]
    assert (
        list(LlamaForCausalLM(small(tie_word_embeddings=False)).state_dict())[-1]
        == "lm_head.weight"
    )


# --- caches and positions ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny-llama-2l", "tiny-mistral-swa", "tiny-yarn"])
def test_cached_decode_equals_full_forward(name):
    # WHY: L8.2 decodes one token at a time through a cache; positions must
    #      continue from what the cache holds, and every layer must use its
    #      own slot. Prefill 5, then decode one by one, equals the full
    #      forward (the Mistral model also through a SinkWindowCache that
    #      evicts beyond its window of 4).
    # KIND: differential
    # CATCHES: s03, s05
    # CHAPTER: L7.9 section 2.5, Positions and the cache
    f = np.load(FIX / "hf_logits.npz")
    m = LlamaForCausalLM.from_pretrained(str(FIX / name))
    ids = f[f"{name}.ids"][:, :12]
    full = m(ids).data
    caches = [ConcatKVCache()] + (
        [SinkWindowCache(0, 4)] if name == "tiny-mistral-swa" else []
    )
    for cache in caches:
        outs = [m(ids[:, :5], cache=cache).data] + [
            m(ids[:, t : t + 1], cache=cache).data for t in range(5, 12)
        ]
        logits_close(np.concatenate(outs, axis=1), full, msg=type(cache).__name__)


def test_latent_attention_and_experts():
    # WHY: the course's own extensions load through the same model: MLA
    #      attention (with a latent cache) and an MoE MLP after one dense
    #      layer. The cached decode equals the full forward, aux_loss sums
    #      the MoE layers, and the count is M05.1's.
    # KIND: differential
    # CATCHES: s03, s07, s08, s10
    # CHAPTER: L7.9 section 2.2, From config.json to modules
    cfg = small(
        attention="mla",
        kv_lora_rank=6,
        q_lora_rank=None,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=3,
        rope_interleaved=True,
        num_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        num_hidden_layers=3,
    )
    m = as_f64(LlamaForCausalLM(cfg, rng=PCG32(seed=41)))
    ids = np.array([[3, 1, 4, 1, 5, 9, 2, 6]])
    full = m(ids).data
    assert m.aux_loss() is not None and m.aux_loss().data > 0
    cache = ConcatLatentCache()
    outs = [m(ids[:, :3], cache=cache).data] + [
        m(ids[:, t : t + 1], cache=cache).data for t in range(3, 8)
    ]
    assert_close(np.concatenate(outs, axis=1), full, rtol=1e-9, atol=1e-10)
    assert m.param_count() == param_count(cfg.model_config())["total"]
    assert (
        type(m.model.layers[0].mlp).__name__ == "GatedMLP"
        and type(m.model.layers[1].mlp).__name__ == "MoE"
    )


def test_sink_tokens_stream_like_full_attention():
    # WHY: tl_sliding_window with tl_sink_tokens: a stream decoded through a
    #      SinkWindowCache (bounded memory) gives the full forward's logits
    #      under L7.7's sink-window mask, with learned sinks on top.
    # KIND: differential
    # CATCHES: s03, s05, s09, s12
    # CHAPTER: L7.9 section 2.5, Positions and the cache
    cfg = small(sliding_window=3, sink_tokens=1, learned_sinks=True)
    m = as_f64(LlamaForCausalLM(cfg, rng=PCG32(seed=42)))
    for layer in m.model.layers:
        layer.self_attn.sinks.data = np.array([0.5, -1.0, 2.0, 0.0])
    ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    full = m(ids).data
    cache = SinkWindowCache(1, 3)
    outs = [m(ids[:, t : t + 1], cache=cache).data for t in range(10)]
    assert_close(np.concatenate(outs, axis=1), full, rtol=1e-9, atol=1e-10)
    assert cache.held(0) <= 1 + 3 - 1
    no_sinks = as_f64(LlamaForCausalLM(small(sliding_window=3), rng=PCG32(seed=42)))
    assert (
        float(np.abs(no_sinks(ids).data - full).max()) > 1e-3
    )  # the sinks do change the output


# --- configs and files ------------------------------------------------------------------------------


def test_config_roundtrip_and_both_rope_forms():
    # WHY: save_pretrained writes to_hf() and from_pretrained reads it back
    #      unchanged; a 4.x config (rope_theta, rope_scaling with "type") and
    #      a 5.x config (rope_parameters) of the same model read the same.
    # KIND: property
    # CATCHES: s02, s11, m03
    # CHAPTER: L7.9 section 2.2, From config.json to modules
    cfgs = [
        small(),
        small(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 16,
            }
        ),
        small(
            attention="mla",
            kv_lora_rank=6,
            q_lora_rank=5,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=3,
        ),
        small(
            num_experts=4,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            router="sigmoid",
        ),
        small(
            sliding_window=5,
            sink_tokens=2,
            learned_sinks=True,
            qkv_bias=True,
            hidden_act="gelu_tanh",
        ),
    ]
    tmp = Path(tempfile.mkdtemp(prefix="l79-"))
    for i, cfg in enumerate(cfgs):
        p = tmp / f"l79-roundtrip-{os.getpid()}-{i}.json"
        p.write_text(json.dumps(cfg.to_hf()))
        try:
            assert LlamaConfig.from_hf(str(p)) == cfg, i
        finally:
            p.unlink()
    v4 = {
        "model_type": "llama",
        "vocab_size": 11,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "rope_theta": 5000.0,
        "rope_scaling": {"type": "linear", "factor": 2.0},
    }
    v5 = dict(
        v4, rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 5000.0}
    )
    del v5["rope_theta"], v5["rope_scaling"]
    a, b = tmp / f"l79-v4-{os.getpid()}.json", tmp / f"l79-v5-{os.getpid()}.json"
    a.write_text(json.dumps(v4))
    b.write_text(json.dumps(v5))
    try:
        ca, cb = LlamaConfig.from_hf(str(a)), LlamaConfig.from_hf(str(b))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert (
        ca == cb
        and ca.rope_theta == 5000.0
        and ca.rope_scaling == {"rope_type": "linear", "factor": 2.0}
    )
    assert ca.head_dim == 4 and ca.num_key_value_heads == 2
    assert_close(
        ca.rope_spec().inv_freq,
        5000.0 ** (-np.arange(0, 4, 2) / 4) / 2.0,
        rtol=1e-12,
        atol=0,
    )


def test_save_load_roundtrip_and_shards():
    # WHY: what you save loads back bit for bit (F32), BF16 rounds every
    #      weight to 8 mantissa bits and still loads, and a sharded checkpoint
    #      (model.safetensors.index.json) is read across its files.
    # KIND: property
    # CATCHES: m08, m09
    # CHAPTER: L7.9 section 2.3, The checkpoint contract
    m = LlamaForCausalLM(small(tie_word_embeddings=False), rng=PCG32(seed=43))
    ids = np.array([[1, 2, 3, 4]])
    tmp = Path(tempfile.mkdtemp(prefix="l79-save-"))
    try:
        m.save_pretrained(str(tmp / "f32"))
        back = LlamaForCausalLM.from_pretrained(str(tmp / "f32"))
        assert back.config == m.config
        assert np.array_equal(back(ids).data, m(ids).data)
        m.save_pretrained(str(tmp / "bf16"), dtype="BF16")
        half = LlamaForCausalLM.from_pretrained(str(tmp / "bf16"))
        w = half.model.layers[0].self_attn.q_proj.weight.data
        assert np.all(w.view(np.uint32) & 0xFFFF == 0)
        assert_close(half(ids).data, m(ids).data, rtol=0.05, atol=0.05)
        sd = m.state_dict()
        names = list(sd)
        shard = tmp / "sharded"
        shard.mkdir()
        shutil.copy(tmp / "f32" / "config.json", shard / "config.json")
        cut = len(names) // 2
        save_safetensors(
            str(shard / "a.safetensors"),
            {k: sd[k] for k in names[:cut]},
            {"format": "pt"},
        )
        save_safetensors(
            str(shard / "b.safetensors"),
            {k: sd[k] for k in names[cut:]},
            {"format": "pt"},
        )
        wm = {
            k: ("a.safetensors" if i < cut else "b.safetensors")
            for i, k in enumerate(names)
        }
        (shard / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": wm})
        )
        assert np.array_equal(
            LlamaForCausalLM.from_pretrained(str(shard))(ids).data, m(ids).data
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gradients_reach_every_parameter():
    # WHY: C1 trains this model: the cross-entropy reaches every parameter,
    #      the tied embedding through both of its uses (lookup and lm_head).
    #      The embedding's gradient is checked against the frozen central
    #      differences in float64.
    # KIND: gradcheck
    # CATCHES: s04
    # CHAPTER: L7.9 section 2.6, Training it
    m = as_f64(
        LlamaForCausalLM(
            small(
                vocab_size=7,
                hidden_size=8,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
            ),
            rng=PCG32(seed=44),
        )
    )
    ids, tgt = np.array([[1, 3, 5]]), np.array([[3, 5, 6]])
    gy = PCG32(seed=45).normal_array((1, 3, 7))
    w0 = m.model.embed_tokens.weight.data.copy()

    def loss(w):
        m.model.embed_tokens.weight.data = w
        return float(np.sum(m(ids).data * gy))

    m.zero_grad()
    F.sum(m(ids) * gy).backward()
    gradcheck(
        loss, [w0], [m.model.embed_tokens.weight.grad], names=["embed_tokens.weight"]
    )
    m.model.embed_tokens.weight.data = w0
    m.zero_grad()
    logp = F.log_softmax(m(ids), axis=-1)
    (-F.sum(F.gather(logp, tgt[..., None], axis=-1))).backward()
    silent = [
        n for n, p in m.named_parameters() if p.grad is None or not np.any(p.grad)
    ]
    assert silent == [], silent


def test_validation():
    # WHY: a wrong key in a checkpoint, an unsupported rope type, a Llama
    #      config whose attention_bias also biases o_proj, and ids outside the
    #      vocabulary must fail loudly, not load into a different model.
    # KIND: boundary
    # CATCHES: s02, s06, m01, m02, m04
    # CHAPTER: L7.9 section 4, The interface
    tmp = Path(tempfile.mkdtemp(prefix="l79-bad-"))
    try:
        shutil.copytree(FIX / "tiny-qwen2", tmp / "q")
        m = LlamaForCausalLM.from_pretrained(str(tmp / "q"))
        sd = m.state_dict()
        sd["model.layers.0.self_attn.extra.weight"] = np.zeros(1, np.float32)
        save_safetensors(str(tmp / "q" / "model.safetensors"), sd, {"format": "pt"})
        with pytest.raises(KeyError):
            LlamaForCausalLM.from_pretrained(str(tmp / "q"))
        base = json.loads((FIX / "tiny-llama-2l" / "config.json").read_text())
        for patch in (
            {
                "rope_parameters": {
                    "rope_type": "dynamic",
                    "factor": 2.0,
                    "rope_theta": 1e4,
                }
            },
            {"attention_bias": True},
            {"model_type": "gpt2", "tl_arch": "gpt"},
            {"hidden_size": None},
        ):
            bad = {k: v for k, v in dict(base, **patch).items() if v is not None}
            (tmp / "c.json").write_text(json.dumps(bad))
            with pytest.raises(ValueError):
                LlamaConfig.from_hf(str(tmp / "c.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    m = LlamaForCausalLM(small())
    for bad in (
        np.array([[0, 23]]),
        np.array([[0.5, 1.0]]),
        np.zeros((1, 1, 2), dtype=np.int64),
    ):
        with pytest.raises(ValueError):
            m(bad)
    assert m(np.array([1, 2, 3])).shape == (3, 23)
