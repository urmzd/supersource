"""Course tests for L7.6: multi-head latent attention (tinyllm/modern/mla.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.6), and the chapter section it comes
from.

The worked example of the chapter (section 3): d = 2, one head, latent
width r = 1, dn = 1, dr = 2, dv = 1, inv_freq = (pi/2,), so position 1
rotates the rope pair by 90 degrees. Token 0 is x = (1, 1) at position 0,
token 1 is x = (0, -1) at position 1. Token 0 sees only itself and outputs
(3, 0); token 1 weighs the two keys by softmax(1/sqrt(3), 0) =
(0.640457, 0.359543) and outputs (0.842745, 0). The cache holds the
latents (1, -1) and the rotated rope keys (1, 1) and (0, -1): r + dr = 3
numbers per token.

The golden fixture (course/fixtures/L7.6/mla_hf.npz) holds two tiny
transformers 5.19.0 DeepseekV3Attention layers (a plain query and a
low-rank query with biases; interleaved and half rope), their inputs,
outputs, and the latents they cache, float32, from
course/oracle/L7.6/mla_hf.py.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.accounting import ModelConfig, kv_bytes_per_token
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.mla import ConcatLatentCache, MLAttention
from tinyllm.modern.rope import RopeSpec

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.6", "mla_hf.npz")
NOT_WEIGHTS = {"x", "positions", "out", "c", "k_rope", "inv_freq", "softmax_scale"}
CONFIGS = {"A": (None, "interleaved", False), "B": (12, "half", True)}


def ladder(r: int, base: float = 10000.0) -> np.ndarray:
    return base ** (-np.arange(0, r, 2, dtype=np.float64) / r)


def golden(prefix: str):
    f = np.load(FIX)
    rq, layout, bias = CONFIGS[prefix]
    spec = RopeSpec(
        inv_freq=f[f"{prefix}.inv_freq"],
        attention_scaling=1.0,
        layout=layout,
        rotary_dim=4,
    )
    m = MLAttention(32, 4, rq, 16, 8, 4, 6, spec, attention_bias=bias)
    sd = {
        k.split(".", 1)[1]: f[k]
        for k in f.files
        if k.startswith(prefix + ".") and k.split(".", 1)[1] not in NOT_WEIGHTS
    }
    return f, m, sd


def tiny(rq=None, layout="half", bias=False, seed=0, dtype=np.float64) -> MLAttention:
    """A random MLA (d 12, 3 heads, r 5, dn 4, dr 4, dv 3) with float64 weights."""
    spec = RopeSpec(
        inv_freq=ladder(4), attention_scaling=1.0, layout=layout, rotary_dim=4
    )
    m = MLAttention(12, 3, rq, 5, 4, 4, 3, spec, attention_bias=bias)
    g = PCG32(seed=100 + seed)
    for name, p in m.named_parameters():
        p.data = (
            0.5 * g.normal_array(p.shape)
            + (1.0 if name.endswith("layernorm.weight") else 0.0)
        ).astype(dtype)
    return m


def hand_mla() -> MLAttention:
    spec = RopeSpec(
        inv_freq=np.array([math.pi / 2]),
        attention_scaling=1.0,
        layout="half",
        rotary_dim=2,
    )
    m = MLAttention(2, 1, None, 1, 1, 2, 1, spec)
    m.load_state_dict(
        {
            "q_proj.weight": np.array(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            ),  # q = [x0 | x0, x1]
            "kv_a_proj_with_mqa.weight": np.array(
                [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
            ),  # c = x0 + x1, k_rope = (x1, x0)
            "kv_a_layernorm.weight": np.array([1.0]),
            "kv_b_proj.weight": np.array([[2.0], [3.0]]),  # k_nope = 2 c, v = 3 c
            "o_proj.weight": np.array([[1.0], [0.0]]),  # out = (o, 0)
        }
    )
    return m


# --- the worked example ------------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: one head, a one-number latent, a
    #      rope pair rotated by 90 degrees at position 1. The outputs, the
    #      attention weights behind them, and the cache contents are the
    #      numbers on paper.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, s08, m01
    # CHAPTER: L7.6 section 3, Worked example by hand
    m = hand_mla()
    cache = ConcatLatentCache()
    y = m(Tensor([[[1.0, 1.0], [0.0, -1.0]]]), np.array([0, 1]), cache=cache).data
    p0 = math.exp(3**-0.5) / (math.exp(3**-0.5) + 1.0)
    assert_close(p0, 0.640457, rtol=0, atol=1e-6)
    assert_close(y, [[[3.0, 0.0], [3.0 * (2 * p0 - 1), 0.0]]], rtol=0, atol=1e-5)
    assert_close(y[0, 1, 0], 0.842745, rtol=0, atol=1e-5)
    c, k = cache.update(0, np.zeros((1, 0, 1)), np.zeros((1, 0, 2)))
    assert_close(c[0, :, 0], [1.0, -1.0], rtol=0, atol=1e-5)
    assert_close(k[0], [[1.0, 1.0], [0.0, -1.0]], rtol=0, atol=1e-6)
    ab = m.absorb_weights().forward(
        np.array([[[1.0, 1.0], [0.0, -1.0]]]), np.array([0, 1]), ConcatLatentCache()
    )
    assert_close(ab, y, rtol=0, atol=1e-5)


# --- against Hugging Face's DeepseekV3Attention ----------------------------------------------


def test_state_dict_keys_and_shapes_are_hf():
    # WHY: a DeepSeek checkpoint loads by name: q_proj (or q_a_proj,
    #      q_a_layernorm, q_b_proj), kv_a_proj_with_mqa, kv_a_layernorm,
    #      kv_b_proj, o_proj, in HF's order and shapes.
    # KIND: unit
    # CATCHES: m04
    # CHAPTER: L7.6 section 4, The interface
    for prefix in CONFIGS:
        _, m, sd = golden(prefix)
        assert list(m.state_dict()) == list(sd), prefix
        for k, v in m.state_dict().items():
            assert v.shape == sd[k].shape and v.dtype == np.float32, k


@pytest.mark.parametrize("prefix", ["A", "B"])
def test_mla_golden(prefix):
    # WHY: the output of HF's DeepseekV3Attention under the causal mask, for
    #      a plain query with interleaved rope (A) and a low-rank query with
    #      biases, half rope, and per-row positions (B). The latent handed to
    #      the cache must be HF's too (normalized, before kv_b_proj), and in
    #      B the rotated rope key.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, s08, m01
    # CHAPTER: L7.6 section 2.2, The latent and the decoupled rope key
    f, m, sd = golden(prefix)
    m.load_state_dict(sd)
    cache = ConcatLatentCache()
    y = m(Tensor(f[f"{prefix}.x"]), f[f"{prefix}.positions"], cache=cache)
    assert y.shape == (2, 5, 32)
    assert_close(y.data, f[f"{prefix}.out"], rtol=1e-4, atol=1e-5, msg="out")
    c, k = cache.update(
        0, np.zeros((2, 0, 16), np.float32), np.zeros((2, 0, 4), np.float32)
    )
    assert_close(c, f[f"{prefix}.c"], rtol=1e-4, atol=1e-5, msg="cached latent")
    if f"{prefix}.k_rope" in f.files:
        assert_close(
            k, f[f"{prefix}.k_rope"], rtol=1e-4, atol=1e-5, msg="cached rope key"
        )
    assert m.softmax_scale == pytest.approx(
        float(f[f"{prefix}.softmax_scale"]), rel=1e-12
    )


# --- absorption and the cache ------------------------------------------------------------------


@pytest.mark.parametrize(
    "rq,layout,bias", [(None, "half", False), (6, "interleaved", True)]
)
def test_absorbed_decode_equals_naive(rq, layout, bias):
    # WHY: the point of MLA at inference: folding W_UK into the query and
    #      W_UV into o_proj gives the same outputs as building every head's
    #      keys and values, step by step through a prefill of 3 and 4 decode
    #      tokens (float64, so any gap is a real bug, not rounding).
    # KIND: differential
    # CATCHES: s01, s02, s07, s08, s10
    # CHAPTER: L7.6 section 2.4, Weight absorption
    m = tiny(rq, layout, bias)
    ab = m.absorb_weights()
    assert ab.w_uk.shape == (3, 4, 5) and ab.w_ov.shape == (3, 12, 5)
    x = PCG32(seed=11).normal_array((2, 7, 12))
    pos = np.arange(7)
    naive, absorbed = ConcatLatentCache(), ConcatLatentCache()
    steps = [(0, 3)] + [(t, t + 1) for t in range(3, 7)]
    for a, b in steps:
        want = m(Tensor(x[:, a:b], dtype=np.float64), pos[a:b], cache=naive).data
        got = ab.forward(x[:, a:b], pos[a:b], absorbed)
        assert got.dtype == np.float64
        assert_close(got, want, rtol=1e-9, atol=1e-11, msg=f"tokens {a}..{b - 1}")
    full = m(Tensor(x, dtype=np.float64), pos).data
    assert_close(want, full[:, 6:], rtol=1e-9, atol=1e-11)


def test_cache_chunks_equal_full_forward():
    # WHY: prefill in chunks of 2, 3, and 1 through ConcatLatentCache must
    #      give the full forward's outputs: each chunk's queries sit at the
    #      end of the cached keys (key index Tk - T + t), and the cache holds
    #      what the full forward would recompute.
    # KIND: differential
    # CATCHES: s08
    # CHAPTER: L7.6 section 2.3, What the cache holds
    m = tiny(4, "half", False, seed=1)
    x = PCG32(seed=12).normal_array((1, 6, 12))
    full = m(Tensor(x, dtype=np.float64), np.arange(6)).data
    cache, outs, t = ConcatLatentCache(), [], 0
    for n in (2, 3, 1):
        outs.append(
            m(
                Tensor(x[:, t : t + n], dtype=np.float64),
                np.arange(t, t + n),
                cache=cache,
                layer=0,
            ).data
        )
        t += n
    assert cache.seq_len(0) == 6 and cache.seq_len(1) == 0
    assert_close(np.concatenate(outs, axis=1), full, rtol=1e-9, atol=1e-11)


def test_cache_bytes_match_m05_1():
    # WHY: the reason to build MLA: a token costs r + dr numbers per layer,
    #      whatever the number of heads. The arrays handed to the cache are
    #      exactly M05.1's kv_bytes_per_token for an mla config, and far
    #      below the GQA cache of the same heads.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: L7.6 section 2.3, What the cache holds
    m = tiny(None, "half", False, dtype=np.float32)
    cache = ConcatLatentCache()
    layers, T = 2, 5
    x = Tensor(PCG32(seed=13).normal_array((1, T, 12)))
    for layer in range(layers):
        m(x, np.arange(T), cache=cache, layer=layer)
    held = 0
    for layer in range(layers):
        c, k = cache.update(
            layer, np.zeros((1, 0, 5), np.float32), np.zeros((1, 0, 4), np.float32)
        )
        assert c.shape == (1, T, 5) and k.shape == (1, T, 4)
        held += (c.shape[-1] + k.shape[-1]) * T * 2  # bf16 bytes
    cfg = ModelConfig(
        vocab=10,
        d_model=12,
        n_layers=layers,
        n_heads=3,
        n_kv_heads=3,
        d_head=4,
        d_ff=8,
        tie_embeddings=True,
        attn="mla",
        kv_lora_rank=5,
        qk_rope_dim=4,
        v_head_dim=3,
    )
    assert held == kv_bytes_per_token(cfg, 2) * T
    gqa = ModelConfig(
        vocab=10,
        d_model=12,
        n_layers=layers,
        n_heads=3,
        n_kv_heads=3,
        d_head=4,
        d_ff=8,
        tie_embeddings=True,
    )
    assert kv_bytes_per_token(cfg, 2) < kv_bytes_per_token(gqa, 2)


def test_scores_see_relative_position_only():
    # WHY: the decoupled rope key keeps RoPE's law: shifting every position
    #      by the same amount changes nothing, because only the rope parts
    #      rotate and they rotate by the same angles in q and k. A rope key
    #      rotated at the wrong position breaks it.
    # KIND: property
    # CATCHES: s03, s06
    # CHAPTER: L7.6 section 2.2, The latent and the decoupled rope key
    m = tiny(None, "interleaved", False, seed=2)
    x = Tensor(PCG32(seed=14).normal_array((2, 5, 12)), dtype=np.float64)
    a = m(x, np.arange(5)).data
    b = m(x, np.arange(5) + 37).data
    assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_padding_mask_and_fully_masked_rows():
    # WHY: mask (True = may attend) hides padded keys in both forms, and a
    #      query that sees no key at all outputs zeros through o_proj (no
    #      bias here), never NaN.
    # KIND: boundary
    # CATCHES: s01, s02, s07, s09, s10
    # CHAPTER: L7.6 section 4, The interface
    m = tiny(None, "half", False, seed=3)
    x = PCG32(seed=15).normal_array((2, 4, 12))
    keep = np.array([[True, True, True, True], [False, True, True, True]])[
        :, None, None, :
    ]
    y = m(Tensor(x, dtype=np.float64), np.arange(4), mask=keep).data
    assert np.isfinite(y).all()
    assert_close(y[1, 0], np.zeros(12), rtol=0, atol=0)
    want = m(Tensor(x[1:, 1:], dtype=np.float64), np.arange(1, 4)).data
    assert_close(y[1, 1:], want[0], rtol=1e-9, atol=1e-11)
    ab = m.absorb_weights().forward(x, np.arange(4), ConcatLatentCache(), mask=keep)
    assert_close(ab, y, rtol=1e-9, atol=1e-11)


def test_gradcheck_through_the_latent():
    # WHY: MLA trains end to end in C1's ablation: the loss must reach x and
    #      kv_a_proj_with_mqa through the latent, its norm, and the rope key.
    #      Checked against the frozen central differences in float64.
    # KIND: gradcheck
    # CATCHES: s11
    # CHAPTER: L7.6 section 2.5, Training the naive form
    m = tiny(3, "half", True, seed=4)
    g = PCG32(seed=16)
    x0, w0 = g.normal_array((1, 3, 12)), m.kv_a_proj_with_mqa.weight.data.copy()
    gy = g.normal_array((1, 3, 12))

    def f(x, w):
        m.kv_a_proj_with_mqa.weight.data = w
        return float(np.sum(m(Tensor(x, dtype=np.float64), np.arange(3)).data * gy))

    m.kv_a_proj_with_mqa.weight.data = w0
    m.zero_grad()
    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    F.sum(m(x, np.arange(3)) * gy).backward()
    gw = m.kv_a_proj_with_mqa.weight.grad
    gradcheck(f, [x0, w0], [x.grad, gw], names=["x", "kv_a_proj_with_mqa.weight"])


def test_concat_latent_cache():
    # WHY: the hook L8.2's LatentCache implements: it appends copies along
    #      time, returns everything held, counts positions per layer, and
    #      rejects a chunk that does not extend what it holds.
    # KIND: unit
    # CATCHES: s08, m03
    # CHAPTER: L7.6 section 2.3, What the cache holds
    cache = ConcatLatentCache()
    c = np.ones((2, 3, 5))
    k = np.zeros((2, 3, 4))
    ca, ka = cache.update(1, c, k)
    c[:] = 7.0
    assert ca.shape == (2, 3, 5) and float(ca.max()) == 1.0
    ca, ka = cache.update(1, np.full((2, 1, 5), 2.0), np.ones((2, 1, 4)))
    assert ca.shape == (2, 4, 5) and ka.shape == (2, 4, 4) and float(ca[0, 3, 0]) == 2.0
    assert cache.seq_len(1) == 4 and cache.seq_len(0) == 0
    for bad in (
        (np.ones((2, 1, 6)), np.ones((2, 1, 4))),
        (np.ones((3, 1, 5)), np.ones((3, 1, 4))),
        (np.ones((2, 1, 5)), np.ones((2, 2, 4))),
        (np.ones((2, 5)), np.ones((2, 4))),
    ):
        with pytest.raises(ValueError):
            cache.update(1, *bad)


def test_validation():
    # WHY: a rope width that differs from qk_rope_dim, an odd rope width,
    #      empty sizes, a bad scale, and wrong positions are wiring bugs of a
    #      config.json; fail loudly instead of attending wrongly.
    # KIND: boundary
    # CATCHES: s05, m02
    # CHAPTER: L7.6 section 4, The interface
    spec4 = RopeSpec(
        inv_freq=ladder(4), attention_scaling=1.0, layout="half", rotary_dim=4
    )
    with pytest.raises(ValueError):
        MLAttention(12, 3, None, 5, 4, 6, 3, spec4)  # rotary_dim != qk_rope_dim
    with pytest.raises(ValueError):
        MLAttention(
            12, 3, None, 5, 4, 3, 3, RopeSpec(ladder(4)[:1], 1.0, "half", 3)
        )  # odd rope width
    for args in (
        (12, 0, None, 5, 4, 4, 3),
        (12, 3, None, 0, 4, 4, 3),
        (12, 3, 0, 5, 4, 4, 3),
        (12, 3, None, 5, 4, 4, 0),
    ):
        with pytest.raises(ValueError):
            MLAttention(*args, spec4)
    with pytest.raises(ValueError):
        MLAttention(12, 3, None, 5, 4, 4, 3, spec4, softmax_scale=0.0)
    m = MLAttention(12, 3, None, 5, 4, 4, 3, spec4)
    assert m.softmax_scale == pytest.approx(8**-0.5)
    x = Tensor(np.ones((1, 2, 12)))
    with pytest.raises(ValueError):
        m(Tensor(np.ones((1, 2, 11))), np.arange(2))
    with pytest.raises(ValueError):
        m(x, np.arange(3))
    with pytest.raises(ValueError):
        m(x, np.arange(2), mask=np.ones((3, 3), dtype=bool))
