"""Course tests for L7.5: grouped-query attention with a cache hook, a sliding
window, and learned sinks (tinyllm/modern/gqa.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.5), and the chapter section it comes
from.

The worked example of the chapter (section 3): d = 2, 2 query heads over 1
kv head of width 2, RoPE off (inv_freq 0). q_proj makes head 0 read x and
head 1 read -x; k = x; v = (x0, 2 x1); o_proj keeps entry 0 of head 0 and
entry 1 of head 1. Tokens (1, 0) and (0, 1). Token 0 sees only itself:
output (1, 0). Token 1, head 0 scores (0, 0.707107), weights
(0.330237, 0.669763); head 1 scores (0, -0.707107), weights
(0.669763, 0.330237); output (0.330237, 0.660474).

The golden fixture (course/fixtures/L7.5/gqa_hf.npz) holds transformers
5.19.0 LlamaAttention (with and without an extra mask), Qwen2Attention (q,
k, v biases), and GptOssAttention (sinks, window 3, per-row positions with
gaps) float32 outputs and gradients, from course/oracle/L7.5/gqa_hf.py.
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
from tinyllm.modern.gqa import ConcatKVCache, GQAttention, repeat_kv
from tinyllm.modern.rope import RopeSpec
from tinyllm.nn.layers import Linear

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.5", "gqa_hf.npz")


class Rng:
    """The frozen PCG32 behind the generator API of M06.3, so no verdict here
    depends on your PCG32."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        return self.g.below(n)


def spec(
    dh: int = 8, scaling: float = 1.0, rot: int | None = None, layout: str = "half"
) -> RopeSpec:
    r = rot or dh
    return RopeSpec(
        inv_freq=1e4 ** (-np.arange(0, r, 2) / r),
        attention_scaling=scaling,
        layout=layout,
        rotary_dim=r,
    )


def attn(seed: int = 0, **kw) -> GQAttention:
    args = dict(d=24, n_heads=6, n_kv_heads=2, d_head=8, rope=spec())
    args.update(kw)
    return GQAttention(**args, rng=Rng(seed))


def inputs(seed: int, B: int = 2, T: int = 6, d: int = 24) -> np.ndarray:
    return PCG32(seed=seed).normal_array((B, T, d)).astype(np.float32)


class CountingCache(ConcatKVCache):
    """A ConcatKVCache that records what the attention layer hands it."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[int, int, tuple]] = []

    def update(self, layer, k_new, v_new):
        self.seen.append((layer, int(k_new.nbytes + v_new.nbytes), tuple(k_new.shape)))
        return super().update(layer, k_new, v_new)


# --- the worked example ------------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: two query heads read ONE shared kv
    #      head; head 0 attends toward token 1, head 1 away from it; the
    #      first token sees only itself (causal).
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: L7.5 section 3, Worked example by hand
    a = GQAttention(2, 2, 1, 2, RopeSpec(np.array([0.0]), 1.0, "half", 2), rng=Rng(0))
    a.load_state_dict(
        {
            "q_proj.weight": [[1, 0], [0, 1], [-1, 0], [0, -1]],
            "k_proj.weight": [[1, 0], [0, 1]],
            "v_proj.weight": [[1, 0], [0, 2]],
            "o_proj.weight": [[1, 0, 0, 0], [0, 0, 0, 1]],
        }
    )
    y = a(Tensor([[[1.0, 0.0], [0.0, 1.0]]]), np.arange(2))
    w = 1.0 / (1.0 + math.exp(math.sqrt(0.5)))
    assert_close(y.data, [[[1.0, 0.0], [w, 2.0 * w]]], dtype="float32")


# --- against Hugging Face ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["llama", "llama-mask", "qwen2-bias", "gptoss"])
def test_golden_hf(name):
    # WHY: Llama's GQA (6 query heads over 2 kv heads, head_dim 8 so
    #      H dh != d), the same with an extra boolean mask, Qwen2's q/k/v
    #      biases, and gpt-oss's learned sinks with a sliding window and
    #      per-row positions with gaps: the same outputs and the same
    #      gradients for x and every parameter as Hugging Face's eager
    #      attention with copied weights.
    # KIND: golden
    # CATCHES: s01, s02, s04, s05, s07, s08, s09, s11
    # CHAPTER: L7.5 section 4, The interface
    f = np.load(FIX)
    kw = {"qwen2-bias": {"qkv_bias": True}, "gptoss": {"window": 3, "sinks": True}}.get(
        name, {}
    )
    a = attn(1, rope=RopeSpec(f[f"{name}.inv_freq"], 1.0, "half", 8), **kw)
    p = f"{name}.param."
    a.load_state_dict({k[len(p) :]: f[k] for k in f.files if k.startswith(p)})
    x = Tensor(f[f"{name}.x"], requires_grad=True)
    mask = f[f"{name}.visible"] if name == "llama-mask" else None
    y = a(x, f[f"{name}.positions"], mask)
    F.sum(y * Tensor(f[f"{name}.g"])).backward()
    assert_close(y.data, f[f"{name}.y"], rtol=1e-4, atol=1e-5)
    assert_close(x.grad, f[f"{name}.grad.x"], rtol=1e-4, atol=1e-5, msg="x")
    for n, prm in a.named_parameters():
        assert_close(prm.grad, f[f"{name}.grad.{n}"], rtol=1e-4, atol=1e-5, msg=n)


def test_gradcheck_every_parameter():
    # WHY: backward reaches x, the four projections with their biases, and
    #      the sinks, through RoPE, the window mask, and the sink column;
    #      checked against the frozen central differences in float64.
    # KIND: gradcheck
    # CATCHES: s12
    # CHAPTER: L7.5 section 2.6, Backward
    a = GQAttention(
        4, 2, 1, 2, spec(2), qkv_bias=True, window=2, sinks=True, rng=Rng(2)
    )
    g = PCG32(seed=3)
    params = [p for _, p in a.named_parameters()]
    for p in params:
        p.data = g.normal_array(p.data.shape)
    x0, gy = g.normal_array((2, 3, 4)), g.normal_array((2, 3, 4))
    start = [p.data.copy() for p in params]

    def f(x, *ps):
        for p, v in zip(params, ps):
            p.data = v
        return float(np.sum(a(Tensor(x, dtype=np.float64), np.arange(3)).data * gy))

    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    F.sum(a(x, np.arange(3)) * gy).backward()
    gradcheck(
        f,
        [x0] + start,
        [x.grad] + [p.grad for p in params],
        names=["x"] + [n for n, _ in a.named_parameters()],
    )


# --- grouping ------------------------------------------------------------------------------------


def test_repeat_kv_repeats_each_head_in_a_row():
    # WHY: query head h reads kv head h // n_rep: kv heads (a, b) with
    #      n_rep 3 become (a, a, a, b, b, b), not (a, b, a, b, a, b); the
    #      gradient of a kv head is the sum over its copies.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L7.5 section 2.2, Sharing kv heads
    x = Tensor(np.array([10.0, 20.0]).reshape(1, 2, 1, 1), requires_grad=True)
    y = repeat_kv(x, 3)
    assert y.shape == (1, 6, 1, 1) and y.data.ravel().tolist() == [
        10,
        10,
        10,
        20,
        20,
        20,
    ]
    F.sum(y * np.arange(1.0, 7.0).reshape(1, 6, 1, 1)).backward()
    assert x.grad.ravel().tolist() == [6.0, 15.0]
    assert repeat_kv(x, 1) is x


def test_gqa_equals_mha_with_repeated_kv_weights():
    # WHY: GQA is MHA whose key and value heads come in identical groups:
    #      an MHA layer (Hkv = H) whose k_proj and v_proj repeat each GQA kv
    #      head n_rep times gives the same output.
    # KIND: differential
    # CATCHES: s01
    # CHAPTER: L7.5 section 2.2, Sharing kv heads
    g = attn(4)
    m = attn(5, n_kv_heads=6)
    sd = g.state_dict()
    for k in ("k_proj.weight", "v_proj.weight"):
        sd[k] = np.repeat(sd[k].reshape(2, 8, 24), 3, axis=0).reshape(48, 24)
    m.load_state_dict(sd)
    x = inputs(6)
    assert_close(
        g(Tensor(x), np.arange(6)).data,
        m(Tensor(x), np.arange(6)).data,
        rtol=1e-5,
        atol=1e-6,
    )


# --- the cache hook ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{}, {"window": 3, "sinks": True}])
def test_cache_chunks_equal_full_forward(kw):
    # WHY: prefilling 3 tokens and decoding the next 3 one at a time through
    #      a cache must reproduce the full forward: each new query sits at
    #      key index Tk - T + t, and RoPE uses the absolute positions.
    # KIND: differential
    # CATCHES: s03, s06
    # CHAPTER: L7.5 section 2.4, The cache hook
    a = attn(7, **kw)
    if a.sinks is not None:
        a.sinks.data[:] = np.linspace(-1, 1, 6)
    x = inputs(8)
    full = a(Tensor(x), np.arange(6)).data
    cache = ConcatKVCache()
    parts = [a(Tensor(x[:, :3]), np.arange(3), cache=cache, layer=4).data]
    for t in range(3, 6):
        parts.append(
            a(Tensor(x[:, t : t + 1]), np.array([t]), cache=cache, layer=4).data
        )
    assert cache.seq_len(4) == 6 and cache.seq_len(0) == 0
    assert_close(np.concatenate(parts, axis=1), full, rtol=1e-5, atol=1e-6)


def test_cache_holds_kv_heads_only():
    # WHY: GQA's memory saving happens in the cache: the hook receives the
    #      Hkv = 2 kv heads per token (after RoPE), never the 6 repeated
    #      ones, and the bytes it receives per token are exactly M05.1's
    #      kv_bytes_per_token for one layer in float32.
    # KIND: property
    # CATCHES: s10
    # CHAPTER: L7.5 section 2.4, The cache hook
    a = attn(9)
    cache = CountingCache()
    a(Tensor(inputs(10, B=1, T=5)), np.arange(5), cache=cache, layer=2)
    ((layer, nbytes, shape),) = cache.seen
    assert layer == 2 and shape == (1, 2, 5, 8)
    cfg = ModelConfig(
        vocab=1,
        d_model=24,
        n_layers=1,
        n_heads=6,
        n_kv_heads=2,
        d_head=8,
        d_ff=1,
        tie_embeddings=True,
    )
    assert nbytes == 5 * kv_bytes_per_token(cfg, 4)


def test_concat_cache():
    # WHY: the reference hook appends along time per layer, keeps copies
    #      (a caller reusing its buffer must not corrupt the cache), and
    #      rejects a chunk that does not extend the layer.
    # KIND: unit
    # CATCHES: m01, m02
    # CHAPTER: L7.5 section 2.4, The cache hook
    c = ConcatKVCache()
    k = np.ones((1, 2, 3, 4), dtype=np.float32)
    ka, va = c.update(0, k, 2 * k)
    k[:] = 7.0
    ka, va = c.update(0, k[:, :, :1], k[:, :, :1])
    assert (
        ka.shape == (1, 2, 4, 4)
        and np.all(ka[:, :, :3] == 1.0)
        and np.all(ka[:, :, 3] == 7.0)
    )
    assert np.all(va[:, :, :3] == 2.0) and c.seq_len(0) == 4
    with pytest.raises(ValueError):
        c.update(0, np.ones((1, 3, 1, 4)), np.ones((1, 3, 1, 4)))
    with pytest.raises(ValueError):
        c.update(1, np.ones((1, 2, 1, 4)), np.ones((1, 2, 1, 3)))


# --- masks, windows, sinks -------------------------------------------------------------------------


def test_causality_bitwise():
    # WHY: a decoder must not see the future: changing tokens 3.. leaves the
    #      outputs at 0..2 bit for bit identical.
    # KIND: property
    # CATCHES: s11
    # CHAPTER: L7.5 section 2.3, Who may see whom
    a = attn(11)
    x = inputs(12)
    y1 = a(Tensor(x), np.arange(6)).data
    x2 = x.copy()
    x2[:, 3:] = 100.0 * inputs(13)[:, 3:]
    y2 = a(Tensor(x2), np.arange(6)).data
    assert np.array_equal(y1[:, :3], y2[:, :3])
    assert np.abs(y1[:, 3:] - y2[:, 3:]).max() > 1e-3


def test_window_reach():
    # WHY: with window 3, query t reads keys t-2, t-1, t: changing token
    #      t-3 leaves output t bitwise unchanged, changing token t-2 does not.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: L7.5 section 2.3, Who may see whom
    a = attn(14, window=3)
    x = inputs(15)
    y = a(Tensor(x), np.arange(6)).data
    far, near = x.copy(), x.copy()
    far[:, 2] += 5.0
    near[:, 3] += 5.0
    assert np.array_equal(a(Tensor(far), np.arange(6)).data[:, 5], y[:, 5])
    assert np.abs(a(Tensor(near), np.arange(6)).data[:, 5] - y[:, 5]).max() > 1e-4


def test_sinks_take_weight_from_every_key():
    # WHY: a sink logit joins every row's softmax and is then dropped: a
    #      large sink takes almost all the weight (the head attends to
    #      nothing, output about 0), a very negative one changes nothing.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: L7.5 section 2.5, Learned sinks
    x = inputs(16)
    plain = attn(17)(Tensor(x), np.arange(6)).data
    a = attn(17, sinks=True)
    a.sinks.data[:] = -1e4
    assert_close(a(Tensor(x), np.arange(6)).data, plain, rtol=1e-6, atol=1e-7)
    a.sinks.data[:] = 40.0
    assert np.abs(a(Tensor(x), np.arange(6)).data).max() < 1e-6


@pytest.mark.parametrize("sinks", [False, True])
def test_fully_masked_row_is_zero(sinks):
    # WHY: a row whose every key is hidden (a padded batch row) gives
    #      weights 0 and output 0, not NaN, with or without sinks; the other
    #      rows are unaffected.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: L7.5 section 2.3, Who may see whom
    a = attn(18, sinks=sinks)
    x = inputs(19)
    mask = np.ones((2, 1, 6, 6), dtype=bool)
    mask[0] = False
    y = a(Tensor(x), np.arange(6), mask).data
    assert np.all(np.isfinite(y)) and np.all(y[0] == 0.0)
    assert_close(y[1], a(Tensor(x), np.arange(6)).data[1], rtol=1e-6, atol=1e-7)


def test_positions_shift_and_per_row():
    # WHY: RoPE makes scores depend on relative position only, so the same
    #      tokens at positions 0..5 or 100..105 give the same outputs (up to
    #      float32 rounding of the angles), which lets a cache continue at
    #      any offset; and [B, T] positions rotate each row by its own
    #      positions, gaps included, exactly as a separate call per row.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: L7.5 section 2.1, Grouped-query attention
    a = attn(20)
    x = inputs(21)
    y0 = a(Tensor(x), np.arange(6)).data
    assert_close(a(Tensor(x), np.arange(100, 106)).data, y0, rtol=1e-4, atol=1e-5)
    rows = np.array([[0, 1, 2, 3, 4, 5], [0, 2, 3, 7, 8, 12]])
    y = a(Tensor(x), rows).data
    for b in range(2):
        assert_close(
            y[b],
            a(Tensor(x[b : b + 1]), rows[b]).data[0],
            rtol=1e-5,
            atol=1e-6,
            msg=f"row {b}",
        )


def test_attention_scaling_squares_into_the_scores():
    # WHY: YaRN's attention scaling s multiplies q and k (through cos and
    #      sin), so every score by s^2: the same as an unscaled layer whose
    #      q_proj is multiplied by s^2.
    # KIND: property
    # CATCHES: s13
    # CHAPTER: L7.5 section 2.1, Grouped-query attention
    s = 1.3
    a = attn(22, rope=spec(scaling=s))
    b = attn(22)
    sd = b.state_dict()
    sd["q_proj.weight"] = sd["q_proj.weight"] * s * s
    b.load_state_dict(sd)
    x = inputs(23)
    assert_close(
        a(Tensor(x), np.arange(6)).data,
        b(Tensor(x), np.arange(6)).data,
        rtol=1e-4,
        atol=1e-5,
    )


# --- structure ------------------------------------------------------------------------------------


def test_parameter_names_shapes_and_draw_order():
    # WHY: the keys are HF's (`q_proj.weight` ... `o_proj.weight`, `sinks`
    #      listed first as the module's own parameter), o_proj has no bias
    #      even with qkv_bias, d_head None means d / H, and the four Linear
    #      layers draw from the one rng in the order q, k, v, o.
    # KIND: unit
    # CATCHES: s09, m03
    # CHAPTER: L7.5 section 4, The interface
    a = attn(24, qkv_bias=True, sinks=True)
    assert [(n, p.data.shape) for n, p in a.named_parameters()] == [
        ("sinks", (6,)),
        ("q_proj.weight", (48, 24)),
        ("q_proj.bias", (48,)),
        ("k_proj.weight", (16, 24)),
        ("k_proj.bias", (16,)),
        ("v_proj.weight", (16, 24)),
        ("v_proj.bias", (16,)),
        ("o_proj.weight", (24, 48)),
    ]
    assert np.all(a.sinks.data == 0.0) and a.sinks.data.dtype == np.float32
    r = Rng(25)
    want = [
        Linear(24, 48, False, r),
        Linear(24, 16, False, r),
        Linear(24, 16, False, r),
        Linear(48, 24, False, r),
    ]
    b = attn(25)
    for (n, p), lin in zip(b.named_parameters(), want):
        assert np.array_equal(p.data, lin.weight.data), n
    c = GQAttention(24, 6, 2, None, spec(4), rng=Rng(0))
    assert (
        c.d_head == 4
        and c.q_proj.weight.shape == (24, 24)
        and c.window is None
        and c.sinks is None
    )


def test_validation():
    # WHY: heads that do not group, a d_head that cannot be inferred, a zero
    #      window, a rotary width larger than the head, a wrong input width,
    #      positions of the wrong shape, and a mask that does not broadcast
    #      are wiring bugs; fail loudly.
    # KIND: boundary
    # CATCHES: m04, m05
    # CHAPTER: L7.5 section 4, The interface
    for kw in (
        dict(n_kv_heads=4),
        dict(d_head=None, d=25),
        dict(window=0),
        dict(rope=spec(16)),
    ):
        with pytest.raises(ValueError):
            attn(0, **kw)
    a = attn(26)
    with pytest.raises(ValueError):
        a(Tensor(np.ones((1, 3, 20))), np.arange(3))
    with pytest.raises(ValueError):
        a(Tensor(inputs(27)), np.arange(5))
    with pytest.raises(ValueError):
        a(Tensor(inputs(27)), np.arange(6), np.ones((2, 6, 5), dtype=bool))
