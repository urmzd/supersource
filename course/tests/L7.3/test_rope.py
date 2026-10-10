"""Course tests for L7.3: rotary position embeddings (tinyllm/modern/rope.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.3), and the chapter section it comes
from.

The worked example of the chapter (section 3): d = 4, inv_freq = (1, 0.01),
position 1, x = (1, 0, 0, 1). Half layout pairs (x0, x2) = (1, 0) and
(x1, x3) = (0, 1); rotated by 1 and 0.01 radians they become
(cos 1, sin 1) and (-sin 0.01, cos 0.01), so the output is
(0.540302, -0.010000, 0.841471, 0.999950). The interleaved layout pairs
(x0, x1) = (1, 0) and (x2, x3) = (0, 1): (0.540302, 0.841471, -0.010000,
0.999950).

The golden fixture (course/fixtures/L7.3/rope_hf.npz) holds transformers
5.19.0 LlamaRotaryEmbedding tables, apply_rotary_pos_emb (half), GPT-NeoX's
partial rotary, and Meta's complex-number apply_rotary_emb (interleaved),
float32, from course/oracle/L7.3/rope_hf.py.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.rope import RopeSpec, apply_rope, rope_cos_sin
from tinyllm.num.rotation import rotate_pairs

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.3", "rope_hf.npz")


def ladder(r: int, base: float = 10000.0) -> np.ndarray:
    return base ** (-np.arange(0, r, 2, dtype=np.float64) / r)


def half_from_interleaved(x: np.ndarray) -> np.ndarray:
    """Reorder the last axis from (a0, b0, a1, b1, ...) to (a0, a1, ..., b0, b1, ...)."""
    return np.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)


# --- the worked example ------------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example in both layouts: the same two
    #      rotations (1 rad and 0.01 rad) applied to different pairs of
    #      entries.
    # KIND: unit
    # CATCHES: s01, s02, s07
    # CHAPTER: L7.3 section 3, Worked example by hand
    cos, sin = rope_cos_sin(np.array([1]), np.array([1.0, 0.01]))
    assert cos.dtype == np.float32 and cos.shape == (1, 2)
    x = Tensor([[1.0, 0.0, 0.0, 1.0]])
    c1, s1, c2, s2 = np.cos(1.0), np.sin(1.0), np.cos(0.01), np.sin(0.01)
    assert_close(
        apply_rope(x, cos, sin, "half").data, [[c1, -s2, s1, c2]], dtype="float32"
    )
    assert_close(
        apply_rope(x, cos, sin, "interleaved").data,
        [[c1, s1, -s2, c2]],
        dtype="float32",
    )


# --- against Hugging Face and Meta --------------------------------------------------------------


def test_cos_sin_table_golden():
    # WHY: the angle is one float32 product, position times inv_freq, as in
    #      HF's rotary embedding; at position 65535 a different rounding
    #      moves the angle visibly. The table must match up to the last bit
    #      or two of cos and sin.
    # KIND: golden
    # CATCHES: s04
    # CHAPTER: L7.3 section 2.3, Computing the angles
    f = np.load(FIX)
    cos, sin = rope_cos_sin(f["table.positions"], f["table.inv_freq"])
    assert cos.shape == (9, 8) and cos.dtype == np.float32 and sin.dtype == np.float32
    assert_close(cos, f["table.cos"], rtol=0, atol=2e-6)
    assert_close(sin, f["table.sin"], rtol=0, atol=2e-6)


def test_cos_sin_attention_scaling_golden():
    # WHY: YaRN (L7.4) multiplies cos and sin by its attention scaling, so
    #      both q and k grow by that factor and every score by its square;
    #      the table must carry it on both.
    # KIND: golden
    # CATCHES: s05
    # CHAPTER: L7.3 section 2.3, Computing the angles
    f = np.load(FIX)
    s = float(f["table-yarn.attention_scaling"])
    assert s > 1.1
    cos, sin = rope_cos_sin(f["table.positions"], f["table-yarn.inv_freq"], s)
    assert_close(cos, f["table-yarn.cos"], rtol=0, atol=2e-6)
    assert_close(sin, f["table-yarn.sin"], rtol=0, atol=2e-6)


def test_half_layout_golden():
    # WHY: HF's apply_rotary_pos_emb pairs x[i] with x[i + d/2] (rotate_half);
    #      SmolLM2 and every HF Llama checkpoint expect it. Per-row
    #      positions (row 1 starts at 1000) pass as cos[:, None].
    # KIND: golden
    # CATCHES: s01, s02
    # CHAPTER: L7.3 section 2.2, Two layouts
    f = np.load(FIX)
    cos, sin = rope_cos_sin(f["half.positions"], f["table.inv_freq"])
    for name in ("q", "k"):
        got = apply_rope(Tensor(f[f"half.{name}"]), cos[:, None], sin[:, None], "half")
        assert_close(got.data, f[f"half.{name}_out"], rtol=1e-5, atol=1e-5, msg=name)


def test_interleaved_layout_golden():
    # WHY: Meta's reference code views (x[2i], x[2i+1]) as one complex
    #      number and multiplies by e^{i angle}; the interleaved layout must
    #      reproduce it, and the output stays interleaved.
    # KIND: golden
    # CATCHES: s02, s07
    # CHAPTER: L7.3 section 2.2, Two layouts
    f = np.load(FIX)
    cos, sin = rope_cos_sin(np.arange(5), f["table.inv_freq"])
    got = apply_rope(Tensor(f["interleaved.q"]), cos, sin, "interleaved")
    assert_close(got.data, f["interleaved.q_out"], rtol=1e-5, atol=1e-5)


def test_partial_rotary_golden():
    # WHY: GPT-NeoX and Phi rotate only the first rotary_dim entries of each
    #      head and pass the rest through unchanged (GPT-NeoX's
    #      apply_rotary_pos_emb, rotary_dim 8 of 16).
    # KIND: golden
    # CATCHES: s03
    # CHAPTER: L7.3 section 2.4, Partial rotary
    f = np.load(FIX)
    cos, sin = rope_cos_sin(np.arange(5), ladder(8).astype(np.float32))
    for name in ("q", "k"):
        got = apply_rope(Tensor(f[f"partial.{name}"]), cos, sin, "half", rotary_dim=8)
        assert_close(got.data, f[f"partial.{name}_out"], rtol=1e-5, atol=1e-5, msg=name)
        assert np.array_equal(got.data[..., 8:], f[f"partial.{name}"][..., 8:])


# --- properties ------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["half", "interleaved"])
def test_scores_depend_only_on_relative_position(layout):
    # WHY: the point of RoPE: <R_m q, R_n k> = <R_{m+s} q, R_{n+s} k> for any
    #      shift s, so attention scores see m - n only. Checked over random
    #      vectors and shifts (float64 data, float32 tables).
    # KIND: property
    # CATCHES: s04
    # CHAPTER: L7.3 section 2.1, Rotations and relative position
    g = PCG32(seed=1)
    inv = ladder(16)
    for _ in range(5):
        q, k = g.normal_array((16,)), g.normal_array((16,))
        m, n, s = g.below(40), g.below(40), g.below(40)
        cos, sin = rope_cos_sin(np.array([m, n, m + s, n + s]), inv)
        qs = apply_rope(
            Tensor(np.stack([q, q]), dtype=np.float64), cos[[0, 2]], sin[[0, 2]], layout
        ).data
        ks = apply_rope(
            Tensor(np.stack([k, k]), dtype=np.float64), cos[[1, 3]], sin[[1, 3]], layout
        ).data
        assert_close(qs[0] @ ks[0], qs[1] @ ks[1], rtol=1e-5, atol=1e-5)


def test_rotation_preserves_length_and_position_zero_is_identity():
    # WHY: a rotation changes direction, never length: |RoPE(x)| = |x| for
    #      every vector, and at position 0 every angle is 0, so RoPE is the
    #      identity there (also for partial rotary).
    # KIND: property
    # CATCHES: s03, s06, s07
    # CHAPTER: L7.3 section 2.1, Rotations and relative position
    g = PCG32(seed=2)
    x = g.normal_array((3, 7, 12))
    cos, sin = rope_cos_sin(np.arange(0, 70000, 10000), ladder(12))
    for layout in ("half", "interleaved"):
        y = apply_rope(Tensor(x, dtype=np.float64), cos, sin, layout).data
        assert_close(
            np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-6, atol=1e-6
        )
        assert_close(y[:, 0], x[:, 0], rtol=1e-12, atol=1e-12)
    c8, s8 = rope_cos_sin(np.zeros(7, dtype=np.int64), ladder(8))
    assert_close(
        apply_rope(Tensor(x, dtype=np.float64), c8, s8, "half", 8).data,
        x,
        rtol=1e-12,
        atol=1e-12,
    )


def test_layouts_are_a_permutation_of_each_other():
    # WHY: the two layouts hold the same rotation on a reordered vector:
    #      interleaved(x) reordered equals half(x reordered). This is the
    #      permutation HF applies to the rows of q_proj and k_proj when it
    #      converts Meta's checkpoints.
    # KIND: property
    # CATCHES: s01, s07
    # CHAPTER: L7.3 section 2.2, Two layouts
    x = PCG32(seed=3).normal_array((2, 5, 8))
    cos, sin = rope_cos_sin(np.arange(5), ladder(8))
    inter = apply_rope(Tensor(x, dtype=np.float64), cos, sin, "interleaved").data
    half = apply_rope(
        Tensor(half_from_interleaved(x), dtype=np.float64), cos, sin, "half"
    ).data
    assert_close(half_from_interleaved(inter), half, rtol=1e-12, atol=1e-12)


def test_interleaved_matches_rotate_pairs():
    # WHY: M00.2's rotate_pairs is the interleaved rotation in numpy; the
    #      autograd version must agree with it on the same angles.
    # KIND: differential
    # CATCHES: s07
    # CHAPTER: L7.3 section 2.2, Two layouts
    x = PCG32(seed=4).normal_array((4, 10))
    pos, inv = np.arange(4), ladder(10)
    cos, sin = rope_cos_sin(pos, inv)
    theta = np.arctan2(sin.astype(np.float64), cos.astype(np.float64))
    got = apply_rope(Tensor(x, dtype=np.float64), cos, sin, "interleaved").data
    assert_close(got, rotate_pairs(x, theta), rtol=1e-6, atol=1e-6)


def test_gradcheck_both_layouts_and_partial():
    # WHY: RoPE sits between the projections and the scores, so the q_proj
    #      and k_proj weights learn through it; checked against the frozen
    #      central differences in float64, for both layouts and partial
    #      rotary.
    # KIND: gradcheck
    # CATCHES: s06
    # CHAPTER: L7.3 section 2.5, Backward
    g = PCG32(seed=5)
    x0 = g.normal_array((2, 3, 8))
    gy = g.normal_array((2, 3, 8))
    for layout, r in (("half", 8), ("interleaved", 8), ("half", 4), ("interleaved", 6)):
        cos, sin = rope_cos_sin(np.array([0, 3, 11]), ladder(r))

        def f(x):
            return float(
                np.sum(
                    apply_rope(Tensor(x, dtype=np.float64), cos, sin, layout, r).data
                    * gy
                )
            )

        x = Tensor(x0, requires_grad=True, dtype=np.float64)
        F.sum(apply_rope(x, cos, sin, layout, r) * gy).backward()
        gradcheck(f, [x0], [x.grad], names=[f"x ({layout}, r={r})"])


def test_gradient_is_the_inverse_rotation():
    # WHY: RoPE is linear and orthogonal in x, so its backward is the
    #      inverse rotation of the upstream gradient: rotating g by the
    #      negative angles (sin -> -sin).
    # KIND: property
    # CATCHES: s06, s07
    # CHAPTER: L7.3 section 2.5, Backward
    g = PCG32(seed=6)
    x0, gy = g.normal_array((4, 6)), g.normal_array((4, 6))
    cos, sin = rope_cos_sin(np.arange(4), ladder(6))
    for layout in ("half", "interleaved"):
        x = Tensor(x0, requires_grad=True, dtype=np.float64)
        F.sum(apply_rope(x, cos, sin, layout) * gy).backward()
        back = apply_rope(Tensor(gy, dtype=np.float64), cos, -sin, layout).data
        assert_close(x.grad, back, rtol=1e-10, atol=1e-12)


def test_rope_spec_fields():
    # WHY: attention layers (L7.5, L7.6) receive how to rotate as one
    #      RopeSpec; its four fields are the contract between L7.4's
    #      frequencies and the attention code.
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: L7.3 section 4, The interface
    spec = RopeSpec(
        inv_freq=ladder(4), attention_scaling=1.0, layout="half", rotary_dim=4
    )
    assert (
        spec.rotary_dim == 4 and spec.layout == "half" and spec.attention_scaling == 1.0
    )
    cos, sin = rope_cos_sin(np.arange(3), spec.inv_freq, spec.attention_scaling)
    x = PCG32(seed=7).normal_array((3, 6))
    y = apply_rope(Tensor(x), cos, sin, spec.layout, spec.rotary_dim).data
    assert y.shape == (3, 6) and np.array_equal(y[:, 4:], x[:, 4:].astype(np.float32))


def test_validation():
    # WHY: an odd rotary dimension, one larger than the head, a cos table of
    #      the wrong width, an unknown layout, and negative or fractional
    #      positions are wiring bugs; fail loudly.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: L7.3 section 4, The interface
    x = Tensor(np.ones((3, 8)))
    cos, sin = rope_cos_sin(np.arange(3), ladder(8))
    for kw in (
        {"rotary_dim": 7},
        {"rotary_dim": 10},
        {"rotary_dim": 0},
        {"layout": "neox"},
    ):
        with pytest.raises(ValueError):
            apply_rope(x, cos, sin, **kw)
    with pytest.raises(ValueError):
        apply_rope(x, cos[:, :3], sin[:, :3])
    with pytest.raises(ValueError):
        apply_rope(x, cos[:2], sin[:2])
    with pytest.raises(ValueError):
        rope_cos_sin(np.array([0, -1]), ladder(8))
    with pytest.raises(ValueError):
        rope_cos_sin(np.array([0.5]), ladder(8))
    with pytest.raises(ValueError):
        rope_cos_sin(np.arange(3), np.ones((2, 2)))
