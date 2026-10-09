"""Course tests for L9.6 (Python side): your C elementwise kernels, called
through your rt.01 loader, against your own Python modules: L7.1's RMSNorm
and L7.3's RoPE (P6: Python is the specification), and numpy for the rest.

The C side under sanitizers is test_elementwise.c. Inputs come from the
frozen PCG32 and closeness from the frozen close.py.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32

from tinyllm.autograd.tensor import Tensor
from tinyllm.ffi.libtinyllm import f32_ptr, load
from tinyllm.modern.norm import RMSNorm
from tinyllm.modern.rope import apply_rope, rope_cos_sin

SEED = int(os.environ.get("SS_SEED", "0"))
FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int32)
I64 = ctypes.c_int64


def lib():
    lb = load()
    lb.declare("tl_rmsnorm_f32", None, [FP, FP, FP, I64, I64, ctypes.c_float])
    lb.declare(
        "tl_rope_f32",
        None,
        [FP, IP, I64, I64, I64, I64, FP, ctypes.c_float, ctypes.c_int],
    )
    lb.declare("tl_silu_mul_f32", None, [FP, FP, FP, I64])
    lb.declare("tl_embedding_f32", None, [FP, IP, FP, I64, I64])
    lb.declare("tl_add_f32", None, [FP, FP, FP, I64])
    lb.declare("tl_argmax_f32", ctypes.c_int32, [FP, I64])
    return lb


def i32_ptr(a: np.ndarray):
    assert a.dtype == np.int32 and a.flags.c_contiguous
    return a.ctypes.data_as(IP)


def c_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.empty_like(x)
    rows, d = x.reshape(-1, x.shape[-1]).shape
    lib().tl_rmsnorm_f32(f32_ptr(x), f32_ptr(w), f32_ptr(y), rows, d, eps)
    return y


def c_rope(x, pos, inv_freq, d_rot, scaling, layout) -> np.ndarray:
    T, H, D = x.shape
    out = np.ascontiguousarray(x, dtype=np.float32).copy()
    p = np.ascontiguousarray(pos, dtype=np.int32)
    f = np.ascontiguousarray(inv_freq, dtype=np.float32)
    lib().tl_rope_f32(
        f32_ptr(out), i32_ptr(p), T, H, D, d_rot, f32_ptr(f), scaling, layout
    )
    return out


def test_hand_example_through_ctypes():
    # WHY: the worked example across the boundary: RMSNorm of [3, 4] with
    #      w = [1, 0.5] is [0.8485281, 0.5656854], and argmax of
    #      [2, 7, 7, -1] is 1.
    # KIND: unit, smoke
    # CATCHES: s03, s12
    # CHAPTER: L9.6 section 3
    y = c_rmsnorm(np.array([[3, 4]]), np.array([1, 0.5], dtype=np.float32), 0.0)
    assert_close(y, [[0.848528137423857, 0.565685424949238]], rtol=1e-6, atol=1e-7)
    x = np.array([2, 7, 7, -1], dtype=np.float32)
    assert lib().tl_argmax_f32(f32_ptr(x), 4) == 1


def test_rmsnorm_matches_your_l7_1():
    # WHY: P6: your L7.1 RMSNorm (Llama's form, eps inside the root) is the
    #      specification. SmolLM2's width d = 576, rows at very different
    #      scales, eps 1e-5, a learned gain.
    # KIND: differential
    # CATCHES: s01, s02, s03, m01
    # CHAPTER: L9.6 section 4
    rng = PCG32(SEED, seq=96)
    d = 576
    x = rng.uniform_array((6, d), -2.0, 2.0).astype(np.float32)
    x *= np.array([1e-3, 0.1, 1, 3, 30, 1e3], dtype=np.float32)[:, None]
    w = rng.uniform_array((d,), 0.5, 1.5).astype(np.float32)
    norm = RMSNorm(d, eps=1e-5)
    norm.weight.data = w.copy()
    want = norm.forward(Tensor(x)).data
    assert_close(c_rmsnorm(x, w, 1e-5), want, rtol=1e-5 * 24, atol=1e-6 * 24)


@pytest.mark.parametrize("layout", ["half", "interleaved"])
def test_rope_matches_your_l7_3(layout):
    # WHY: P6: your L7.3 rope_cos_sin and apply_rope are the specification.
    #      SmolLM2's head width 64, full and partial rotary, positions up to
    #      8191 (where the angle's float32 rounding matters), and a YaRN
    #      style attention scaling.
    # KIND: differential
    # CATCHES: s04, s05, s06, s07, s08
    # CHAPTER: L9.6 section 4
    rng = PCG32(SEED, seq=97)
    T, H, D = 5, 3, 64
    x = rng.uniform_array((T, H, D), -1.0, 1.0).astype(np.float32)
    pos = np.array([0, 1, 17, 1000, 8191], dtype=np.int32)
    for d_rot, scaling in ((64, 1.0), (32, 1.2)):
        inv_freq = (10000.0 ** (-np.arange(0, d_rot, 2) / d_rot)).astype(np.float32)
        cos, sin = rope_cos_sin(pos, inv_freq, scaling)
        # apply_rope wants [..., T, dh]: put heads first, tokens second.
        want = apply_rope(
            Tensor(x.transpose(1, 0, 2)), cos, sin, layout, rotary_dim=d_rot
        ).data.transpose(1, 0, 2)
        got = c_rope(x, pos, inv_freq, d_rot, scaling, 0 if layout == "half" else 1)
        assert_close(got, want, rtol=1e-5, atol=1e-5, msg=f"d_rot={d_rot}")


def test_silu_mul_matches_numpy():
    # WHY: silu(g) * up over the whole useful range of g, against a float64
    #      evaluation of g / (1 + e^-g) * up.
    # KIND: differential
    # CATCHES: s09, s10
    # CHAPTER: L9.6 section 4
    rng = PCG32(SEED, seq=98)
    g = rng.uniform_array((4096,), -90.0, 90.0).astype(np.float32)
    up = rng.uniform_array((4096,), -3.0, 3.0).astype(np.float32)
    y = np.empty_like(g)
    lib().tl_silu_mul_f32(f32_ptr(g), f32_ptr(up), f32_ptr(y), g.size)
    g64 = g.astype(np.float64)
    want = g64 / (1.0 + np.exp(-g64)) * up
    assert_close(y, want, rtol=2e-5, atol=1e-6)


def test_embedding_and_add_match_numpy():
    # WHY: the embedding gather against numpy's take on a vocabulary-sized
    #      table, then the residual add of two such rows.
    # KIND: differential
    # CATCHES: s11, m03
    # CHAPTER: L9.6 section 4
    rng = PCG32(SEED, seq=99)
    table = rng.uniform_array((1000, 64), -1.0, 1.0).astype(np.float32)
    ids = np.array([999, 0, 5, 5, 123], dtype=np.int32)
    out = np.zeros((5, 64), dtype=np.float32)
    lib().tl_embedding_f32(f32_ptr(table), i32_ptr(ids), f32_ptr(out), 5, 64)
    assert np.array_equal(out, table[ids])
    y = np.empty_like(out)
    lib().tl_add_f32(f32_ptr(out), f32_ptr(table[:5].copy()), f32_ptr(y), out.size)
    assert np.array_equal(y, out + table[:5])


def test_argmax_matches_numpy():
    # WHY: greedy decoding on vocabulary-sized rows with many ties (values
    #      on a coarse grid) and scattered NaN, against numpy's nanargmax,
    #      which also returns the first maximum.
    # KIND: differential
    # CATCHES: s12, s13
    # CHAPTER: L9.6 section 4
    rng = PCG32(SEED, seq=100)
    for _ in range(20):
        x = np.floor(rng.uniform_array((49152,), 0.0, 8.0)).astype(np.float32)
        x[rng.below(49152)] = np.nan
        x[0] = np.nan
        assert lib().tl_argmax_f32(f32_ptr(x), x.size) == int(np.nanargmax(x))
