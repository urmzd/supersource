"""Course tests for M09.6 (Python side): your tl_exp_f32, called through your
rt.01 loader, against your M02.1 exp_range_reduced and numpy.

M02.1 wrote the same algorithm in float64: round x / ln 2, reduce in two
parts, a Taylor polynomial, scale by 2^k. With degree 7 its truncation error
is below 1e-8, a tenth of a float32 ulp, so it is an oracle for the float32
kernel; numpy's float64 exp checks the oracle in the same test, so a wrong
Python and a wrong C cannot agree by accident (DESIGN 4.0, differential).
Inputs come from the frozen PCG32.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import f32_ptr, load
from tinyllm.num.taylor import exp_range_reduced

SEED = int(os.environ.get("SS_SEED", "0"))
X_OVER = np.float32(88.7228317)


def lib():
    lb = load()
    fp = ctypes.POINTER(ctypes.c_float)
    lb.declare("tl_expf", ctypes.c_float, [ctypes.c_float])
    lb.declare("tl_exp_f32", None, [fp, fp, ctypes.c_int64])
    return lb


def c_exp(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.full_like(x, np.nan)
    lib().tl_exp_f32(f32_ptr(x), f32_ptr(y), x.size)
    return y


def ulp32(r: np.ndarray) -> np.ndarray:
    """One float32 ulp at the magnitude of each float64 r (normal range)."""
    _, e = np.frexp(np.abs(r))
    return np.ldexp(1.0, e.astype(np.int64) - 1 - 23)


def test_hand_example_through_ctypes():
    # WHY: the chapter's x = 1 through your loader: the float32 nearest to e.
    #      Your M02.1 polynomial of degree 7 agrees to far better than a
    #      float32 ulp.
    # KIND: unit, smoke
    # CATCHES: s08, m001
    # CHAPTER: M09.6 section 3
    assert lib().tl_expf(1.0) == np.float32(2.7182817)
    assert abs(float(exp_range_reduced(1.0, 7)) - np.e) < 1e-8 * np.e
    assert c_exp(np.array([1.0, -1.0, 10.0], dtype=np.float32)).tolist() == [
        np.float32(2.7182817),
        np.float32(0.36787945),
        np.float32(22026.465),
    ]


def test_within_2_ulp_of_exp_range_reduced():
    # WHY: the port's accuracy against its Python source on 50000 inputs
    #      over the normal range (float32 arithmetic costs at most 2 ulp of
    #      the float64 result), with the oracle itself checked against
    #      numpy's exp to 1e-8 relative.
    # KIND: differential
    # CATCHES: s01, s02, s03, s04
    # CHAPTER: M09.6 section 2.3
    x = PCG32(SEED + 3).uniform_array((50000,), -87.3, 88.7).astype(np.float32)
    x64 = x.astype(np.float64)
    want = exp_range_reduced(x64, 7)
    ref = np.exp(x64)
    assert np.max(np.abs(want - ref) / ref) < 1e-8, "the M02.1 oracle disagrees with numpy"
    got = c_exp(x).astype(np.float64)
    err = np.abs(got - want) / ulp32(want)
    i = int(np.argmax(err))
    assert err[i] <= 2.0, f"{err[i]:.3f} ulp at x = {x[i]!r}: C {got[i]!r}, Python {want[i]!r}"


def test_edges_through_ctypes():
    # WHY: the overflow edge (k = 128, applied in two halves) and the special
    #      values reach Python unchanged: exp(88.7228317) is finite, the next
    #      float is +inf, exp(-inf) = 0, NaN stays NaN.
    # KIND: boundary
    # CATCHES: s04, s05, s07
    # CHAPTER: M09.6 section 5
    x = np.array([X_OVER, np.nextafter(X_OVER, np.float32(np.inf)), -np.inf, np.inf, np.nan, -104.0], dtype=np.float32)
    y = c_exp(x)
    assert np.isfinite(y[0]) and y[0] > 3.4e38
    assert np.isposinf(y[1]) and np.isposinf(y[3])
    assert y[2] == 0.0 and y[5] == 0.0
    assert np.isnan(y[4])
