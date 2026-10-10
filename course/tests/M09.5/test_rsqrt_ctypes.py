"""Course tests for M09.5 (Python side): your tl_rsqrt_f32, called through your
rt.01 loader, against your M01.2 rsqrt_newton, bit for bit.

The C side under sanitizers is test_rsqrt.c, which checks the contract's
accuracy against double precision. Here the claim is stronger and narrower:
the C kernel is the Python algorithm (P6, Python is the semantic source of
truth), so on the same inputs it returns the same bits. The algorithm is the
chapter's: the bit-pattern guess, two float32 Newton steps, one float64 step.
Inputs come from the frozen PCG32.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import f32_ptr, load
from tinyllm.num.newton import rsqrt_newton

SEED = int(os.environ.get("SS_SEED", "0"))
MAGIC = 0x5F3759DF


def lib():
    lb = load()
    fp = ctypes.POINTER(ctypes.c_float)
    lb.declare("tl_rsqrtf", ctypes.c_float, [ctypes.c_float])
    lb.declare("tl_rsqrt_f32", None, [fp, fp, ctypes.c_int64])
    return lb


def c_rsqrt(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.full_like(x, np.nan)
    lib().tl_rsqrt_f32(f32_ptr(x), f32_ptr(y), x.size)
    return y


def guess(x: np.ndarray) -> np.ndarray:
    """Step 1 of the chapter: bits(MAGIC - (bits(x) >> 1)), as float32."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (np.uint32(MAGIC) - (u >> np.uint32(1))).view(np.float32)


def python_rsqrt(x: np.ndarray) -> np.ndarray:
    """Steps 2 and 3 with your M01.2 rsqrt_newton: two float32 steps, then
    one float64 step on that result, rounded to float32."""
    x32 = np.asarray(x, dtype=np.float32)
    y2 = rsqrt_newton(x32, guess(x32), 2)
    assert y2.dtype == np.float32
    y3 = rsqrt_newton(x32.astype(np.float64), y2.astype(np.float64), 1)
    return y3.astype(np.float32)


def normals(n: int, seed: int) -> np.ndarray:
    """n positive normal float32 values, log-uniform over 2^-125 .. 2^126."""
    rng = PCG32(seed)
    m = rng.uniform_array((n,), 1.0, 2.0)
    e = np.array([rng.below(251) for _ in range(n)]) - 125
    return np.ldexp(m, e).astype(np.float32)


def test_hand_example_through_ctypes():
    # WHY: the chapter's x = 4 reaches C through your loader and comes back
    #      as 0.5 exactly; your rsqrt_newton, started from the same guess,
    #      makes the same three steps.
    # KIND: unit, smoke
    # CATCHES: s01, s02, s08
    # CHAPTER: M09.5 section 3
    assert lib().tl_rsqrtf(4.0) == 0.5
    x = np.array([4.0], dtype=np.float32)
    assert guess(x).view(np.uint32)[0] == 0x3EF759DF
    y2 = rsqrt_newton(x, guess(x), 2)
    assert y2[0] == np.float32(0.49999782)
    assert python_rsqrt(x)[0] == np.float32(0.5) == c_rsqrt(x)[0]


def test_matches_rsqrt_newton_bit_for_bit():
    # WHY: the kernel is a port of M01.2's iteration. Same guess, same
    #      products in the same order, same precision per step, so the same
    #      bits on 20000 inputs spread over every binade. A third float32
    #      step instead of the float64 one stays near 2 ulp yet changes
    #      bits here.
    # KIND: differential
    # CATCHES: s01, s02, s04, s08, m001, m002
    # CHAPTER: M09.5 section 2.3
    x = normals(20000, SEED + 1)
    want, got = python_rsqrt(x), c_rsqrt(x)
    bad = np.flatnonzero(want.view(np.uint32) != got.view(np.uint32))
    assert bad.size == 0, (
        f"{bad.size} of {x.size} differ; first at x = {x[bad[0]]!r}: "
        f"C {got[bad[0]]!r}, rsqrt_newton {want[bad[0]]!r}"
    )


def test_two_float_steps_are_not_enough():
    # WHY: the chapter's convergence claim, measured through your Python: the
    #      guess is within 3.5%, one float32 step within 1.8e-3, two within
    #      5e-6 (e' = 1.5 e^2 - 0.5 e^3 per step, plus float32 rounding of a
    #      few units of 2^-24), and that is
    #      still 40 to 70 float32 ulps, which is why the C kernel takes a
    #      third, float64 step. The C result must beat it by a wide margin.
    # KIND: property
    # CATCHES: s01, s04
    # CHAPTER: M09.5 section 2.2
    x = normals(5000, SEED + 2)
    exact = 1.0 / np.sqrt(x.astype(np.float64))
    rel = [
        np.max(np.abs(rsqrt_newton(x, guess(x), k).astype(np.float64) - exact) / exact)
        for k in (0, 1, 2)
    ]
    assert rel[0] < 0.035 and rel[1] < 1.8e-3 and rel[2] < 5e-6
    rounding = 4 * 2.0**-24
    assert (
        rel[1] <= 1.5 * rel[0] ** 2 + rounding
        and rel[2] <= 1.5 * rel[1] ** 2 + rounding
    )
    c_rel = np.max(np.abs(c_rsqrt(x).astype(np.float64) - exact) / exact)
    assert c_rel < 1.2e-7, f"C relative error {c_rel:.3g}: the float64 step is missing"
