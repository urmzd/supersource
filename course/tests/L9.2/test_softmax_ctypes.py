"""Course tests for L9.2 (Python side): your C softmax kernels, called through
your rt.01 loader, against your own M09.2 softmax (P6: the Python version is
the specification).

The C side under sanitizers is test_softmax.c. Inputs come from the frozen
PCG32 and closeness from the frozen close.py.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import STATUS, TL_EINVAL, TlError, f32_ptr, load
from tinyllm.num.stable import softmax as py_softmax

SEED = int(os.environ.get("SS_SEED", "0"))
FP = ctypes.POINTER(ctypes.c_float)
KERNELS = ["tl_softmax_f32", "tl_softmax_online_f32"]


def lib():
    lb = load()
    for name in KERNELS:
        lb.declare(name, STATUS, [FP, FP, ctypes.c_int64, ctypes.c_int64])
    return lb


def c_softmax(name: str, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.empty_like(x) if out is None else out
    rows, cols = x.shape
    getattr(lib(), name)(f32_ptr(x), f32_ptr(y), rows, cols)
    return y


@pytest.mark.parametrize("name", KERNELS)
def test_hand_example_through_ctypes(name):
    # WHY: the worked example across the boundary: [1, 2, 3] gives
    #      [0.0900306, 0.2447285, 0.6652410] from both kernels.
    # KIND: unit, smoke
    # CATCHES: s01, s02
    # CHAPTER: L9.2 section 3
    y = c_softmax(name, np.array([[1, 2, 3]], dtype=np.float32))
    assert_close(
        y, np.array([[0.09003057317038046, 0.24472847105479764, 0.6652409557748219]])
    , rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", KERNELS)
def test_matches_your_m09_2_softmax(name):
    # WHY: P6: your M09.2 softmax is the specification. Rows of 1 to 4096
    #      columns (a vocabulary row, an attention row) at scales from 0.01
    #      to 1e4, plus rows with masked -inf entries, agree within the
    #      float32 tolerance of close.py.
    # KIND: differential
    # CATCHES: s01, s02, s03, s05, s06, s08, m01, m02
    # CHAPTER: L9.2 section 4
    rng = PCG32(SEED, seq=92)
    for cols in (1, 2, 7, 64, 333, 4096):
        for scale in (0.01, 1.0, 30.0, 1e4):
            x = (rng.uniform_array((5, cols), -1.0, 1.0) * scale).astype(np.float32)
            if cols > 2:
                x[1, ::3] = -np.inf
            want = py_softmax(x.astype(np.float64), axis=-1)
            assert_close(
                c_softmax(name, x), want, dtype="float32", msg=f"{name} cols={cols} scale={scale}"
            )


@pytest.mark.parametrize("name", KERNELS)
def test_in_place_through_ctypes(name):
    # WHY: the same buffer as input and output, as the Python backend
    #      (L9.7) passes a scores array to normalize in place.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: L9.2 section 4
    rng = PCG32(SEED, seq=93)
    x = rng.uniform_array((3, 50), -4.0, 4.0).astype(np.float32)
    want = py_softmax(x.astype(np.float64), axis=-1)
    buf = x.copy()
    c_softmax(name, buf, out=buf)
    assert_close(buf, want, dtype="float32")


def test_einval_raises_through_the_loader():
    # WHY: a NULL output with rows * cols > 0 must come back to Python as
    #      TlError with TL_EINVAL, not a segfault.
    # KIND: boundary
    # CATCHES: s12
    # CHAPTER: L9.2 section 4
    x = np.zeros((2, 3), dtype=np.float32)
    for name in KERNELS:
        with pytest.raises(TlError) as e:
            getattr(lib(), name)(f32_ptr(x), None, 2, 3)
        assert e.value.status == TL_EINVAL
