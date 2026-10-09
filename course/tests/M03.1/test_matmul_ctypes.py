"""Course tests for M03.1 (Python side): your tl_matmul_f32, called from Python
through your rt.01 loader, against numpy.

The C side under sanitizers is test_matmul.c. Inputs come from the frozen
PCG32 (course/tests/_lib) and closeness from the frozen close.py, whose
reduction bound scales the float32 tolerances by sqrt(K) (DESIGN 5.11).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close_bounded
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import TL_EINVAL, TlError, f32_ptr, load

SEED = int(os.environ.get("SS_SEED", "0"))


def matmul(
    A: np.ndarray,
    B: np.ndarray,
    *,
    trans_b: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray | None = None,
) -> np.ndarray:
    """alpha * A @ op(B) + beta * C through tl_matmul_f32. A, B, and C may be
    row-strided views: the row stride becomes the leading dimension."""
    M, K = A.shape
    N = B.shape[0] if trans_b else B.shape[1]
    if C is None:
        C = np.zeros((M, N), dtype=np.float32)
    lda, ldb, ldc = (x.strides[0] // 4 for x in (A, B, C))
    load().tl_matmul_f32(
        f32_ptr(A),
        f32_ptr(B),
        f32_ptr(C),
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        alpha,
        beta,
        int(trans_b),
        None,
    )
    return C


def rand(rng: PCG32, shape: tuple[int, ...]) -> np.ndarray:
    return rng.uniform_array(shape, -1.0, 1.0).astype(np.float32)


def test_hand_example_through_ctypes():
    # WHY: the chapter's worked example again, this time crossing the
    #      boundary: numpy arrays in, a C loop, the answer back in Python.
    #      This is exactly how L0.0 computes its logits.
    # KIND: unit, smoke
    # CATCHES: s01, m001
    # CHAPTER: M03.1 section 3
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    assert matmul(A, B).tolist() == [[58.0, 64.0], [139.0, 154.0]]


@pytest.mark.parametrize("trans_b", [False, True])
def test_matches_numpy_across_shapes(trans_b):
    # WHY: shapes of 1 catch loops that assume at least two rows; odd sizes
    #      (7, 33, 65) catch code that silently assumes even or power-of-two
    #      sizes, which L9.1's tiling will. Compared in float64 within the
    #      frozen sqrt(K) bound, so any summation order passes and a wrong
    #      index does not.
    # KIND: differential
    # CATCHES: s02, s05, s06, m001, m002
    # CHAPTER: M03.1 section 4
    rng = PCG32(SEED, seq=31)
    for M, N, K in [
        (1, 1, 1),
        (1, 7, 3),
        (5, 1, 2),
        (3, 4, 1),
        (7, 33, 65),
        (16, 16, 16),
        (2, 65, 128),
    ]:
        A = rand(rng, (M, K))
        B = rand(rng, (N, K) if trans_b else (K, N))
        want = A.astype(np.float64) @ (B.T if trans_b else B).astype(np.float64)
        assert_close_bounded(
            matmul(A, B, trans_b=trans_b), want, k=K, msg=f"M,N,K={M},{N},{K}"
        )


def test_strided_views_use_leading_dimensions():
    # WHY: L10.0 and later the attention code multiply views into larger
    #      buffers. A row slice of a wider array has a row stride bigger than
    #      its row length; the kernel must read and write through lda, ldb,
    #      and ldc, never assume they equal K and N.
    # KIND: differential
    # CATCHES: s05, s06
    # CHAPTER: M03.1 section 3
    rng = PCG32(SEED, seq=32)
    big_a, big_b, big_c = rand(rng, (6, 11)), rand(rng, (9, 10)), rand(rng, (6, 12))
    A, B = big_a[:, 2:7], big_b[:5, 1:4]  # 6 x 5 and 5 x 3, rows 11 and 10 floats apart
    C = big_c[:, 3:6]  # 6 x 3 inside a 6 x 12 buffer
    before = big_c.copy()
    matmul(A, B, C=C)
    want = A.astype(np.float64) @ B.astype(np.float64)
    assert_close_bounded(C, want, k=5)
    outside = np.ones(big_c.shape, dtype=bool)
    outside[:, 3:6] = False
    assert np.array_equal(big_c[outside], before[outside]), "wrote outside C's columns"


def test_alpha_beta_property():
    # WHY: scaling by 2 is exact in binary floating point, so
    #      matmul(alpha=2) must equal 2 * matmul(alpha=1) bit for bit, and with
    #      beta = 1 the old C must be added once, whatever order the sum uses.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: M03.1 section 2
    rng = PCG32(SEED, seq=33)
    for _ in range(10):
        M, N, K = 1 + rng.below(9), 1 + rng.below(9), 1 + rng.below(40)
        A, B = rand(rng, (M, K)), rand(rng, (K, N))
        one = matmul(A, B)
        assert np.array_equal(matmul(A, B, alpha=2.0), 2 * one)
        C0 = rand(rng, (M, N))
        acc = matmul(A, B, beta=1.0, C=C0.copy())
        assert_close_bounded(acc, one.astype(np.float64) + C0, k=K + 1)


def test_einval_raises_through_the_loader():
    # WHY: a bad leading dimension must come back to Python as TlError with
    #      TL_EINVAL and the C message, not as a segfault or a silently wrong
    #      array: the ABI's error path, end to end.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M03.1 section 4
    A = np.ones((2, 3), dtype=np.float32)
    B = np.ones((3, 2), dtype=np.float32)
    C = np.full((2, 2), 7.0, dtype=np.float32)
    with pytest.raises(TlError) as e:
        load().tl_matmul_f32(
            f32_ptr(A), f32_ptr(B), f32_ptr(C), 2, 2, 3, 2, 2, 2, 1.0, 0.0, 0, None
        )
    assert e.value.status == TL_EINVAL
    assert "tl_matmul_f32" in e.value.message
    assert (C == 7.0).all(), "C changed although the call was rejected"
