"""Course tests for L9.1 (Python side): your tiled tl_matmul_f32 through your
rt.01 loader, against numpy in float64, on the shapes that cross tile edges,
on strided views, and for batch invariance through the frozen invariance
helper.

The C side under sanitizers is test_matmul_tiled.c. Inputs come from the
frozen PCG32 and closeness from the frozen close.py, whose reduction bound
scales the float32 tolerances by sqrt(K).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close_bounded
from _lib.invariance import assert_batch_invariant
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import f32_ptr, load

SEED = int(os.environ.get("SS_SEED", "0"))


def matmul(A, B, *, trans_b=False, alpha=1.0, beta=0.0, C=None):
    M, K = A.shape
    N = B.shape[0] if trans_b else B.shape[1]
    if C is None:
        C = np.zeros((M, N), dtype=np.float32)
    lda, ldb, ldc = (x.strides[0] // 4 for x in (A, B, C))
    load().tl_matmul_f32(
        f32_ptr(A), f32_ptr(B), f32_ptr(C), M, N, K, lda, ldb, ldc, alpha, beta,
        int(trans_b), None,
    )
    return C


def rand(rng: PCG32, shape) -> np.ndarray:
    return rng.uniform_array(shape, -1.0, 1.0).astype(np.float32)


def test_hand_example_through_ctypes():
    # WHY: the worked example across the boundary: the same four numbers
    #      M03.1 produced, now from packed, padded micro-tiles.
    # KIND: unit, smoke
    # CATCHES: m01
    # CHAPTER: L9.1 section 3
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    assert matmul(A, B).tolist() == [[58.0, 64.0], [139.0, 154.0]]


@pytest.mark.parametrize("trans_b", [False, True])
def test_matches_numpy_across_tile_edges(trans_b):
    # WHY: sizes from 1 to 257 around every tile constant (4, 16, 64, 128),
    #      both trans_b settings, against float64 within the frozen sqrt(K)
    #      bound: any correct blocking passes, any lost or doubled block
    #      fails.
    # KIND: differential
    # CATCHES: s01, s03, s05, m01, m03
    # CHAPTER: L9.1 section 4
    rng = PCG32(SEED, seq=91)
    for M, N, K in [(1, 1, 1), (3, 17, 129), (4, 16, 128), (63, 130, 5), (2, 257, 257), (65, 3, 200)]:
        A = rand(rng, (M, K))
        B = rand(rng, (N, K) if trans_b else (K, N))
        want = A.astype(np.float64) @ (B.T if trans_b else B).astype(np.float64)
        assert_close_bounded(matmul(A, B, trans_b=trans_b), want, k=K, msg=f"{M},{N},{K}")


def test_strided_views_with_alpha_beta():
    # WHY: views into wider buffers (one head's slice of a fused projection)
    #      with alpha = 2 and beta = 0.5, K = 300 across three k-blocks:
    #      packing must read through lda and ldb, the store must write
    #      through ldc, and beta must apply once.
    # KIND: differential
    # CATCHES: s01, s04, s06
    # CHAPTER: L9.1 section 4
    rng = PCG32(SEED, seq=92)
    big_a, big_b, big_c = rand(rng, (9, 310)), rand(rng, (300, 50)), rand(rng, (9, 60))
    A, B, C = big_a[:, 5:305], big_b[:, 7:31], big_c[:, 10:34]
    C0 = C.astype(np.float64).copy()
    before = big_c.copy()
    matmul(A, B, alpha=2.0, beta=0.5, C=C)
    want = 2.0 * (A.astype(np.float64) @ B.astype(np.float64)) + 0.5 * C0
    assert_close_bounded(C, want, k=300)
    outside = np.ones(big_c.shape, dtype=bool)
    outside[:, 10:34] = False
    assert np.array_equal(big_c[outside], before[outside]), "wrote outside C's columns"


def test_batch_invariance_through_ctypes():
    # WHY: the frozen helper checks every row of a 16-row batch alone and
    #      inside batches of 1, 2, 3, 4, 5, 7, 8, 16 at every offset, bit for
    #      bit, for a Linear layer x @ W^T with K = 300: what makes batched
    #      greedy decoding equal serial decoding.
    # KIND: property
    # CATCHES: s10
    # CHAPTER: L9.1 section 2
    rng = PCG32(SEED, seq=93)
    W = rand(rng, (24, 300))
    x = rand(rng, (16, 300))
    assert_batch_invariant(lambda xb: matmul(np.ascontiguousarray(xb), W, trans_b=True), x)
