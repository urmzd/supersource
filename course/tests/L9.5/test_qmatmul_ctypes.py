"""Course tests for L9.5 (Python side): your fused int4 and int8 kernels,
called through your rt.01 loader, against "dequantize, then numpy".

The weights are packed here in numpy by the byte layout of
course/contracts/formats/safetensors.md (two signed nibbles per byte, low
nibble first; f16 scales per group), which is the layout L8.5's quantizer
writes and L10.1's runner reads. The C side under sanitizers is
test_qmatmul.c. Inputs come from the frozen PCG32 and closeness from the
frozen close.py. One test quantizes with your own L8.5 quantizer (P6: the
Python side is the specification of what the kernel consumes).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest
from _lib.close import assert_close_bounded
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import STATUS, TL_ESHAPE, TlError, f32_ptr, load
from tinyllm.infer.quant import (
    dequantize,
    quantize_int4_group,
    quantize_int8_per_channel,
)

SEED = int(os.environ.get("SS_SEED", "0"))
FP = ctypes.POINTER(ctypes.c_float)
U8P = ctypes.POINTER(ctypes.c_uint8)
U16P = ctypes.POINTER(ctypes.c_uint16)
I8P = ctypes.POINTER(ctypes.c_int8)
I64 = ctypes.c_int64


def lib():
    lb = load()
    lb.declare(
        "tl_matmul_q4_f32",
        STATUS,
        [FP, U8P, U16P, FP, I64, I64, I64, I64, ctypes.c_void_p],
    )
    lb.declare(
        "tl_matmul_q8_f32", STATUS, [FP, I8P, FP, FP, I64, I64, I64, ctypes.c_void_p]
    )
    return lb


def pack_q4(q: np.ndarray) -> np.ndarray:
    """[N, K] ints in [-8, 7] -> [N, K/2] bytes, column 2b in the low nibble."""
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    return np.ascontiguousarray(u[:, 0::2] | (u[:, 1::2] << 4))


def q4(x, packed, scales_f16, N, K, group) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.zeros((x.shape[0], N), dtype=np.float32)
    sc = np.ascontiguousarray(scales_f16.view(np.uint16))
    lib().tl_matmul_q4_f32(
        f32_ptr(x),
        packed.ctypes.data_as(U8P),
        sc.ctypes.data_as(U16P),
        f32_ptr(y),
        x.shape[0],
        N,
        K,
        group,
        None,
    )
    return y


def test_hand_example_through_ctypes():
    # WHY: the worked example across the boundary, packed by numpy:
    #      q = [[1, -2, 3, -8], [7, 0, -1, 4]], f16 scales [[0.5, 2], [1, 0.25]],
    #      x = [1, 2, 3, 4] gives exactly [-47.5, 10.25].
    # KIND: unit, smoke
    # CATCHES: s01, s02, s03
    # CHAPTER: L9.5 section 3
    q = np.array([[1, -2, 3, -8], [7, 0, -1, 4]])
    packed = pack_q4(q)
    assert packed.tolist() == [[0xE1, 0x83], [0x07, 0x4F]]
    scales = np.array([[0.5, 2.0], [1.0, 0.25]], dtype=np.float16)
    y = q4(np.array([[1, 2, 3, 4]], dtype=np.float32), packed, scales, 2, 4, 2)
    assert y.tolist() == [[-47.5, 10.25]]


@pytest.mark.parametrize("group", [32, 64, 192])
def test_q4_matches_dequantize_then_numpy(group):
    # WHY: SmolLM2-sized projections (K = 576, a decode row and a 5-token
    #      chunk) with the common group sizes and one longer than the
    #      64-weight unpack chunk, against the dequantized
    #      weight (q times its group's f16 scale) multiplied in float64.
    # KIND: differential
    # CATCHES: s03, s04, s05, s06, m02
    # CHAPTER: L9.5 section 4
    rng = PCG32(SEED, seq=95)
    N, K = 96, 576
    q = (np.floor(rng.uniform_array((N, K), 0.0, 16.0)) - 8).astype(np.int64)
    scales = rng.uniform_array((N, K // group), 0.001, 0.05).astype(np.float16)
    w = q.astype(np.float64) * np.repeat(scales.astype(np.float64), group, axis=1)
    for M in (1, 5):
        x = rng.uniform_array((M, K), -1.0, 1.0).astype(np.float32)
        assert_close_bounded(
            q4(x, pack_q4(q), scales, N, K, group),
            x.astype(np.float64) @ w.T,
            k=K,
            msg=f"M={M}",
        )


def test_q8_matches_dequantize_then_numpy():
    # WHY: per-channel int8 (one f32 scale per output row) against the
    #      dequantized weight in float64.
    # KIND: differential
    # CATCHES: s07, s08
    # CHAPTER: L9.5 section 4
    rng = PCG32(SEED, seq=96)
    N, K, M = 40, 300, 3
    q = (np.floor(rng.uniform_array((N, K), 0.0, 255.0)) - 127).astype(np.int8)
    s = rng.uniform_array((N,), 0.001, 0.02).astype(np.float32)
    x = rng.uniform_array((M, K), -1.0, 1.0).astype(np.float32)
    y = np.zeros((M, N), dtype=np.float32)
    lib().tl_matmul_q8_f32(
        f32_ptr(x), q.ctypes.data_as(I8P), f32_ptr(s), f32_ptr(y), M, N, K, None
    )
    want = (
        x.astype(np.float64) @ (q.astype(np.float64) * s[:, None].astype(np.float64)).T
    )
    assert_close_bounded(y, want, k=K)


def test_matches_your_l8_5_quantizer():
    # WHY: P6: your L8.5 quantize_int4_group and quantize_int8_per_channel
    #      produce what L10.1's runner will feed this kernel. The fused
    #      kernels on those exact bytes must equal x @ dequantize(...)^T in
    #      float64, for a decode row and a short prefill chunk.
    # KIND: differential
    # CATCHES: s01, s02, s03, s07
    # CHAPTER: L9.5 section 6
    rng = PCG32(SEED, seq=97)
    N, K = 64, 256
    w = rng.normal_array((N, K), scale=0.05).astype(np.float32)
    q4t = quantize_int4_group(w, 32)
    w4 = dequantize(q4t).astype(np.float64)
    q8, s8 = quantize_int8_per_channel(w)
    w8 = dequantize((q8, s8)).astype(np.float64)
    for M in (1, 3):
        x = rng.uniform_array((M, K), -1.0, 1.0).astype(np.float32)
        got4 = q4(
            x,
            np.ascontiguousarray(q4t.packed, dtype=np.uint8),
            np.asarray(q4t.scales, dtype=np.float16),
            N,
            K,
            32,
        )
        assert_close_bounded(got4, x.astype(np.float64) @ w4.T, k=K, msg=f"int4 M={M}")
        y8 = np.zeros((M, N), dtype=np.float32)
        q8c = np.ascontiguousarray(q8, dtype=np.int8)
        s8c = np.ascontiguousarray(s8, dtype=np.float32)
        lib().tl_matmul_q8_f32(
            f32_ptr(x),
            q8c.ctypes.data_as(I8P),
            f32_ptr(s8c),
            f32_ptr(y8),
            M,
            N,
            K,
            None,
        )
        assert_close_bounded(y8, x.astype(np.float64) @ w8.T, k=K, msg=f"int8 M={M}")


def test_eshape_raises_through_the_loader():
    # WHY: a group that does not divide K is a shape error, and it must
    #      reach Python as TlError with TL_ESHAPE before anything is read.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: L9.5 section 4
    q = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(TlError) as e:
        q4(
            np.zeros((1, 6), dtype=np.float32),
            q,
            np.ones((2, 2), dtype=np.float16),
            2,
            6,
            4,
        )
    assert e.value.status == TL_ESHAPE
