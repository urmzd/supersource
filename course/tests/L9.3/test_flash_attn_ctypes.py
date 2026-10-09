"""Course tests for L9.3 (Python side): your FlashAttention kernel, called
through your rt.01 loader, against your own Python attention: L7.7's
windowed_attention (GQA, causal, sliding window, learned sinks, q_offset,
lse), the numpy reference the catalog names for this kernel, and L5.1's
sdpa_forward for attention without a causal mask (P6: Python is the
specification).

The C side under sanitizers is test_flash_attn.c. Inputs come from the
frozen PCG32 and closeness from the frozen close.py.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.invariance import assert_bits_equal
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import STATUS, f32_ptr, load
from tinyllm.modern.window import windowed_attention
from tinyllm.xfmr.sdpa import sdpa_forward

SEED = int(os.environ.get("SS_SEED", "0"))
FP = ctypes.POINTER(ctypes.c_float)
I64 = ctypes.c_int64


def lib():
    lb = load()
    lb.declare(
        "tl_flash_attn_fwd_f32",
        STATUS,
        [FP, FP, FP, FP, FP, I64, I64, I64, I64, I64, I64, ctypes.c_float, I64, ctypes.c_int,
         I64, FP, I64, I64, ctypes.c_void_p, ctypes.c_void_p],
    )
    return lb


def flash(q, k, v, *, scale, q_offset=0, causal=True, window=0, sinks=None, Br=0, Bc=0):
    q, k, v = (np.ascontiguousarray(a, dtype=np.float32) for a in (q, k, v))
    B, H, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    o = np.zeros_like(q)
    lse = np.zeros((B, H, Tq), dtype=np.float32)
    s = None if sinks is None else np.ascontiguousarray(sinks, dtype=np.float32)
    lib().tl_flash_attn_fwd_f32(
        f32_ptr(q), f32_ptr(k), f32_ptr(v), f32_ptr(o), f32_ptr(lse), B, H, Hkv, Tq, Tk, D,
        scale, q_offset, int(causal), window, f32_ptr(s), Br, Bc, None, None,
    )
    return o, lse


def rand(rng, shape):
    return rng.uniform_array(shape, -1.0, 1.0).astype(np.float32)


def test_hand_example_through_ctypes():
    # WHY: the worked example across the boundary: q = [1, 0] against keys
    #      [1, 0], [0, 1], [1, 1] and values [1, 2], [3, 4], [5, 6] gives
    #      o = [3, 4] and lse = 1 + ln(2 + e^-1).
    # KIND: unit, smoke
    # CATCHES: s08, m02
    # CHAPTER: L9.3 section 3
    q = np.array([[[[1, 0]]]], dtype=np.float32)
    k = np.array([[[[1, 0], [0, 1], [1, 1]]]], dtype=np.float32)
    v = np.array([[[[1, 2], [3, 4], [5, 6]]]], dtype=np.float32)
    o, lse = flash(q, k, v, scale=1.0, causal=False, Bc=2)
    assert_close(o, [[[[3.0, 4.0]]]], rtol=1e-6, atol=1e-6)
    assert_close(lse, [[[1.0 + np.log(2.0 + np.exp(-1.0))]]], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "H,Hkv,Tq,Tk,q_offset,window,sinks",
    [
        (9, 3, 12, 12, 0, 0, False),   # SmolLM2's 9 query heads on 3 KV heads, causal prefill
        (4, 1, 3, 40, 37, 0, False),   # MQA, a 3-row chunk at the end of a 40-token cache
        (4, 2, 20, 20, 0, 7, False),   # sliding window of 7 (Mistral)
        (2, 2, 6, 30, 24, 5, True),    # window and learned sinks (gpt-oss)
    ],
)
def test_matches_your_l7_7_windowed_attention(H, Hkv, Tq, Tk, q_offset, window, sinks):
    # WHY: P6: your L7.7 windowed_attention is the specification of every
    #      flag this kernel takes: GQA head sharing, causal masking at
    #      absolute positions (q_offset), the window, learned sinks, and the
    #      lse. Head width 64 like SmolLM2, tiles that divide nothing.
    # KIND: differential
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, m01
    # CHAPTER: L9.3 section 4
    rng = PCG32(SEED, seq=93 + Tk)
    D = 64
    q, k, v = rand(rng, (2, H, Tq, D)), rand(rng, (2, Hkv, Tk, D)), rand(rng, (2, Hkv, Tk, D))
    sl = rand(rng, (H,)) * 2 if sinks else None
    scale = D**-0.5
    o, lse = flash(q, k, v, scale=scale, q_offset=q_offset, window=window, sinks=sl, Br=5, Bc=7)
    want_o, want_lse = windowed_attention(
        q, k, v, window=window or None, sink_logits=sl, q_offset=q_offset, scale=scale
    )
    assert_close(o, want_o, rtol=1e-4, atol=1e-5, msg="o")
    assert_close(lse, want_lse, rtol=1e-5, atol=1e-5, msg="lse")


def test_non_causal_matches_your_l5_1_sdpa():
    # WHY: with causal = 0 and no window every key is visible (an encoder,
    #      or cross attention): the kernel equals your L5.1 sdpa_forward.
    # KIND: differential
    # CATCHES: s01, s02, m01
    # CHAPTER: L9.3 section 4
    rng = PCG32(SEED, seq=94)
    q, k, v = rand(rng, (2, 3, 11, 16)), rand(rng, (2, 3, 23, 16)), rand(rng, (2, 3, 23, 16))
    o, _ = flash(q, k, v, scale=0.25, causal=False, Br=4, Bc=6)
    want, _ = sdpa_forward(q, k, v, scale=0.25)
    assert_close(o, want, rtol=1e-4, atol=1e-5)


def test_chunked_prefill_is_bitwise_equal_through_ctypes():
    # WHY: what L10.3 relies on: a 25-token prompt computed in one call, or
    #      in chunks of 4 with q_offset = 0, 4, 8, ..., gives the same bits
    #      for every query row (key tiles aligned to absolute positions).
    # KIND: property
    # CATCHES: s06, s10, s14
    # CHAPTER: L9.3 section 2
    rng = PCG32(SEED, seq=95)
    T, H, Hkv, D = 25, 4, 2, 32
    q, k, v = rand(rng, (1, H, T, D)), rand(rng, (1, Hkv, T, D)), rand(rng, (1, Hkv, T, D))
    whole, _ = flash(q, k, v, scale=0.2, window=9, Br=8, Bc=8)
    parts = [
        flash(q[:, :, i : i + 4], k, v, scale=0.2, q_offset=i, window=9, Br=8, Bc=8)[0]
        for i in range(0, T, 4)
    ]
    assert_bits_equal(np.concatenate(parts, axis=2), whole, "chunked vs whole")
