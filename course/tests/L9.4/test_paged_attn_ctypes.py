"""Course tests for L9.4 (Python side): your paged decode kernel reading the
blocks of your own L8.3 PagedKVCache, against L8.3's gather followed by
your L7.7 windowed_attention (P6: Python is the specification). Sequences
are forked so they share prefix blocks, and appended past the fork so the
copy on write puts their tails in different blocks.

The C side under sanitizers is test_paged_attn.c. Inputs come from the
frozen PCG32 and closeness from the frozen close.py.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
from _lib.close import assert_close
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import STATUS, f32_ptr, load
from tinyllm.infer.paged import PagedKVCache
from tinyllm.modern.window import windowed_attention

SEED = int(os.environ.get("SS_SEED", "0"))
FP = ctypes.POINTER(ctypes.c_float)
U32P = ctypes.POINTER(ctypes.c_uint32)
I32P = ctypes.POINTER(ctypes.c_int32)
I64 = ctypes.c_int64


def lib():
    lb = load()
    lb.declare(
        "tl_paged_attn_decode_f32",
        STATUS,
        [
            FP,
            ctypes.c_void_p,
            ctypes.c_uint32,
            U32P,
            ctypes.c_int32,
            I32P,
            FP,
            I64,
            I64,
            I64,
            I64,
            ctypes.c_float,
            I64,
            ctypes.c_void_p,
        ],
    )
    return lb


def decode(cache: PagedKVCache, seqs, q, layer, *, scale, window=0) -> np.ndarray:
    """q [B, H, D] for the sequences `seqs` (their newest token is the query)."""
    tables = [cache.block_table(s) for s in seqs]
    max_blocks = max(len(t) for t in tables)
    tb = np.zeros((len(seqs), max_blocks), dtype=np.uint32)
    for i, t in enumerate(tables):
        tb[i, : len(t)] = t
    ctx = np.array([cache.seq_len(s, layer) for s in seqs], dtype=np.int32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    out = np.zeros_like(q)
    B, H, D = q.shape
    lib().tl_paged_attn_decode_f32(
        f32_ptr(q),
        cache.pool,
        layer,
        tb.ctypes.data_as(U32P),
        max_blocks,
        ctx.ctypes.data_as(I32P),
        f32_ptr(out),
        B,
        H,
        cache.n_kv_heads,
        D,
        scale,
        window,
        None,
    )
    return out


def test_hand_example_through_ctypes():
    # WHY: the worked example written through your L8.3 cache: keys
    #      [1,0], [0,1], [1,1], values [1,2], [3,4], [5,6], two positions per
    #      block; the query [1, 0] at position 2 gives o = [3, 4].
    # KIND: unit, smoke
    # CATCHES: s07, s08
    # CHAPTER: L9.4 section 3
    with PagedKVCache(
        load(), num_blocks=4, block_size=2, n_layers=1, n_kv_heads=1, d_head=2
    ) as c:
        c.add_seq(7)
        k = np.array([[[1, 0], [0, 1], [1, 1]]], dtype=np.float32)
        v = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
        c.append(7, 0, k, v)
        o = decode(c, [7], np.array([[[1, 0]]], dtype=np.float32), 0, scale=1.0)
    assert_close(o, [[[3.0, 4.0]]], rtol=1e-6, atol=1e-6)


def test_matches_your_l8_3_gather_and_l7_7_attention():
    # WHY: P6: your L8.3 gather (the same float16 values in position order)
    #      followed by your L7.7 windowed_attention, with the query at
    #      q_offset = len - 1, is the specification. SmolLM2's 9:3 heads and
    #      head width 64, block size 16, two layers, a forked pair sharing a
    #      prefix, lengths 1, 23, 40, and 41, with and without a window.
    # KIND: differential
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, m01
    # CHAPTER: L9.4 section 4
    rng = PCG32(SEED, seq=94)
    H, Hkv, D = 9, 3, 64
    with PagedKVCache(
        load(), num_blocks=32, block_size=16, n_layers=2, n_kv_heads=Hkv, d_head=D
    ) as c:

        def kv(T):
            return (
                rng.uniform_array((Hkv, T, D), -1.0, 1.0),
                rng.uniform_array((Hkv, T, D), -1.0, 1.0),
            )

        c.add_seq(0)
        c.add_seq(1)
        c.add_seq(2)
        for layer in (0, 1):
            c.append(0, layer, *kv(23))
            c.append(1, layer, *kv(1))
            c.append(2, layer, *kv(40))
        c.fork(2, 3)  # 3 shares sequence 2's blocks...
        for layer in (0, 1):
            c.append(3, layer, *kv(1))  # ...until its own token at position 40
        seqs = [0, 1, 2, 3]
        q = rng.uniform_array((len(seqs), H, D), -1.0, 1.0).astype(np.float32)
        for layer in (0, 1):
            for window in (0, 12):
                got = decode(c, seqs, q, layer, scale=D**-0.5, window=window)
                for i, s in enumerate(seqs):
                    K, V = c.gather(s, layer)
                    n = K.shape[1]
                    want, _ = windowed_attention(
                        q[i][None, :, None, :],
                        K[None].astype(np.float32),
                        V[None].astype(np.float32),
                        window=window or None,
                        q_offset=n - 1,
                        scale=D**-0.5,
                    )
                    assert_close(
                        got[i],
                        want[0, :, 0, :],
                        rtol=1e-4,
                        atol=1e-5,
                        msg=f"seq {s} layer {layer} w {window}",
                    )
