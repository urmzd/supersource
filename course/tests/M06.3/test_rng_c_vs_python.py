"""Course tests for M06.3: your C PCG32 against your Python PCG32, through
ctypes (DESIGN 5.8 `parity/rng`, here as a module test).

A differential test alone passes when both sides share a bug, so the Python
side is also held to the spec vectors (test_rng.py) and the C side to them
here. libtinyllm is the unsanitized -O2 build ss makes for this check
($TINYLLM_LIB); the functions are declared with ctypes directly, because
the rt.01 loader declares only the v0 symbols.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from _lib.pcg32 import PCG32 as FrozenPCG32
from tinyllm.num import rng as learner_rng

VECTORS = json.loads(
    (Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M06.3" / "pcg32.vectors.json").read_text()
)


class TlPcg32(ctypes.Structure):
    _fields_ = [("state", ctypes.c_uint64), ("inc", ctypes.c_uint64)]


def lib() -> ctypes.CDLL:
    dll = ctypes.CDLL(os.environ["TINYLLM_LIB"])
    p = ctypes.POINTER(TlPcg32)
    dll.tl_pcg32_seed.argtypes = [p, ctypes.c_uint64, ctypes.c_uint64]
    dll.tl_pcg32_seed.restype = None
    dll.tl_pcg32_next.argtypes = [p]
    dll.tl_pcg32_next.restype = ctypes.c_uint32
    dll.tl_pcg32_uniform.argtypes = [p]
    dll.tl_pcg32_uniform.restype = ctypes.c_double
    dll.tl_fnv1a64.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    dll.tl_fnv1a64.restype = ctypes.c_uint64
    return dll


def c_gen(dll: ctypes.CDLL, seed: int, seq: int) -> TlPcg32:
    r = TlPcg32()
    dll.tl_pcg32_seed(ctypes.byref(r), seed, seq)
    return r


def test_c_struct_is_16_bytes_and_seeds_like_python():
    # WHY: tl_pcg32 is {uint64 state, inc} with no padding (c/ABI.md), so
    #      Python can hold one and Rust can mirror it; after seeding, the two
    #      sides hold the same (state, inc).
    # KIND: conformance, smoke
    # CATCHES: s17, m04
    # CHAPTER: M06.3 section 4
    assert ctypes.sizeof(TlPcg32) == 16
    dll = lib()
    for seed, seq in [(0, 54), (42, 54), (2**64 - 1, 2**63 - 1), (7, 0)]:
        r = c_gen(dll, seed, seq)
        assert (r.state, r.inc) == learner_rng.PCG32(seed, seq).state()


def test_c_matches_spec_vectors():
    # WHY: the oracle check for the C side: the first 1024 outputs of each
    #      spec seed, so a bug shared by both of your implementations fails.
    # KIND: golden
    # CATCHES: s13, s17
    # CHAPTER: M06.3 section 2.3
    dll = lib()
    for seed in ("0", "1", str(2**63)):
        r = c_gen(dll, int(seed), 54)
        assert [dll.tl_pcg32_next(ctypes.byref(r)) for _ in range(1024)] == VECTORS["next_u32"][seed]
        r = c_gen(dll, int(seed), 54)
        assert [dll.tl_pcg32_uniform(ctypes.byref(r)) for _ in range(8)] == VECTORS["uniform_f64"][seed]


def test_c_stream_equals_python_stream_100k():
    # WHY: 10^5 draws on random seeds and streams (from the frozen PCG32),
    #      half as u32 and half as uniforms: the Python training code and the
    #      C runtime must see the same numbers for the same seed (D10).
    # KIND: differential
    # CATCHES: s13, s15
    # CHAPTER: M06.3 section 2.3
    dll = lib()
    meta = FrozenPCG32(int(os.environ.get("SS_SEED", "0")), seq=64)
    for _ in range(4):
        seed = (meta.next_u32() << 32) | meta.next_u32()
        seq = meta.next_u32()
        r = c_gen(dll, seed, seq)
        g = learner_rng.PCG32(seed, seq)
        assert [dll.tl_pcg32_next(ctypes.byref(r)) for _ in range(12_500)] == [
            g.next_u32() for _ in range(12_500)
        ]
        assert [dll.tl_pcg32_uniform(ctypes.byref(r)) for _ in range(6_250)] == [
            g.uniform() for _ in range(6_250)
        ]


def test_c_fnv_equals_python_fnv():
    # WHY: rt.04 (C) hashes KV blocks and the Python paged cache (L8.3)
    #      looks the same blocks up by hash: one function, two languages.
    # KIND: differential
    # CATCHES: s16
    # CHAPTER: M06.3 section 2.7
    dll = lib()
    meta = FrozenPCG32(int(os.environ.get("SS_SEED", "0")), seq=65)
    for _ in range(200):
        data = bytes(meta.below(256) for _ in range(meta.below(64)))
        h = (meta.next_u32() << 32) | meta.next_u32()
        assert dll.tl_fnv1a64(data, len(data), h) == learner_rng.fnv1a64(data, h)
