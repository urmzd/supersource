"""Course tests for M09.4: your C conversions against your Python ones,
through ctypes (DESIGN 5.8 `parity/quant.fp8`, here as a module test).

A differential test alone passes when both sides share a bug, so each side
is also held to the ml_dtypes golden fixture: the Python side in
test_lowp.py, the C side here. bf16 and f16 have their Python twin in
M09.1 (tinyllm.num.fp), so the C bf16 and f16 conversions are compared with
your M09.1 code and with numpy's float16. libtinyllm is the unsanitized -O2
build ss makes for this check ($TINYLLM_LIB).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from _lib.pcg32 import PCG32
from tinyllm.num.fp import f32_to_bf16_bits, round_to_fp16
from tinyllm.num.lowp import f32_to_fp8_bits, fp8_bits_to_f32

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M09.4" / "lowp_golden.npz"


def lib() -> ctypes.CDLL:
    dll = ctypes.CDLL(os.environ["TINYLLM_LIB"])
    for enc, dec, t in [
        ("e4m3", "e4m3", ctypes.c_uint8),
        ("e5m2", "e5m2", ctypes.c_uint8),
        ("bf16", "bf16", ctypes.c_uint16),
        ("f16", "f16", ctypes.c_uint16),
    ]:
        f = getattr(dll, f"tl_f32_to_{enc}")
        f.argtypes, f.restype = [ctypes.c_float], t
        g = getattr(dll, f"tl_{dec}_to_f32")
        g.argtypes, g.restype = [t], ctypes.c_float
    return dll


def c_encode(dll, fmt: str, x: np.ndarray) -> np.ndarray:
    f = getattr(dll, f"tl_f32_to_{fmt}")
    dt = np.uint8 if fmt in ("e4m3", "e5m2") else np.uint16
    return np.array([f(float(v)) for v in np.asarray(x, dtype=np.float32)], dtype=dt)


def c_decode(dll, fmt: str, codes) -> np.ndarray:
    g = getattr(dll, f"tl_{fmt}_to_f32")
    return np.array([g(int(c)) for c in codes], dtype=np.float32)


def golden() -> dict:
    with np.load(FIX, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def specials() -> np.ndarray:
    return np.array(
        [
            0.0,
            -0.0,
            448.0,
            464.0,
            500.0,
            57344.0,
            61440.0,
            1e30,
            -1e30,
            np.inf,
            -np.inf,
            np.nan,
            2.0**-10,
            -(2.0**-10),
            2.0**-16,
            2.0**-17,
            1.4e-45,
            3.4028235e38,
        ],
        dtype=np.float32,
    )


def test_c_fp8_decode_equals_python_on_every_code():
    # WHY: all 256 codes of both fp8 formats decode to the same float32 bits
    #      in C and in Python (NaN codes to NaN), and the C side matches
    #      ml_dtypes: the KV format v2 (craft.13) writes in one language and
    #      reads in the other.
    # KIND: differential, smoke
    # CATCHES: s04, s05, s24
    # CHAPTER: M09.4 section 4
    dll, g = lib(), golden()
    for fmt in ("e4m3", "e5m2"):
        c = c_decode(dll, fmt, range(256))
        p = fp8_bits_to_f32(np.arange(256), fmt)
        assert np.array_equal(np.isnan(c), np.isnan(p)), fmt
        ok = ~np.isnan(c)
        assert np.array_equal(c[ok].view(np.uint32), p[ok].view(np.uint32)), fmt
        assert np.array_equal(
            c[ok].view(np.uint32), g[f"dec_{fmt}"][ok].view(np.uint32)
        ), fmt


def test_c_fp8_encode_equals_python_and_golden():
    # WHY: every fixture input (ties, near-ties, subnormals, random bit
    #      patterns), the saturation and NaN cases, and 20000 random float32
    #      bit patterns round to the same code in C and Python; C is also held
    #      to ml_dtypes where ml_dtypes defines the answer.
    # KIND: differential
    # CATCHES: s01, s02, s03, s06, s07, s08, s09, s19, s20, s21, s22, s23, s28, m01
    # CHAPTER: M09.4 section 4
    dll, g = lib(), golden()
    rng = PCG32(int(os.environ.get("SS_SEED", "0")), 44)
    rand = np.array([rng.next_u32() for _ in range(20000)], dtype=np.uint32).view(
        np.float32
    )
    for fmt in ("e4m3", "e5m2"):
        x = g[f"x_{fmt}"]
        c = c_encode(dll, fmt, x)
        assert np.array_equal(c, g[f"c_{fmt}"]), fmt
        for xs in (x, specials(), rand):
            c = c_encode(dll, fmt, xs)
            p = f32_to_fp8_bits(xs, fmt)
            bad = np.flatnonzero(c != p)
            assert bad.size == 0, (
                f"{fmt}: C {c[bad[0]]:#04x} != Python {p[bad[0]]:#04x} at x={xs[bad[0]]!r}"
            )


def test_c_bf16_f16_equal_m091_and_numpy():
    # WHY: the C bf16 and f16 conversions have their Python twin in M09.1:
    #      C must equal your f32_to_bf16_bits and round_to_fp16 and numpy's
    #      float16 cast on the fixture inputs and on random bit patterns, and
    #      decode all 65536 codes of each format to the bits Python expects.
    # KIND: differential
    # CATCHES: s19, s20, s21, s24, s25, s27, s28
    # CHAPTER: M09.4 section 2.7
    dll, g = lib(), golden()
    rng = PCG32(int(os.environ.get("SS_SEED", "0")), 45)
    rand = np.array([rng.next_u32() for _ in range(20000)], dtype=np.uint32).view(
        np.float32
    )
    rand = rand[~np.isnan(rand)]
    xb = np.concatenate([g["x_bf16"], rand])
    assert np.array_equal(c_encode(dll, "bf16", xb), f32_to_bf16_bits(xb))
    assert np.array_equal(c_encode(dll, "bf16", g["x_bf16"]), g["c_bf16"])
    xf = np.concatenate([g["x_f16"], rand])
    cf = c_encode(dll, "f16", xf)
    with np.errstate(over="ignore"):
        assert np.array_equal(cf, xf.astype(np.float16).view(np.uint16))
    back = c_decode(dll, "f16", cf)
    assert np.array_equal(back.view(np.uint32), round_to_fp16(xf).view(np.uint32))
    every = np.arange(65536, dtype=np.uint32)
    b = c_decode(dll, "bf16", every)
    nan = ((every >> 7) & 0xFF == 0xFF) & (every & 0x7F != 0)
    # NaN payloads do not survive the float -> double -> float trip through
    # ctypes (signaling NaNs come back quieted): compare NaN codes as NaN.
    assert np.array_equal(np.isnan(b), nan)
    assert np.array_equal(b[~nan].view(np.uint32), every[~nan] << 16)
    f = c_decode(dll, "f16", every)
    want = every.astype(np.uint16).view(np.float16).astype(np.float32)
    ok = ~np.isnan(want)
    assert np.array_equal(np.isnan(f), ~ok) and np.array_equal(
        f[ok].view(np.uint32), want[ok].view(np.uint32)
    )
