# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "ml_dtypes==0.6.0"]
# ///
"""Maintainer generator for course/fixtures/M09.4/lowp_golden.npz.

The oracle is ml_dtypes 0.6.0 (Apache-2.0, the dtype library JAX and
TensorFlow use): float8_e4m3fn, float8_e5m2, float4_e2m1fn, float8_e8m0fnu,
and bfloat16, plus numpy's own float32 -> float16 cast. ml_dtypes rounds to
nearest even but does NOT saturate (e4m3fn overflows to NaN, e5m2 to inf),
so the encode arrays keep only inputs whose correct result is finite and
below the saturation point; saturation is a course rule tested by hand.

Arrays (float32 inputs, uint8/uint16 codes):
  dec_e4m3, dec_e5m2  [256]  float32 value of every fp8 code
  dec_fp4             [16]   float32 value of every fp4 code
  dec_e8m0            [256]  float32 value of every e8m0 code
  x_e4m3 / c_e4m3            inputs and codes, |x| < 464
  x_e5m2 / c_e5m2            inputs and codes, |x| < 61440
  x_bf16 / c_bf16            inputs and codes (finite results only)
  x_f16  / c_f16             inputs and codes (finite results only)
  mx_x                [16, 64] blocks of varied magnitude (float32)
  mx_fp4_codes, mx_fp4_scales      block 32, elem fp4_e2m1
  mx_e4m3_codes, mx_e4m3_scales    block 32, elem fp8_e4m3
Inputs: every code's value, the midpoints between neighbouring codes and
their float32 neighbours (the ties and near-ties), and random bit patterns
from a stdlib PCG32 with a fixed seed.

    uv run --offline --script course/oracle/M09.4/lowp_golden.py
Run from the repo root; it prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "course" / "fixtures" / "M09.4" / "lowp_golden.npz"
M64, M32 = (1 << 64) - 1, (1 << 32) - 1


class PCG32:
    def __init__(self, seed: int, seq: int = 54):
        self.state, self.inc = 0, ((seq << 1) | 1) & M64
        self.next()
        self.state = (self.state + seed) & M64
        self.next()

    def next(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & M64
        xs = (((old >> 18) ^ old) >> 27) & M32
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & M32


def f32(bits) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def near(values: np.ndarray) -> np.ndarray:
    """values, midpoints of neighbours, and the float32 neighbours of both."""
    v = np.unique(values[np.isfinite(values)].astype(np.float32))
    v = v[v >= 0]
    mids = ((v[:-1].astype(np.float64) + v[1:]) / 2).astype(np.float32)
    base = np.concatenate([v, mids])
    up = np.nextafter(base, np.float32(np.inf))
    down = np.nextafter(base, np.float32(-np.inf))
    allv = np.concatenate([base, up, down])
    return np.concatenate([allv, -allv]).astype(np.float32)


def codes_of(x: np.ndarray, dt) -> np.ndarray:
    return x.astype(dt).view(np.uint8 if np.dtype(dt).itemsize == 1 else np.uint16)


def main() -> None:
    rng = PCG32(20261009)
    dec = {}
    for name, dt in [
        ("e4m3", ml_dtypes.float8_e4m3fn),
        ("e5m2", ml_dtypes.float8_e5m2),
        ("e8m0", ml_dtypes.float8_e8m0fnu),
    ]:
        dec[name] = np.arange(256, dtype=np.uint8).view(dt).astype(np.float32)
    fp4 = (
        np.array([i for i in range(16)], dtype=np.uint8)
        .view(ml_dtypes.float4_e2m1fn)
        .astype(np.float32)
    )
    rand = f32([rng.next() for _ in range(6000)])
    rand = rand[np.isfinite(rand)]
    small = (
        np.array([rng.next() for _ in range(3000)], dtype=np.float64) / 2**32 * 2 - 1
    )

    out = {
        "dec_e4m3": dec["e4m3"],
        "dec_e5m2": dec["e5m2"],
        "dec_fp4": fp4,
        "dec_e8m0": dec["e8m0"],
    }
    for name, dt, lim in [
        ("e4m3", ml_dtypes.float8_e4m3fn, 464.0),
        ("e5m2", ml_dtypes.float8_e5m2, 61440.0),
    ]:
        cand = np.concatenate(
            [
                near(dec[name]),
                rand,
                (small * lim).astype(np.float32),
                np.float32([0.0, -0.0]),
            ]
        )
        cand = cand[np.isfinite(cand) & (np.abs(cand) < lim)]
        out[f"x_{name}"] = cand
        out[f"c_{name}"] = codes_of(cand, dt)
    bf_vals = (
        np.arange(65536, dtype=np.uint32)
        .astype(np.uint16)
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )
    f16_vals = (
        np.arange(65536, dtype=np.uint32)
        .astype(np.uint16)
        .view(np.float16)
        .astype(np.float32)
    )
    for name, vals, dt in [
        ("bf16", bf_vals, ml_dtypes.bfloat16),
        ("f16", f16_vals, np.float16),
    ]:
        cand = np.concatenate(
            [
                near(vals[(np.arange(65536) % 97 == 0) | (np.arange(65536) < 64)]),
                rand,
                np.float32([0.0, -0.0]),
            ]
        )
        with np.errstate(over="ignore"):
            r = cand.astype(dt)
        keep = np.isfinite(r.astype(np.float32))
        out[f"x_{name}"] = cand[keep]
        out[f"c_{name}"] = r[keep].view(np.uint16)
    # MX blocks: each row its own magnitude, a few exact zeros and ties.
    mx = np.array([[rng.next() / 2**32 * 2 - 1 for _ in range(64)] for _ in range(16)])
    mx *= np.ldexp(1.0, np.arange(16)[:, None] * 3 - 20)
    mx[3, :32] = 0.0
    mx[5, 7] = 0.0
    mx = mx.astype(np.float32)
    for elem, dt, emax, mx_max in [
        ("fp4", ml_dtypes.float4_e2m1fn, 2, 6.0),
        ("e4m3", ml_dtypes.float8_e4m3fn, 8, 448.0),
    ]:
        blk = mx.astype(np.float64).reshape(16, 2, 32)
        amax = np.abs(blk).max(axis=-1)
        e = np.frexp(np.where(amax == 0, 1.0, amax))[1] - 1
        sc = np.clip(e - emax + 127, 0, 254)
        q = np.clip(np.ldexp(blk, (127 - sc)[..., None]), -mx_max, mx_max)
        codes = q.astype(dt).view(np.uint8)
        if elem == "fp4":
            codes = codes & 0x0F
        out[f"mx_{elem}_codes"] = codes.reshape(16, 64).astype(np.uint8)
        out[f"mx_{elem}_scales"] = sc.astype(np.uint8)
        # cross-check the scale with ml_dtypes' own e8m0 decode
        assert np.array_equal(
            np.ldexp(1.0, sc - 127),
            sc.astype(np.uint8).view(ml_dtypes.float8_e8m0fnu).astype(np.float64),
        )
    out["mx_x"] = mx
    meta = {
        "generator": "course/oracle/M09.4/lowp_golden.py",
        "oracle": f"ml_dtypes=={ml_dtypes.__version__}, numpy=={np.__version__}",
        "seed": 20261009,
        "shapes": {k: list(v.shape) for k, v in out.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out, __meta__=np.array(json.dumps(meta, sort_keys=True)))
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT.relative_to(ROOT)),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M09.4/lowp_golden.py",
                f"ml_dtypes=={ml_dtypes.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
