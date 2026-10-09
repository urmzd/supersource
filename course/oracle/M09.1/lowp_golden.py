# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2"]
# ///
"""Maintainer generator for course/fixtures/M09.1/round_f32.npz.

The oracle is exact rational arithmetic (fractions.Fraction), not a library:
each float32 bit pattern is turned into its exact value, rounded to the
nearest bfloat16 and binary16 value with ties to even and IEEE overflow
(round as if the exponent were unbounded; a result above the largest finite
value is infinity), and encoded. numpy's float32 -> float16 cast is checked
against the binary16 column as a second, independent oracle. (DESIGN 4.1 names
a torch bf16 fixture; torch is not installed in this environment and nothing
may be installed, so the rational oracle stands in: DEVIATIONS M091-01.)

Inputs: special values, every class boundary (subnormal, normal, max finite,
overflow), round-to-even ties and their neighbours for both formats across
the whole exponent range, and random bit patterns from a stdlib PCG32.

    uv run --script course/oracle/M09.1/lowp_golden.py   # rewrite + MANIFEST row
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "course" / "fixtures" / "M09.1" / "round_f32.npz"

# name: (precision p incl. the implicit bit, min normal exponent, max exponent, bias)
FORMATS = {"bf16": (8, -126, 127, 127), "f16": (11, -14, 15, 15)}
NAN_CODE = {"bf16": 0x7FC0, "f16": 0x7E00}


def f32_value(bits: int) -> Fraction | float:
    s, e, m = bits >> 31, (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if e == 0xFF:
        return float("nan") if m else (float("-inf") if s else float("inf"))
    v = Fraction(m, 1 << 23) * Fraction(2) ** -126 if e == 0 else (1 + Fraction(m, 1 << 23)) * Fraction(2) ** (e - 127)
    return -v if s else v


def floor_log2(a: Fraction) -> int:
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


def round_code(bits: int, fmt: str) -> int:
    p, emin, emax, bias = FORMATS[fmt]
    mbits = p - 1
    sign = (bits >> 31) << 15
    v = f32_value(bits)
    if isinstance(v, float):
        if v != v:
            return NAN_CODE[fmt]
        return sign | (((1 << (15 - mbits)) - 1) << mbits)  # infinity: exponent all ones
    a = abs(v)
    if a == 0:
        return sign
    e = max(floor_log2(a), emin)
    quantum = Fraction(2) ** (e - mbits)
    q = a / quantum
    fl = q.numerator // q.denominator
    rem = q - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    if fl == 1 << p:  # carried into the next binade
        fl, e = 1 << mbits, e + 1
    max_finite = (2 - Fraction(2) ** -mbits) * Fraction(2) ** emax
    if fl * Fraction(2) ** (e - mbits) > max_finite:
        return sign | (((1 << (15 - mbits)) - 1) << mbits)
    if fl < 1 << mbits:  # subnormal (only when e == emin)
        return sign | fl
    return sign | ((e + bias) << mbits) | (fl - (1 << mbits))


class Pcg32:
    """spec/pcg32.md, stdlib only."""

    def __init__(self, seed: int, seq: int = 54) -> None:
        self.state, self.inc = 0, ((seq << 1) | 1) & (2**64 - 1)
        self.next()
        self.state = (self.state + seed) & (2**64 - 1)
        self.next()

    def next(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & (2**64 - 1)
        xs = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & 0xFFFFFFFF


def inputs() -> list[int]:
    f = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]  # noqa: E731
    out = [
        0x00000000, 0x80000000, 0x7F800000, 0xFF800000,  # zeros, infinities
        0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FBFFFFF, 0xFFFFFFFF, 0x7FFFFFFF,  # NaNs
        0x00000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,  # subnormal, normal edges
        0x7F7F7FFF, 0x7F7F8000, 0x7F7F8001, 0x7F7FFFFF,  # bf16 overflow boundary
        f(1.0), f(-2.5), f(0.1), f(-6.25), f(1 / 3), f(65504.0), f(65519.0), f(65520.0),
        f(65536.0), f(2.0**-24), f(2.0**-25), f(3 * 2.0**-26), f(2.0**-14), f(5.9604645e-08),
        f(1.00390625), f(1.01171875),
    ]
    for e in range(1, 255):  # bf16 ties and neighbours in every binade
        for hi in (0x00, 0x01, 0x7E, 0x7F):
            base = (e << 23) | (hi << 16)
            for low in (0x7FFF, 0x8000, 0x8001):
                out += [base | low, (base | low) | 0x80000000]
    for e in range(127 - 26, 127 + 17):  # f16 ties and neighbours, subnormal to overflow
        for hi in (0x000, 0x001, 0x3FE, 0x3FF):
            base = (e << 23) | (hi << 13)
            for low in (0x0FFF, 0x1000, 0x1001):
                out += [base | low, (base | low) | 0x80000000]
    g = Pcg32(9109)
    out += [g.next() for _ in range(3000)]
    seen, uniq = set(), []
    for b in out:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def main() -> int:
    bits = inputs()
    bf = [round_code(b, "bf16") for b in bits]
    hf = [round_code(b, "f16") for b in bits]
    f32 = np.array(bits, dtype=np.uint32)
    with np.errstate(all="ignore"):
        npf16 = f32.view(np.float32).astype(np.float16).view(np.uint16)
    for b, mine, theirs in zip(bits, hf, npf16.tolist()):
        if (mine & 0x7C00) == 0x7C00 and (mine & 0x3FF):  # NaN: any NaN code
            assert (theirs & 0x7C00) == 0x7C00 and (theirs & 0x3FF), hex(b)
        else:
            assert mine == theirs, (hex(b), hex(mine), hex(theirs))
    meta = {
        "generator": "course/oracle/M09.1/lowp_golden.py",
        "oracle": "fractions.Fraction exact rounding; f16 also equal to numpy float32->float16",
        "rules": "round to nearest, ties to even; overflow to inf; NaN -> bf16 0x7FC0, f16 0x7E00",
        "n": len(bits),
        "numpy": np.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as fh:
        np.savez_compressed(
            fh,
            f32_bits=f32,
            bf16_bits=np.array(bf, dtype=np.uint16),
            f16_bits=np.array(hf, dtype=np.uint16),
            __meta__=np.array(json.dumps(meta, sort_keys=True)),
        )
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.relative_to(ROOT).as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M09.1/lowp_golden.py",
                f"numpy=={np.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
