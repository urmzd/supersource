# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the `quant.fp8` parity suite (DESIGN 5.8).

The oracle is an independent, table-driven reading of OCP FP8 (E4M3FN and
E5M2) as contracts/py/tinyllm/num/lowp.pyi and tinyllm/numerics.h state it:
decode every code exactly from its fields, and encode a float32 by searching
the decoded finite values for the nearest one (ties to the code with an even
last bit), with the contract's edge rules (NaN gives 0x7F; e4m3 saturates
to +-448 for |x| > 448 and infinities; e5m2 saturates finite values to
+-57344 and keeps infinities). No float conversion library is involved.

Each case holds `fmt` (4: e4m3, 5: e5m2), `x_bits` (float32 bit patterns to
encode), and `codes` (all 256 codes to decode). The expected output is
{"codes": [encoded uint8], "values_bits": [float32 bits of each decoded code,
-1 for NaN]}. Python (M09.4) and the optional C mirror (M09.7) must agree
with it exactly.

    uv run --script course/oracle/parity/quant_fp8_golden.py

Run from the repo root. It rewrites course/fixtures/parity/quant_fp8.json
and prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "course" / "tests"))
from _lib.pcg32 import PCG32  # noqa: E402

OUT = ROOT / "course" / "fixtures" / "parity" / "quant_fp8.json"
FMT = {4: (4, 3, 7), 5: (5, 2, 15)}  # exponent bits, mantissa bits, bias


def decode(code: int, fmt: int) -> float:
    eb, mb, bias = FMT[fmt]
    s = -1.0 if code & 0x80 else 1.0
    e = (code >> mb) & ((1 << eb) - 1)
    m = code & ((1 << mb) - 1)
    if fmt == 4 and e == 15 and m == 7:
        return math.nan
    if fmt == 5 and e == 31:
        return s * math.inf if m == 0 else math.nan
    if e == 0:
        return s * (m / (1 << mb)) * 2.0 ** (1 - bias)
    return s * (1 + m / (1 << mb)) * 2.0 ** (e - bias)


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def encode(bits: int, fmt: int) -> int:
    x = f32(bits)
    sign = 0x80 if bits >> 31 else 0
    if math.isnan(x):
        return 0x7F
    if math.isinf(x):
        return sign | (0x7E if fmt == 4 else 0x7C)
    a = abs(x)
    best = None
    for c in range(0x80):  # positive codes, finite only
        v = decode(c, fmt)
        if math.isnan(v) or math.isinf(v):
            continue
        d = abs(a - v)
        if (
            best is None
            or d < best[0]
            or (d == best[0] and c % 2 == 0 and best[1] % 2 == 1)
        ):
            best = (d, c)
    return sign | best[1]


def x_cases(fmt: int, seed: int) -> list[int]:
    vals = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        448.0,
        -448.0,
        449.0,
        464.0,
        1e6,
        -1e6,
        57344.0,
        61440.0,
        65536.0,
        2.0**-6,
        2.0**-9,
        2.0**-10,
        3 * 2.0**-11,
        2.0**-14,
        2.0**-16,
        2.0**-17,
        1.0625,
        1.1875,
        0.3,
        3.14159,
    ]
    xs = [bits32(v) for v in vals]
    xs += [0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x80000001]
    # every midpoint between neighbouring finite codes: the ties
    finite = sorted(
        {decode(c, fmt) for c in range(0x80)} - {math.inf} - {math.nan},
        key=lambda v: v if not math.isnan(v) else 0,
    )
    finite = [v for v in finite if not math.isnan(v)]
    for lo, hi in zip(finite, finite[1:]):
        mid = (lo + hi) / 2
        if bits32(mid) and f32(bits32(mid)) == mid:
            xs += [bits32(mid), bits32(-mid)]
    g = PCG32(seed)
    for _ in range(256):
        e = g.below(40) - 20
        xs.append(bits32(((g.next_u32() / 2**32) * 2 - 1) * 2.0**e))
    return xs


def main() -> None:
    cases = []
    for fmt, name in ((4, "e4m3"), (5, "e5m2")):
        xs = x_cases(fmt, 2026 + fmt)
        codes = list(range(256))
        vals = [decode(c, fmt) for c in codes]
        cases.append(
            {
                "name": name,
                "input": {"fmt": fmt, "x_bits": xs, "codes": codes},
                "output": {
                    "codes": [encode(b, fmt) for b in xs],
                    "values_bits": [-1 if math.isnan(v) else bits32(v) for v in vals],
                },
            }
        )
    data = (
        json.dumps(
            {"generator": "course/oracle/parity/quant_fp8_golden.py", "cases": cases},
            separators=(",", ":"),
        )
        + "\n"
    )
    OUT.write_text(data)
    b = data.encode()
    print(
        f"course/fixtures/parity/quant_fp8.json\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\t"
        "course/oracle/parity/quant_fp8_golden.py\t-\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
