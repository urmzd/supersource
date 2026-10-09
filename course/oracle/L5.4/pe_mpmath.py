# /// script
# requires-python = ">=3.12"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Maintainer generator for the L5.4 golden fixture (sinusoidal positions).

PE(p, 2i) = sin(p * base^(-2i/d)), PE(p, 2i + 1) = cos(p * base^(-2i/d)),
evaluated with mpmath at 50 significant digits and rounded to float64, for
two settings: d = 16, base = 10000 (the paper) and d = 8, base = 500. The
positions include large ones (up to 65535), where computing the angle in
float32 is already wrong in the fourth decimal.

    uv run --offline --python 3.12 --script course/oracle/L5.4/pe_mpmath.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mpmath

OUT = Path("course/fixtures/L5.4/pe_mpmath.json")
POSITIONS = [0, 1, 2, 3, 7, 50, 511, 1000, 4097, 9999, 10000, 65535]
SETTINGS = [(16, 10000), (8, 500)]


def main() -> None:
    mpmath.mp.dps = 50
    cases = []
    for d, base in SETTINGS:
        rows = []
        for p in POSITIONS:
            row = []
            for i in range(d // 2):
                a = mpmath.mpf(p) * mpmath.power(base, mpmath.mpf(-2 * i) / d)
                row += [float(mpmath.sin(a)), float(mpmath.cos(a))]
            rows.append(row)
        freqs = [
            float(mpmath.power(base, mpmath.mpf(-2 * i) / d)) for i in range(d // 2)
        ]
        cases.append(
            {"d": d, "base": base, "positions": POSITIONS, "freqs": freqs, "pe": rows}
        )
    doc = {
        "generator": "course/oracle/L5.4/pe_mpmath.py",
        "mpmath": mpmath.__version__,
        "dps": 50,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L5.4/pe_mpmath.py\t"
        f"mpmath=={mpmath.__version__}\t-\tApache-2.0"
    )


main()
