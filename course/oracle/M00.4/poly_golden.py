# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Maintainer generator for the M00.4 golden values (Horner, quadratic roots).

mpmath at 80 significant digits evaluates each polynomial at the exact float64
coefficients and points, and solves each quadratic exactly; values are stored
as the nearest float64. The quadratics are the cases the textbook formula
gets wrong: b^2 much larger than 4ac, so one root is tiny next to the other.

    uv run --script course/oracle/M00.4/poly_golden.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mpmath

OUT = Path("course/fixtures/M00.4/poly.json")
mpmath.mp.dps = 80

QUADS = [(1.0, -(10.0**k), 1.0) for k in range(1, 16, 2)]
QUADS += [(1.0, 10.0**k, 1.0) for k in (4, 8, 12)]
QUADS += [
    (3.0, -7.0e9, 2.0),
    (-2.5, 1.0e7, 4.0e-3),
    (1e-3, 1.0, 1e-3),
    (4.0, 4.0001, 1.0),
]

# Taylor coefficients of exp (1/k!) and of cos, and a polynomial with sign changes.
POLYS = {
    "exp_taylor_8": [1.0 / math.factorial(k) for k in range(9)],
    "cos_taylor_10": [
        0.0 if k % 2 else (-1.0) ** (k // 2) / math.factorial(k) for k in range(11)
    ],
    "wilkinson_like_5": [
        -120.0,
        274.0,
        -225.0,
        85.0,
        -15.0,
        1.0,
    ],  # (x-1)(x-2)(x-3)(x-4)(x-5)
}
POINTS = [-2.0, -0.75, -0.1, 0.0, 0.3, 0.5, 1.0, 1.7, 2.5, 3.25]


def nearest(v) -> float:
    return float(mpmath.mpf(v))


def main() -> None:
    quads = []
    for a, b, c in QUADS:
        A, B, C = (mpmath.mpf(v) for v in (a, b, c))
        s = mpmath.sqrt(B * B - 4 * A * C)
        r = sorted([(-B - s) / (2 * A), (-B + s) / (2 * A)])
        quads.append({"a": a, "b": b, "c": c, "roots": [nearest(r[0]), nearest(r[1])]})
    polys = []
    for name, coeffs in POLYS.items():
        vals = [
            nearest(
                mpmath.fsum(
                    mpmath.mpf(ck) * mpmath.mpf(x) ** k for k, ck in enumerate(coeffs)
                )
            )
            for x in POINTS
        ]
        polys.append({"name": name, "coeffs": coeffs, "x": POINTS, "p": vals})
    doc = {
        "generator": "course/oracle/M00.4/poly_golden.py",
        "library": f"mpmath=={mpmath.__version__}",
        "dps": mpmath.mp.dps,
        "quadratics": quads,
        "horner": polys,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M00.4/poly_golden.py",
                f"mpmath=={mpmath.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
