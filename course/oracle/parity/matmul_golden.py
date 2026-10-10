# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6"]
# ///
"""Maintainer generator for the `matmul` parity suite (course/DESIGN.md 5.8).

Inputs are float32 matrices drawn from the frozen PCG32 (seed 2024, one child
stream per case) as uniforms in [-1, 1); the oracle output is numpy's float64
product of those same float32 values. A C implementation (M03.1 v0, L9.1)
passes when every element is within the frozen dot-product bound
4 * eps32 * sqrt(K) * max(|A| @ |B|) of the float64 answer.

    uv run --script course/oracle/parity/matmul_golden.py

Run from the repo root. It rewrites course/fixtures/parity/matmul.json and
prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "course" / "tests"))
from _lib.pcg32 import PCG32  # noqa: E402

OUT = ROOT / "course" / "fixtures" / "parity" / "matmul.json"
SHAPES = [
    (1, 1, 1),
    (2, 3, 4),
    (4, 8, 3),
    (7, 13, 5),
    (1, 64, 1),
    (16, 32, 16),
    (3, 100, 2),
]


def main() -> None:
    root = PCG32(2024, 54)
    cases = []
    for i, (m, k, n) in enumerate(SHAPES):
        g = root.split(i)
        a = np.array([g.uniform() * 2 - 1 for _ in range(m * k)], dtype=np.float32)
        b = np.array([g.uniform() * 2 - 1 for _ in range(k * n)], dtype=np.float32)
        # Apple Accelerate raises spurious floating-point flags inside matmul
        # (numpy 2.x warns about them); the outputs are checked finite below.
        with np.errstate(all="ignore"):
            c = a.astype(np.float64).reshape(m, k) @ b.astype(np.float64).reshape(k, n)
        assert np.isfinite(c).all()
        cases.append(
            {
                "name": f"{m}x{k}x{n}",
                "input": {
                    "m": m,
                    "k": k,
                    "n": n,
                    "a": [float(x) for x in a],
                    "b": [float(x) for x in b],
                },
                "output": [float(x) for x in c.reshape(-1)],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generator": "course/oracle/parity/matmul_golden.py",
                "numpy": np.__version__,
                "cases": cases,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.relative_to(ROOT).as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/parity/matmul_golden.py",
                f"numpy=={np.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
