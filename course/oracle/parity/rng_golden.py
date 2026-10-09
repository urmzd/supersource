# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the `rng` parity suite (course/DESIGN.md 5.8, D10).

The oracle is the frozen PCG32 every course test draws from
(course/tests/_lib/pcg32.py, PCG-XSH-RR 64/32 seeded as `pcg32_srandom_r`,
stream 54). For seeds 0, 1, and 2^63 it records the first 1024 `next_u32`
outputs, then 64 `uniform()` doubles and 64 Box-Muller normals from a fresh
generator. The normals follow spec/pcg32.md (pairs from two uniforms, cosine
first, the sine kept as the spare), computed here from the frozen uniforms:
the frozen `normal()` returns only the cosine half (DEVIATIONS K11, M063-03).
Every implementation (Python and C M06.3, Rust L10.1, Go load.01) must
reproduce them bit for bit.

    uv run --script course/oracle/parity/rng_golden.py

Run from the repo root. It rewrites course/fixtures/parity/rng.json and prints
the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "course" / "tests"))
from _lib.pcg32 import PCG32  # noqa: E402

OUT = ROOT / "course" / "fixtures" / "parity" / "rng.json"
SEEDS = [0, 1, 2**63]


def spec_normals(g: PCG32, n: int) -> list[float]:
    """spec/pcg32.md normal(): Box-Muller with a spare."""
    out: list[float] = []
    while len(out) < n:
        u1, u2 = g.uniform(), g.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        out += [r * math.cos(2.0 * math.pi * u2), r * math.sin(2.0 * math.pi * u2)]
    return out[:n]


def main() -> None:
    cases = []
    for seed in SEEDS:
        g = PCG32(seed, 54)
        u32 = [g.next_u32() for _ in range(1024)]
        g = PCG32(seed, 54)
        uni = [g.uniform() for _ in range(64)]
        g = PCG32(seed, 54)
        nor = spec_normals(g, 64)
        cases.append(
            {
                "name": f"seed-{seed}",
                "input": {
                    "seed": seed,
                    "seq": 54,
                    "n_u32": 1024,
                    "n_uniform": 64,
                    "n_normal": 64,
                },
                "output": {"u32": u32, "uniform": uni, "normal": nor},
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"generator": "course/oracle/parity/rng_golden.py", "cases": cases},
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
                "course/oracle/parity/rng_golden.py",
                "-",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
