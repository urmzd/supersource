# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6"]
# ///
"""Maintainer generator for the `softmax.online` parity suite (DESIGN 5.8).

Each case is a float32 matrix drawn from the frozen PCG32 (seed 2025, one
child stream per case) as uniforms in [-1, 1) times a scale, and a kernel
selector `online` (0: tl_softmax_f32, 1: tl_softmax_online_f32). The oracle
output is the naive row softmax computed in float64 from those same float32
values: e^(x - max) / sum. JSON has no infinities, so masked entries are
not part of this suite (the L9.2 course tests cover them).

    uv run --script course/oracle/parity/softmax_golden.py

Run from the repo root. It rewrites course/fixtures/parity/softmax.json and
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

OUT = ROOT / "course" / "fixtures" / "parity" / "softmax.json"
SHAPES = [
    (1, 1, 1.0),
    (1, 3, 1.0),
    (2, 5, 4.0),
    (3, 17, 30.0),
    (1, 64, 1e4),
    (4, 100, 10.0),
    (2, 333, 0.01),
]


def main() -> None:
    root = PCG32(2025, 54)
    cases = []
    for i, (rows, cols, scale) in enumerate(SHAPES):
        for online in (0, 1):
            g = root.split(2 * i + online)
            x = np.array(
                [(g.uniform() * 2 - 1) * scale for _ in range(rows * cols)],
                dtype=np.float32,
            )
            x64 = x.astype(np.float64).reshape(rows, cols)
            e = np.exp(x64 - x64.max(axis=1, keepdims=True))
            y = e / e.sum(axis=1, keepdims=True)
            cases.append(
                {
                    "name": f"{'online' if online else 'three_pass'}_{rows}x{cols}_s{scale:g}",
                    "input": {
                        "rows": rows,
                        "cols": cols,
                        "online": online,
                        "x": [float(v) for v in x],
                    },
                    "output": [float(v) for v in y.reshape(-1)],
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generator": "course/oracle/parity/softmax_golden.py",
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
                "course/oracle/parity/softmax_golden.py",
                f"numpy=={np.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
