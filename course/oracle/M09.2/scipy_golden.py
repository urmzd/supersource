# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy==1.17.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M09.2 golden fixture.

Writes course/fixtures/M09.2/scipy_golden.npz: float64 inputs of several
shapes and scales (including rows near 1e4 and rows with -inf entries),
with scipy.special's logsumexp, softmax, and log_softmax along each axis.

    uv run --script course/oracle/M09.2/scipy_golden.py

Run from the repo root, then update the fixture's row in
course/fixtures/MANIFEST.tsv (the script prints it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.special import log_softmax, logsumexp, softmax

OUT = Path("course/fixtures/M09.2/scipy_golden.npz")


def cases() -> list[tuple[str, np.ndarray, int]]:
    rng = np.random.default_rng(20261009)
    out = []
    out.append(("small_rows", rng.normal(0.0, 1.0, size=(4, 5)), -1))
    out.append(("columns", rng.normal(0.0, 3.0, size=(6, 3)), 0))
    out.append(("large_scale", rng.normal(0.0, 1.0, size=(3, 7)) * 50 + 9000.0, -1))
    out.append(("negative_scale", rng.normal(0.0, 1.0, size=(3, 7)) * 30 - 9000.0, -1))
    masked = rng.normal(0.0, 2.0, size=(5, 6))
    masked[masked < -1.0] = -np.inf
    masked[:, 0] = np.maximum(masked[:, 0], 0.0)  # keep one finite entry per row
    out.append(("masked", masked, -1))
    out.append(("rank3_middle", rng.normal(0.0, 4.0, size=(2, 5, 3)), 1))
    out.append(("vocab", rng.normal(0.0, 6.0, size=(2, 512)), -1))
    return out


def main() -> None:
    data: dict[str, np.ndarray] = {}
    names = []
    for name, x, axis in cases():
        names.append(name)
        data[f"{name}/x"] = x
        data[f"{name}/axis"] = np.array(axis, dtype=np.int64)
        data[f"{name}/logsumexp"] = logsumexp(x, axis=axis)
        data[f"{name}/softmax"] = softmax(x, axis=axis)
        data[f"{name}/log_softmax"] = log_softmax(x, axis=axis)
    meta = {
        "generator": "course/oracle/M09.2/scipy_golden.py",
        "oracle_versions": f"scipy=={scipy.__version__} numpy=={np.__version__}",
        "seed": 20261009,
        "cases": names,
    }
    data["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        np.savez(f, **data)
    raw = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT),
                hashlib.sha256(raw).hexdigest(),
                str(len(raw)),
                "course/oracle/M09.2/scipy_golden.py",
                meta["oracle_versions"],
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
