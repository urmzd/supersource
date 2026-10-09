# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy==1.17.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M11.1 golden fixture.

Writes course/fixtures/M11.1/scipy_golden.npz: pairs of distributions
(dense, with zeros, peaked, and a batch along axis 0) with scipy's entropy
(scipy.stats.entropy, nats), KL (scipy.stats.entropy(p, q)), cross-entropy
(entropy + KL), and Jensen-Shannon (scipy.spatial.distance.jensenshannon,
squared: scipy returns the JS distance, the square root of the divergence).

    uv run --script course/oracle/M11.1/scipy_golden.py

Run from the repo root, then update the fixture's row in
course/fixtures/MANIFEST.tsv (the script prints it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

OUT = Path("course/fixtures/M11.1/scipy_golden.npz")


def norm(a: np.ndarray, axis: int) -> np.ndarray:
    return a / a.sum(axis=axis, keepdims=True)


def cases() -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(1101)
    out = []
    out.append(("dense", norm(rng.random((5, 8)), -1), norm(rng.random((5, 8)), -1), -1))
    p = rng.random((4, 6))
    p[p < 0.3] = 0.0  # zeros in p only: KL stays finite
    p[:, 0] += 0.1
    out.append(("zeros_in_p", norm(p, -1), norm(rng.random((4, 6)) + 0.05, -1), -1))
    peaked = norm(np.exp(8.0 * rng.normal(size=(3, 50))), -1)
    out.append(("peaked", peaked, norm(rng.random((3, 50)), -1), -1))
    out.append(("axis0", norm(rng.random((7, 3)), 0), norm(rng.random((7, 3)), 0), 0))
    return out


def main() -> None:
    data: dict[str, np.ndarray] = {}
    names = []
    for name, p, q, axis in cases():
        names.append(name)
        data[f"{name}/p"] = p
        data[f"{name}/q"] = q
        data[f"{name}/axis"] = np.array(axis, dtype=np.int64)
        h = entropy(p, axis=axis)
        k = entropy(p, q, axis=axis)
        data[f"{name}/entropy"] = h
        data[f"{name}/kl"] = k
        data[f"{name}/cross_entropy"] = h + k
        if axis == -1:
            data[f"{name}/js"] = jensenshannon(p, q, axis=-1) ** 2
    meta = {
        "generator": "course/oracle/M11.1/scipy_golden.py",
        "oracle_versions": f"scipy=={scipy.__version__} numpy=={np.__version__}",
        "seed": 1101,
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
                "course/oracle/M11.1/scipy_golden.py",
                meta["oracle_versions"],
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
