# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M01.3 golden fixture (DESIGN 4.0 `O`).

Evaluates seven activations and their derivatives with PyTorch autograd in
float64 at 1000 points: the grid -40, -39.9, ..., 40 (801 points, 0 and
+-40 included) and 199 standard normal draws scaled by 3 from the frozen
PCG32 at seed 1301. Writes course/fixtures/M01.3/activations_torch.npz
(allow_pickle=False, float64 arrays, a __meta__ JSON string).

    uv run --script course/oracle/M01.3/activations_torch.py

Run from the repo root, then update course/fixtures/MANIFEST.tsv with the
printed row. softplus uses threshold=1e9 so torch evaluates log1p(exp(x))
everywhere, the definition the course uses (torch's default threshold of 20
returns x itself above 20, an approximation off by exp(-20) = 2e-9).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "course/tests")
from _lib.pcg32 import PCG32  # noqa: E402

OUT = Path("course/fixtures/M01.3/activations_torch.npz")
F = torch.nn.functional

FUNCS = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": torch.relu,
    "softplus": lambda t: F.softplus(t, beta=1.0, threshold=1e9),
    "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"),
    "gelu_erf": lambda t: F.gelu(t, approximate="none"),
    "silu": F.silu,
}


def points() -> np.ndarray:
    grid = np.round(np.arange(-400, 401) / 10.0, 1)
    rng = PCG32(seed=1301)
    extra = np.array([3.0 * rng.normal() for _ in range(199)])
    return np.concatenate([grid, extra]).astype(np.float64)


def main() -> None:
    x = points()
    out: dict[str, np.ndarray] = {"x": x}
    for name, fn in FUNCS.items():
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        y = fn(t)
        (g,) = torch.autograd.grad(y.sum(), t)
        out[name] = y.detach().numpy().copy()
        out["d" + name] = g.numpy().copy()
    meta = {
        "generator": "course/oracle/M01.3/activations_torch.py",
        "torch": torch.__version__,
        "seed": 1301,
        "dtype": "float64",
        "points": int(x.size),
        "note": "softplus threshold=1e9; relu grad at 0 is 0 (torch)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, __meta__=np.array(json.dumps(meta, sort_keys=True)), **out)
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M01.3/activations_torch.py",
                f"torch=={torch.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
