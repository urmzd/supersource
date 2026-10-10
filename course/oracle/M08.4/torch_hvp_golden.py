# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M08.4 golden fixture.

Writes course/fixtures/M08.4/torch_hvp.npz: exact Hessian-vector products
and one dense Hessian of the mean softmax cross-entropy of a linear model,
L(W) = mean_rows CE(X @ W, t), computed by torch's double backward
(torch.autograd.functional.hvp and .hessian) in float64.

    uv run --script course/oracle/M08.4/torch_hvp_golden.py

Run from the repo root, then update the fixture's row in
course/fixtures/MANIFEST.tsv (the script prints it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path("course/fixtures/M08.4/torch_hvp.npz")
torch.set_default_dtype(torch.float64)


def loss_of(X: np.ndarray, t: np.ndarray):
    Xt, tt = torch.tensor(X), torch.tensor(t, dtype=torch.int64)
    return lambda W: F.cross_entropy(Xt @ W, tt)


def main() -> None:
    rng = np.random.default_rng(804)
    d: dict[str, np.ndarray] = {}
    # HVPs along three directions at one point (8 rows, 5 features, 4 classes).
    X = rng.standard_normal((8, 5))
    t = rng.integers(0, 4, size=8)
    W = 0.5 * rng.standard_normal((5, 4))
    V = rng.standard_normal((3, 5, 4))
    f = loss_of(X, t)
    hv = [
        torch.autograd.functional.hvp(f, torch.tensor(W), torch.tensor(v))[1].numpy()
        for v in V
    ]
    d.update({"ce/X": X, "ce/t": t.astype(np.int64), "ce/W": W, "ce/V": V, "ce/HV": np.stack(hv)})
    # A dense Hessian of a smaller case (W [3, 2] flattened row-major: 6 x 6).
    X2 = rng.standard_normal((6, 3))
    t2 = rng.integers(0, 2, size=6)
    W2 = 0.5 * rng.standard_normal((3, 2))
    f2 = loss_of(X2, t2)
    H = torch.autograd.functional.hessian(lambda w: f2(w.reshape(3, 2)), torch.tensor(W2).reshape(-1))
    d.update({"hess/X": X2, "hess/t": t2.astype(np.int64), "hess/W": W2, "hess/H": H.numpy()})
    meta = {
        "generator": "course/oracle/M08.4/torch_hvp_golden.py",
        "seed": 804,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "dtype": "float64",
    }
    d["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **d)
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M08.4/torch_hvp_golden.py",
                f"torch=={torch.__version__.split('+')[0]} numpy=={np.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
