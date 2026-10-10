# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M10.2 golden fixture.

Writes course/fixtures/M10.2/torch_sgd.npz: 20-step torch.optim.SGD
trajectories (float64) on a fixed two-parameter problem, for five settings
of momentum, nesterov, and weight decay. The loss is

    L(p1, p2) = p1^T H p1 / 2 - b^T p1 + sum(C * (p2 - D)^2) / 2

so the course test can compute the same gradients in numpy:
dL/dp1 = H p1 - b, dL/dp2 = C * (p2 - D).

    uv run --script course/oracle/M10.2/torch_sgd_golden.py

Run from the repo root, then update the fixture's row in
course/fixtures/MANIFEST.tsv (the script prints it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/M10.2/torch_sgd.npz")
STEPS = 20
CONFIGS = {
    "plain": dict(lr=0.05),
    "momentum": dict(lr=0.05, momentum=0.9),
    "nesterov": dict(lr=0.05, momentum=0.9, nesterov=True),
    "momentum_wd": dict(lr=0.05, momentum=0.8, weight_decay=0.01),
    "wd": dict(lr=0.1, weight_decay=0.1),
}


def main() -> None:
    rng = np.random.default_rng(1002)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    H = (Q * np.array([0.5, 2.0, 8.0])) @ Q.T
    b = rng.normal(size=3)
    C = rng.uniform(0.5, 4.0, size=(2, 2))
    D = rng.normal(size=(2, 2))
    p1_0, p2_0 = rng.normal(size=3), rng.normal(size=(2, 2))
    data: dict[str, np.ndarray] = {
        "H": H,
        "b": b,
        "C": C,
        "D": D,
        "p1_0": p1_0,
        "p2_0": p2_0,
    }
    tH, tb, tC, tD = (torch.tensor(a, dtype=torch.float64) for a in (H, b, C, D))
    for name, cfg in CONFIGS.items():
        p1 = torch.tensor(p1_0, dtype=torch.float64, requires_grad=True)
        p2 = torch.tensor(p2_0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.SGD([p1, p2], **cfg)
        t1, t2 = [], []
        for _ in range(STEPS):
            opt.zero_grad()
            loss = 0.5 * p1 @ tH @ p1 - tb @ p1 + 0.5 * (tC * (p2 - tD) ** 2).sum()
            loss.backward()
            opt.step()
            t1.append(p1.detach().numpy().copy())
            t2.append(p2.detach().numpy().copy())
        data[f"{name}/p1"] = np.stack(t1)
        data[f"{name}/p2"] = np.stack(t2)
    meta = {
        "generator": "course/oracle/M10.2/torch_sgd_golden.py",
        "oracle_versions": f"torch=={torch.__version__} numpy=={np.__version__}",
        "seed": 1002,
        "steps": STEPS,
        "configs": CONFIGS,
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
                "course/oracle/M10.2/torch_sgd_golden.py",
                meta["oracle_versions"],
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
