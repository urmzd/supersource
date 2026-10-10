# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.6"]
# ///
"""Maintainer generator for the M10.3 golden trajectories (torch Adam and AdamW).

Each case fixes the parameters' initial values, a gradient for every step,
and the hyperparameters (a per-step lr list when the lr changes between
steps). torch runs the single-tensor path (`foreach=False`) on the CPU, and
the parameters after every step are recorded. The course test replays the
same gradients through the learner's optimizer and compares.

    uv run --offline --script course/oracle/M10.3/adamw_golden.py

Run from the repo root. It prints the MANIFEST.tsv row for the file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/M10.3/adam_torch.json")
STEPS = 20


def case(name, opt, dtype, shapes, lr, betas, eps, wd, grad_scale, seed):
    rng = np.random.default_rng(seed)
    init = [rng.standard_normal(s) for s in shapes]
    grads = [
        [grad_scale * rng.standard_normal(s) for s in shapes] for _ in range(STEPS)
    ]
    lrs = lr if isinstance(lr, list) else [lr] * STEPS
    tdt = torch.float64 if dtype == "float64" else torch.float32
    params = [torch.tensor(x, dtype=tdt) for x in init]
    cls = torch.optim.AdamW if opt == "AdamW" else torch.optim.Adam
    o = cls(params, lr=lrs[0], betas=betas, eps=eps, weight_decay=wd, foreach=False)
    traj = []
    for t in range(STEPS):
        for g in o.param_groups:
            g["lr"] = lrs[t]
        for p, gr in zip(params, grads[t]):
            p.grad = torch.tensor(gr, dtype=tdt)
        o.step()
        traj.append([p.detach().numpy().astype(np.float64).tolist() for p in params])
    st = [o.state[p] for p in params]
    return {
        "name": name,
        "optimizer": opt,
        "dtype": dtype,
        "lr": lrs,
        "betas": list(betas),
        "eps": eps,
        "weight_decay": wd,
        # inputs are cast to dtype exactly as torch.tensor(x, dtype) does
        "init": [np.asarray(x, dtype=dtype).astype(np.float64).tolist() for x in init],
        "grads": [
            [np.asarray(g, dtype=dtype).astype(np.float64).tolist() for g in step]
            for step in grads
        ],
        "trajectory": traj,
        "final_exp_avg": [s["exp_avg"].numpy().astype(np.float64).tolist() for s in st],
        "final_exp_avg_sq": [
            s["exp_avg_sq"].numpy().astype(np.float64).tolist() for s in st
        ],
    }


def main() -> None:
    shapes = [(3,), (2, 3)]
    sched = [
        1e-2 * (0.5 + 0.5 * np.cos(np.pi * t / STEPS)) + 1e-4 for t in range(STEPS)
    ]
    cases = [
        case(
            "adamw", "AdamW", "float64", shapes, 1e-2, (0.9, 0.999), 1e-8, 0.1, 1.0, 1
        ),
        case(
            "adam_l2", "Adam", "float64", shapes, 1e-2, (0.9, 0.999), 1e-8, 0.1, 1.0, 2
        ),
        case(
            "adamw_lr_per_step",
            "AdamW",
            "float64",
            shapes,
            [float(x) for x in sched],
            (0.9, 0.95),
            1e-8,
            0.1,
            1.0,
            3,
        ),
        case(
            "adamw_tiny_grads",
            "AdamW",
            "float64",
            shapes,
            1e-3,
            (0.9, 0.999),
            1e-6,
            0.0,
            1e-6,
            4,
        ),
        case(
            "adamw_float32",
            "AdamW",
            "float32",
            shapes,
            3e-3,
            (0.8, 0.99),
            1e-8,
            0.01,
            1.0,
            5,
        ),
    ]
    doc = {
        "generator": "course/oracle/M10.3/adamw_golden.py",
        "torch": torch.__version__,
        "steps": STEPS,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1) + "\n").encode()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M10.3/adamw_golden.py\t"
        f"torch=={torch.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
