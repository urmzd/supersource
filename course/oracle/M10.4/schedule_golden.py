# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.4.6"]
# ///
"""Maintainer generator for the M10.4 goldens: HF learning-rate schedules and
torch gradient clipping.

Schedules: each case builds a torch SGD optimizer with lr = lr_max, wraps it
in the HF scheduler, and records get_last_lr() before the first
scheduler.step() (step 0) and after each later one (steps 1..n). That is the
lr the optimizer uses for update number step + 1, the convention of
tinyllm.optim.schedule (step = optimizer steps already taken).

`compare_from` is the first step where the course schedule and HF agree:
HF's get_wsd_schedule ramps from min_lr_ratio * lr_max during warmup, the
course's wsd ramps from 0 like the cosine schedule (DEVIATIONS B34-03).

Clipping: torch.nn.utils.clip_grad_norm_(params, max_norm) on fixed
gradients (some None), recording the returned norm and the gradients after.

    uv run --offline --script course/oracle/M10.4/schedule_golden.py

Run from the repo root. It prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_wsd_schedule,
)

OUT = Path("course/fixtures/M10.4/schedule_hf.json")


def record(make, lr_max: float, n: int) -> list[float]:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=lr_max)
    sch = make(opt)
    out = [float(sch.get_last_lr()[0])]
    for _ in range(n):
        opt.step()
        sch.step()
        out.append(float(sch.get_last_lr()[0]))
    return out


def schedules() -> list[dict]:
    cases = []
    lr_max, lr_min, warm, total = 3e-4, 3e-5, 10, 100
    cases.append(
        {
            "name": "cosine_min_lr",
            "fn": "cosine_with_warmup",
            "args": {
                "warmup": warm,
                "total": total,
                "lr_max": lr_max,
                "lr_min": lr_min,
            },
            "compare_from": 0,
            "lr": record(
                lambda o: get_cosine_with_min_lr_schedule_with_warmup(
                    o, num_warmup_steps=warm, num_training_steps=total, min_lr=lr_min
                ),
                lr_max,
                total,
            ),
        }
    )
    cases.append(
        {
            "name": "cosine_no_warmup_to_zero",
            "fn": "cosine_with_warmup",
            "args": {"warmup": 0, "total": 40, "lr_max": 1e-3, "lr_min": 0.0},
            "compare_from": 0,
            "lr": record(
                lambda o: get_cosine_schedule_with_warmup(
                    o, num_warmup_steps=0, num_training_steps=40
                ),
                1e-3,
                40,
            ),
        }
    )
    w, s, d = 5, 25, 20
    cases.append(
        {
            "name": "wsd_linear_floor",
            "fn": "wsd",
            "args": {
                "warmup": w,
                "stable": s,
                "decay": d,
                "lr_max": 1e-3,
                "lr_min": 1e-4,
            },
            "compare_from": w,
            "lr": record(
                lambda o: get_wsd_schedule(
                    o,
                    num_warmup_steps=w,
                    num_decay_steps=d,
                    num_stable_steps=s,
                    decay_type="linear",
                    min_lr_ratio=0.1,
                ),
                1e-3,
                w + s + d + 10,
            ),
        }
    )
    cases.append(
        {
            "name": "wsd_linear_to_zero",
            "fn": "wsd",
            "args": {
                "warmup": 8,
                "stable": 0,
                "decay": 12,
                "lr_max": 6e-4,
                "lr_min": 0.0,
            },
            "compare_from": 0,
            "lr": record(
                lambda o: get_wsd_schedule(
                    o,
                    num_warmup_steps=8,
                    num_decay_steps=12,
                    num_stable_steps=0,
                    decay_type="linear",
                    min_lr_ratio=0.0,
                ),
                6e-4,
                25,
            ),
        }
    )
    return cases


def clips() -> list[dict]:
    rng = np.random.default_rng(10)
    cases = []
    for name, dtype, shapes, scale, max_norm, none_at in (
        ("clip_scaled", "float64", [(4,), (3, 5), (2,)], 1.0, 1.0, []),
        ("clip_below_max", "float64", [(4,), (3, 5)], 0.05, 1.0, []),
        ("clip_with_none", "float64", [(3,), (2, 2), (5,)], 2.0, 0.5, [1]),
        ("clip_float32", "float32", [(6,), (4, 4)], 3.0, 2.0, []),
    ):
        grads = [
            None if i in none_at else (scale * rng.standard_normal(s)).astype(dtype)
            for i, s in enumerate(shapes)
        ]
        tdt = torch.float64 if dtype == "float64" else torch.float32
        ps = []
        for s, g in zip(shapes, grads):
            p = torch.nn.Parameter(torch.zeros(s, dtype=tdt))
            p.grad = None if g is None else torch.tensor(g, dtype=tdt)
            ps.append(p)
        norm = torch.nn.utils.clip_grad_norm_(ps, max_norm, foreach=False)
        cases.append(
            {
                "name": name,
                "dtype": dtype,
                "max_norm": max_norm,
                "grads": [
                    None if g is None else g.astype(np.float64).tolist() for g in grads
                ],
                "norm": float(norm),
                "after": [
                    None
                    if p.grad is None
                    else p.grad.numpy().astype(np.float64).tolist()
                    for p in ps
                ],
            }
        )
    return cases


def main() -> None:
    doc = {
        "generator": "course/oracle/M10.4/schedule_golden.py",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "schedules": schedules(),
        "clip": clips(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1) + "\n").encode()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M10.4/schedule_golden.py\t"
        f"torch=={torch.__version__},transformers=={transformers.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
