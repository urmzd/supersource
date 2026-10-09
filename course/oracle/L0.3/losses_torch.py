# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L0.3 torch golden fixture (losses).

For every case: the logits (or predictions), the targets, the options, torch's
loss, and torch's gradient of loss * g (g a random upstream gradient of the
loss's shape). Written to course/fixtures/L0.3/losses_torch.npz
(allow_pickle=False; `__meta__` is a JSON string describing each case).

    uv run --offline --python 3.12 --script course/oracle/L0.3/losses_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

OUT = Path("course/fixtures/L0.3/losses_torch.npz")
rng = np.random.default_rng(20261010)


def ce_case(name, shape, v, reduction="mean", ignore_index=-100, smoothing=0.0, n_ignored=0, scale=3.0, dtype="float64"):
    x = rng.uniform(-scale, scale, size=shape + (v,))
    t = rng.integers(0, v, size=shape)
    flat = t.reshape(-1)
    if n_ignored:
        flat[rng.choice(flat.size, size=n_ignored, replace=False)] = ignore_index
    return dict(name=name, kind="ce", x=x.astype(dtype), t=flat.reshape(shape), kwargs=dict(reduction=reduction, ignore_index=ignore_index, label_smoothing=smoothing), dtype=dtype)


CASES = [
    ce_case("ce_mean", (5,), 7),
    ce_case("ce_ignore", (6,), 4, n_ignored=2),
    ce_case("ce_ignore_zero", (6,), 4, ignore_index=0, n_ignored=0),
    ce_case("ce_smooth_ignore", (6,), 5, smoothing=0.1, n_ignored=2),
    ce_case("ce_sum", (5,), 6, reduction="sum", n_ignored=1),
    ce_case("ce_none_3d", (2, 3), 5, reduction="none", n_ignored=2),
    ce_case("ce_mean_3d", (2, 4), 6, n_ignored=3),
    ce_case("ce_smooth_none", (2, 3), 4, reduction="none", smoothing=0.25, n_ignored=1),
    ce_case("ce_large", (4,), 6, scale=1000.0),
    ce_case("ce_f32", (8,), 10, n_ignored=2, dtype="float32"),
    dict(name="mse", kind="mse", x=rng.uniform(-2, 2, (3, 4)), t=rng.uniform(-2, 2, (3, 4)), kwargs={}, dtype="float64"),
    dict(name="mse_f32", kind="mse", x=rng.uniform(-2, 2, (3, 4)).astype("float32"), t=rng.uniform(-2, 2, (3, 4)).astype("float32"), kwargs={}, dtype="float32"),
    dict(name="bce", kind="bce", x=rng.uniform(-4, 4, (4, 3)), t=rng.integers(0, 2, (4, 3)).astype("float64"), kwargs={}, dtype="float64"),
    dict(name="bce_soft_pw", kind="bce", x=rng.uniform(-4, 4, (4, 3)), t=rng.uniform(0, 1, (4, 3)), kwargs={}, pw=np.array([0.5, 2.0, 3.0]), dtype="float64"),
    dict(name="bce_large", kind="bce", x=np.array([[-1000.0, 1000.0, -40.0, 40.0]]), t=np.array([[1.0, 0.0, 1.0, 0.0]]), kwargs={}, dtype="float64"),
]


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for i, c in enumerate(CASES):
        key = f"c{i:02d}"
        x = torch.tensor(c["x"], requires_grad=True)
        if c["kind"] == "ce":
            v = c["x"].shape[-1]
            kw = c["kwargs"]
            y = TF.cross_entropy(x.reshape(-1, v), torch.from_numpy(c["t"].reshape(-1)), **kw)
            if kw["reduction"] == "none":
                y = y.reshape(c["t"].shape)
        elif c["kind"] == "mse":
            y = TF.mse_loss(x, torch.from_numpy(c["t"]))
        else:
            pw = torch.from_numpy(c["pw"]) if "pw" in c else None
            y = TF.binary_cross_entropy_with_logits(x, torch.from_numpy(c["t"]), pos_weight=pw)
        g = np.asarray(rng.uniform(0.5, 1.5, size=tuple(y.shape)), dtype=c["dtype"])
        (y * torch.from_numpy(g)).sum().backward()
        arrays[f"{key}_x"] = c["x"]
        arrays[f"{key}_t"] = c["t"]
        arrays[f"{key}_g"] = g
        arrays[f"{key}_y"] = y.detach().numpy()
        arrays[f"{key}_gx"] = x.grad.numpy()
        if "pw" in c:
            arrays[f"{key}_pw"] = c["pw"]
        meta.append({"key": key, "name": c["name"], "kind": c["kind"], "kwargs": c["kwargs"], "pos_weight": "pw" in c, "dtype": c["dtype"]})
    arrays["__meta__"] = np.array(json.dumps({"generator": "course/oracle/L0.3/losses_torch.py", "torch": torch.__version__, "numpy": np.__version__, "seed": 20261010, "cases": meta}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L0.3/losses_torch.py\ttorch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0")


main()
