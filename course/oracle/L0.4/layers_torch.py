# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L0.4 torch golden fixture (layers).

Each case: the module's parameters (copied into the learner's layer by
state_dict name), an input, a random upstream gradient g, torch's output,
and torch's gradients of sum(output * g) for the input and every parameter.
float32 throughout. `__meta__` also records torch's state_dict key order for
a nested model, which is the safetensors key contract.

    uv run --offline --python 3.12 --script course/oracle/L0.4/layers_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path("course/fixtures/L0.4/layers_torch.npz")
torch.manual_seed(20261011)
rng = np.random.default_rng(20261011)


class Net(nn.Module):
    """The nested structure whose key order the test rebuilds."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(7, 4)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
                for _ in range(2)
            ]
        )
        self.ln = nn.LayerNorm(4)
        self.head = nn.Linear(4, 7, bias=False)


def f32(*shape, lo=-1.0, hi=1.0):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


CASES = []


def add(name, module, x, int_input=False):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.from_numpy(f32(*p.shape)))
    xt = torch.from_numpy(x) if int_input else torch.tensor(x, requires_grad=True)
    y = module(xt)
    g = f32(*y.shape)
    (y * torch.from_numpy(g)).sum().backward()
    CASES.append((name, module, x, xt, y, g, int_input))


add("linear", nn.Linear(3, 4), f32(2, 5, 3))
add("linear_nobias", nn.Linear(3, 2, bias=False), f32(4, 3))
add("layernorm", nn.LayerNorm(6), f32(3, 6, lo=-4, hi=4))
add(
    "embedding",
    nn.Embedding(5, 3),
    np.array([[1, 4, 1, 0], [3, 3, 2, 4]]),
    int_input=True,
)
add("mlp", nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2)), f32(6, 3))
add(
    "mlp_gelu",
    nn.Sequential(
        nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1)
    ),
    f32(4, 3),
)


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for i, (name, module, x, xt, y, g, int_input) in enumerate(CASES):
        key = f"c{i:02d}"
        arrays[f"{key}_x"] = x
        arrays[f"{key}_y"] = y.detach().numpy()
        arrays[f"{key}_g"] = g
        if not int_input:
            arrays[f"{key}_gx"] = xt.grad.numpy()
        names = []
        for pname, p in module.named_parameters():
            names.append(pname)
            arrays[f"{key}_p_{pname}"] = p.detach().numpy()
            arrays[f"{key}_gp_{pname}"] = p.grad.numpy()
        meta.append({"key": key, "name": name, "params": names, "int_input": int_input})
    keys = list(Net().state_dict().keys())
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L0.4/layers_torch.py",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": 20261011,
                "cases": meta,
                "net_state_dict_keys": keys,
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L0.4/layers_torch.py\ttorch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
