# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L0.2 torch golden fixture (DESIGN 5.11 `ops-torch`).

For every case: the inputs, the auxiliary integer or boolean arrays, a random
upstream gradient g, torch's forward output, and torch's gradient of
sum(output * g) with respect to every input. Float64 unless the case says
float32. Written to course/fixtures/L0.2/ops_torch.npz (allow_pickle=False;
`__meta__` is a JSON string naming each case's op, kwargs, and arrays).

    uv run --offline --python 3.12 --script course/oracle/L0.2/ops_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

OUT = Path("course/fixtures/L0.2/ops_torch.npz")
rng = np.random.default_rng(20261009)


def r(*shape, lo=-2.0, hi=2.0):
    return rng.uniform(lo, hi, size=shape)


def case(name, op, fn, inputs, kwargs=None, aux=None, dtype="float64"):
    return dict(name=name, op=op, fn=fn, inputs=inputs, kwargs=kwargs or {}, aux=aux or {}, dtype=dtype)


ties = np.array([[1.0, 3.0, 3.0, -1.0], [2.0, 2.0, 2.0, 0.5]])
masked = r(2, 4)
masked[0, 1] = -np.inf
masked[1, 3] = -np.inf
gidx = np.array([[2, 0, 2], [1, 1, 0]])
gidx0 = np.array([[1, 0, 2], [1, 1, 1]])
ids = np.array([[1, 3, 1], [0, 3, 3]])
cond = np.array([[True, False, True], [False, True, True]])
mask = np.array([[False, True, False], [True, False, False]])

CASES = [
    case("exp", "exp", lambda x: torch.exp(x), [r(2, 3)]),
    case("log", "log", lambda x: torch.log(x), [r(2, 3, lo=0.2, hi=3.0)]),
    case("tanh", "tanh", lambda x: torch.tanh(x), [r(2, 3)]),
    case("sigmoid", "sigmoid", lambda x: torch.sigmoid(x), [r(2, 3, lo=-6, hi=6)]),
    case("relu", "relu", lambda x: torch.relu(x), [r(2, 3)]),
    case("silu", "silu", lambda x: TF.silu(x), [r(2, 3, lo=-4, hi=4)]),
    case("gelu", "gelu", lambda x: TF.gelu(x), [r(2, 3, lo=-4, hi=4)]),
    case("gelu_tanh", "gelu", lambda x: TF.gelu(x, approximate="tanh"), [r(2, 3, lo=-4, hi=4)], {"approximate": "tanh"}),
    case("sum_all", "sum", lambda x: torch.sum(x), [r(2, 3)]),
    case("sum_axis1", "sum", lambda x: torch.sum(x, dim=1), [r(2, 3)], {"axis": 1}),
    case("sum_axes02_keep", "sum", lambda x: torch.sum(x, dim=(0, 2), keepdim=True), [r(2, 3, 4)], {"axis": [0, 2], "keepdims": True}),
    case("mean_all", "mean", lambda x: torch.mean(x), [r(3, 4)]),
    case("mean_last_keep", "mean", lambda x: torch.mean(x, dim=-1, keepdim=True), [r(3, 4)], {"axis": -1, "keepdims": True}),
    case("max_last", "max", lambda x: torch.amax(x, dim=-1), [r(3, 4)], {"axis": -1}),
    case("max_all", "max", lambda x: torch.amax(x), [r(3, 4)]),
    case("max_ties", "max", lambda x: torch.amax(x, dim=-1), [ties], {"axis": -1}),
    case("var_last", "var", lambda x: torch.var(x, dim=-1, correction=0), [r(3, 4)], {"axis": -1, "correction": 0}),
    case("var_all_c1", "var", lambda x: torch.var(x, correction=1), [r(3, 4)], {"correction": 1}),
    case("reshape", "reshape", lambda x: torch.reshape(x, (4, 3)), [r(2, 6)], {"shape": [4, 3]}),
    case("transpose", "transpose", lambda x: torch.transpose(x, 0, 2), [r(2, 3, 4)], {"a": 0, "b": 2}),
    case("permute", "permute", lambda x: x.permute(2, 0, 1), [r(2, 3, 4)], {"dims": [2, 0, 1]}),
    case("concat", "concat", lambda a, b: torch.cat([a, b], dim=1), [r(2, 3), r(2, 2)], {"axis": 1}),
    case("stack0", "stack", lambda a, b: torch.stack([a, b], dim=0), [r(2, 3), r(2, 3)], {"axis": 0}),
    case("stack_last", "stack", lambda a, b: torch.stack([a, b], dim=-1), [r(2, 3), r(2, 3)], {"axis": -1}),
    case("where", "where", lambda a, b: torch.where(torch.from_numpy(cond), a, b), [r(2, 3), r(3)], aux={"cond": cond}),
    case("gather1", "gather", lambda x: torch.gather(x, 1, torch.from_numpy(gidx)), [r(2, 3)], {"axis": 1}, {"idx": gidx}),
    case("gather0", "gather", lambda x: torch.gather(x, 0, torch.from_numpy(gidx0)), [r(3, 3)], {"axis": 0}, {"idx": gidx0}),
    case("embedding", "embedding", lambda w: TF.embedding(torch.from_numpy(ids), w), [r(4, 3)], aux={"ids": ids}),
    case("masked_fill", "masked_fill", lambda x: x.masked_fill(torch.from_numpy(mask), -7.5), [r(2, 3)], {"value": -7.5}, {"mask": mask}),
    case("softmax_last", "softmax", lambda x: torch.softmax(x, dim=-1), [r(3, 4)], {"axis": -1}),
    case("softmax_first", "softmax", lambda x: torch.softmax(x, dim=0), [r(3, 4)], {"axis": 0}),
    case("softmax_masked", "softmax", lambda x: torch.softmax(x, dim=-1), [masked], {"axis": -1}),
    case("log_softmax", "log_softmax", lambda x: torch.log_softmax(x, dim=-1), [r(3, 4)], {"axis": -1}),
    case("log_softmax_large", "log_softmax", lambda x: torch.log_softmax(x, dim=-1), [r(2, 5, lo=-1000, hi=1000)], {"axis": -1}),
    case("logsumexp", "logsumexp", lambda x: torch.logsumexp(x, dim=-1), [r(3, 4)], {"axis": -1}),
    case("logsumexp_keep0", "logsumexp", lambda x: torch.logsumexp(x, dim=0, keepdim=True), [r(3, 4)], {"axis": 0, "keepdims": True}),
    case("matmul", "matmul", lambda a, b: a @ b, [r(3, 4), r(4, 2)]),
    case("matmul_batched", "matmul", lambda a, b: a @ b, [r(2, 1, 3, 4), r(5, 4, 2)]),
    case("add_broadcast", "add", lambda a, b: a + b, [r(2, 3), r(3)]),
    case("div", "div", lambda a, b: a / b, [r(2, 3), r(2, 1, lo=0.5, hi=2.0)]),
    case("pow3", "pow", lambda x: x**3.0, [r(2, 3)], {"exponent": 3.0}),
    case("getitem_repeat", "getitem", lambda x: x[torch.tensor([2, 0, 2])], [r(3, 2)], aux={"index": np.array([2, 0, 2])}),
    case("relu_f32", "relu", lambda x: torch.relu(x), [r(4, 5)], dtype="float32"),
    case("softmax_f32", "softmax", lambda x: torch.softmax(x, dim=-1), [r(4, 5, lo=-30, hi=30)], {"axis": -1}, dtype="float32"),
    case("gelu_f32", "gelu", lambda x: TF.gelu(x), [r(4, 5, lo=-4, hi=4)], dtype="float32"),
]


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for i, c in enumerate(CASES):
        np_dt = np.dtype(c["dtype"])
        xs = [np.asarray(x, dtype=np_dt) for x in c["inputs"]]
        ts = [torch.tensor(x, requires_grad=True) for x in xs]
        y = c["fn"](*ts)
        g = np.asarray(rng.uniform(-1.0, 1.0, size=tuple(y.shape)), dtype=np_dt)
        (y * torch.from_numpy(g)).sum().backward()
        key = f"c{i:02d}"
        for k, x in enumerate(xs):
            arrays[f"{key}_x{k}"] = x
            arrays[f"{key}_gx{k}"] = ts[k].grad.detach().numpy()
        for name, a in c["aux"].items():
            arrays[f"{key}_aux_{name}"] = a
        arrays[f"{key}_g"] = g
        arrays[f"{key}_y"] = y.detach().numpy()
        meta.append({"key": key, "name": c["name"], "op": c["op"], "kwargs": c["kwargs"], "n_inputs": len(xs), "aux": sorted(c["aux"]), "dtype": c["dtype"]})
    arrays["__meta__"] = np.array(json.dumps({"generator": "course/oracle/L0.2/ops_torch.py", "torch": torch.__version__, "numpy": np.__version__, "seed": 20261009, "cases": meta}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L0.2/ops_torch.py\ttorch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0")


main()
