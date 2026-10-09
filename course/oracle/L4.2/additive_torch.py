# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L4.2 torch golden fixture (Bahdanau attention).

torch has no Bahdanau module, so the formula of contracts/py/tinyllm/seq2seq/
additive.pyi is written here with torch ops, float32, weights copied by the
course's parameter names:

    e = v . tanh(q W^T + k U^T + b), masked to -inf past each length,
    a = softmax(e), context = sum_s a_s k_s

and the gradients of sum(context * gc) + sum(a * gw) for the query, the keys,
and every parameter. Lengths 5, 3, 1 (no fully masked row: torch's softmax of
an all -inf row is NaN, the course defines it as 0 and tests that apart).

    uv run --offline --python 3.12 --script course/oracle/L4.2/additive_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/L4.2/additive_torch.npz")
SEED = 20261009
rng = np.random.default_rng(SEED)


def f32(*shape):
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def main() -> None:
    B, S, Dq, Dk, A = 3, 5, 4, 6, 7
    params = {
        "query.weight": f32(A, Dq),
        "key.weight": f32(A, Dk),
        "key.bias": f32(A),
        "v.weight": f32(1, A),
    }
    q, k = f32(B, Dq), f32(B, S, Dk)
    lengths = np.array([5, 3, 1], dtype=np.int64)
    gc, gw = f32(B, Dk), f32(B, S)
    t = {n: torch.tensor(a, requires_grad=True) for n, a in params.items()}
    tq, tk = torch.tensor(q, requires_grad=True), torch.tensor(k, requires_grad=True)
    mask = torch.arange(S)[None, :] < torch.tensor(lengths)[:, None]
    proj = tk @ t["key.weight"].T + t["key.bias"]
    wq = (tq @ t["query.weight"].T)[:, None, :]
    e = (torch.tanh(proj + wq) @ t["v.weight"].T).squeeze(-1)
    e = e.masked_fill(~mask, float("-inf"))
    a = torch.softmax(e, dim=-1)
    ctx = torch.bmm(a[:, None, :], tk).squeeze(1)
    (ctx * torch.tensor(gc)).sum().add((a * torch.tensor(gw)).sum()).backward()
    arrays = {f"param.{n}": a_ for n, a_ in params.items()}
    arrays.update(
        query=q,
        keys=k,
        lengths=lengths,
        gc=gc,
        gw=gw,
        context=ctx.detach().numpy(),
        weights=a.detach().numpy(),
        **{"grad.query": tq.grad.numpy(), "grad.keys": tk.grad.numpy()},
        **{f"grad.{n}": p.grad.numpy() for n, p in t.items()},
    )
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L4.2/additive_torch.py",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "shapes": {"B": B, "S": S, "Dq": Dq, "Dk": Dk, "A": A},
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L4.2/additive_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
