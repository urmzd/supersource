# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L4.3 torch golden fixture (Luong attention).

The three scores of contracts/py/tinyllm/seq2seq/luong.pyi written with torch
ops, float32, weights copied by the course's parameter names:

    dot      e_s = h . k_s
    general  e_s = h . (k_s W_a^T)
    concat   e_s = v . tanh([h ; k_s] W_a^T)
    a = softmax(e masked to -inf past each length), c = sum_s a_s k_s,
    h~ = tanh([c ; h] W_c^T)

and the gradients of sum(h~ * gh) + sum(a * gw) for the query, the keys, and
every parameter, per score. Lengths 4, 2, 3.

    uv run --offline --python 3.12 --script course/oracle/L4.3/luong_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/L4.3/luong_torch.npz")
SEED = 20261010
rng = np.random.default_rng(SEED)


def f32(*shape):
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def main() -> None:
    B, S, d = 3, 4, 5
    q, k = f32(B, d), f32(B, S, d)
    lengths = np.array([4, 2, 3], dtype=np.int64)
    gh, gw = f32(B, d), f32(B, S)
    arrays = {"query": q, "keys": k, "lengths": lengths, "gh": gh, "gw": gw}
    for score in ("dot", "general", "concat"):
        params = {}
        if score == "general":
            params["score_proj.weight"] = f32(d, d)
        elif score == "concat":
            params["score_proj.weight"] = f32(d, 2 * d)
            params["v.weight"] = f32(1, d)
        params["combine.weight"] = f32(d, 2 * d)
        t = {n: torch.tensor(a, requires_grad=True) for n, a in params.items()}
        tq, tk = (
            torch.tensor(q, requires_grad=True),
            torch.tensor(k, requires_grad=True),
        )
        if score == "dot":
            e = torch.einsum("bsd,bd->bs", tk, tq)
        elif score == "general":
            e = torch.einsum("bsd,bd->bs", tk @ t["score_proj.weight"].T, tq)
        else:
            hq = tq[:, None, :].expand(B, S, d)
            e = (
                torch.tanh(torch.cat([hq, tk], dim=-1) @ t["score_proj.weight"].T)
                @ t["v.weight"].T
            ).squeeze(-1)
        mask = torch.arange(S)[None, :] < torch.tensor(lengths)[:, None]
        a = torch.softmax(e.masked_fill(~mask, float("-inf")), dim=-1)
        c = torch.bmm(a[:, None, :], tk).squeeze(1)
        ht = torch.tanh(torch.cat([c, tq], dim=-1) @ t["combine.weight"].T)
        (ht * torch.tensor(gh)).sum().add((a * torch.tensor(gw)).sum()).backward()
        arrays.update({f"{score}.param.{n}": a_ for n, a_ in params.items()})
        arrays.update(
            {
                f"{score}.context": c.detach().numpy(),
                f"{score}.weights": a.detach().numpy(),
                f"{score}.attentional": ht.detach().numpy(),
                f"{score}.grad.query": tq.grad.numpy(),
                f"{score}.grad.keys": tk.grad.numpy(),
            }
        )
        arrays.update({f"{score}.grad.{n}": p.grad.numpy() for n, p in t.items()})
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L4.3/luong_torch.py",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "shapes": {"B": B, "S": S, "d": d},
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L4.3/luong_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
