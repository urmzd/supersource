# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L5.3 torch golden fixture (multi-head attention).

torch.nn.MultiheadAttention(d_model=8, num_heads=2, batch_first=True), float32,
dropout 0, with random weights. Its packed in_proj_weight [3d, d] holds the
query, key, and value projections stacked in that order; the fixture stores
the split (q_proj, k_proj, v_proj, out_proj) under the course's names, and the
packed tensors as torch names them. Two cases:

    self   x_q = x_kv [2, 4, 8], causal AND key padding (lengths 4, 3)
    cross  x_q [2, 3, 8], x_kv [2, 5, 8], key padding only (lengths 5, 2)

Masks are stored in the course's convention (bool, True = may attend); torch
receives their negation. Per case: the output, the per-head weights
(average_attn_weights=False), and the gradients of sum(out * g) for x_q,
x_kv, and every parameter (split by the course's names).

    uv run --offline --python 3.12 --script course/oracle/L5.3/mha_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/L5.3/mha_torch.npz")
SEED = 20261011
D, H = 8, 2


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    mha = torch.nn.MultiheadAttention(D, H, dropout=0.0, bias=True, batch_first=True)
    with torch.no_grad():
        for p in mha.parameters():
            p.copy_(
                torch.tensor(
                    rng.uniform(-0.6, 0.6, size=tuple(p.shape)).astype(np.float32)
                )
            )
    arrays: dict[str, np.ndarray] = {
        "in_proj_weight": mha.in_proj_weight.detach().numpy().copy(),
        "in_proj_bias": mha.in_proj_bias.detach().numpy().copy(),
        "out_proj.weight": mha.out_proj.weight.detach().numpy().copy(),
        "out_proj.bias": mha.out_proj.bias.detach().numpy().copy(),
    }
    cases = {
        "self": dict(Tq=4, Tk=4, lengths=[4, 3], causal=True),
        "cross": dict(Tq=3, Tk=5, lengths=[5, 2], causal=False),
    }
    for name, c in cases.items():
        B = 2
        xq = rng.normal(size=(B, c["Tq"], D)).astype(np.float32)
        xkv = (
            xq
            if name == "self"
            else rng.normal(size=(B, c["Tk"], D)).astype(np.float32)
        )
        keep = np.arange(c["Tk"])[None, :] < np.array(c["lengths"])[:, None]  # [B, Tk]
        mask = np.broadcast_to(keep[:, None, :], (B, c["Tq"], c["Tk"])).copy()
        if c["causal"]:
            mask &= np.tril(np.ones((c["Tq"], c["Tk"]), dtype=bool))[None]
        g = rng.normal(size=(B, c["Tq"], D)).astype(np.float32)
        mha.zero_grad()
        tq = torch.tensor(xq, requires_grad=True)
        tkv = tq if name == "self" else torch.tensor(xkv, requires_grad=True)
        # torch: a 3-D bool attn_mask is [B * H, Tq, Tk] with True = NOT allowed.
        tmask = torch.tensor(~mask).repeat_interleave(H, dim=0)
        out, w = mha(
            tq, tkv, tkv, attn_mask=tmask, need_weights=True, average_attn_weights=False
        )
        (out * torch.tensor(g)).sum().backward()
        gw, gb = mha.in_proj_weight.grad.numpy(), mha.in_proj_bias.grad.numpy()
        arrays.update(
            {
                f"{name}.x_q": xq,
                f"{name}.x_kv": xkv,
                f"{name}.mask": mask,
                f"{name}.g": g,
                f"{name}.out": out.detach().numpy(),
                f"{name}.weights": w.detach().numpy(),
                f"{name}.grad.x_q": tq.grad.numpy(),
                f"{name}.grad.out_proj.weight": mha.out_proj.weight.grad.numpy(),
                f"{name}.grad.out_proj.bias": mha.out_proj.bias.grad.numpy(),
            }
        )
        if name == "cross":
            arrays[f"{name}.grad.x_kv"] = tkv.grad.numpy()
        for j, p in enumerate(("q_proj", "k_proj", "v_proj")):
            arrays[f"{name}.grad.{p}.weight"] = gw[j * D : (j + 1) * D]
            arrays[f"{name}.grad.{p}.bias"] = gb[j * D : (j + 1) * D]
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L5.3/mha_torch.py",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "shapes": {"d_model": D, "n_heads": H},
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L5.3/mha_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
