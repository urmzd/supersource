# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L3.5 torch golden fixture (ELMo biLM).

The biLM of contracts/py/tinyllm/rnn/elmo.pyi with torch modules and the
course's weights copied in: nn.Embedding, a stack of single-direction
nn.LSTM per direction over pack_padded_sequence (the backward stack reads
each sequence reversed inside its length), the shared output layer; then
the layers R^0..R^L, both directions' logits, bilm_loss, a ScalarMix with
s = (0.3, -0.2, 0.5), gamma = 1.5, and the gradients of
bilm_loss + sum(mix * g) for every parameter. float32, lengths 5, 3.

    uv run --offline --python 3.12 --script course/oracle/L3.5/elmo_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

OUT = Path("course/fixtures/L3.5/elmo_torch.npz")
SEED = 20261013
rng = np.random.default_rng(SEED)
V, D, L, B, T = 9, 4, 2, 2, 5


def f32(*shape):
    return rng.uniform(-0.7, 0.7, size=shape).astype(np.float32)


def rev(x, lens):
    idx = torch.arange(T)[:, None].expand(T, B).clone()
    for b, n in enumerate(lens):
        idx[:n, b] = torch.arange(n - 1, -1, -1)
    return x[idx, torch.arange(B)[None, :]]


def main() -> None:
    p = {"emb.weight": f32(V, D)}
    for d in ("fwd", "bwd"):
        for k in range(L):
            p[f"{d}.{k}.weight_ih_l0"] = f32(4 * D, D)
            p[f"{d}.{k}.weight_hh_l0"] = f32(4 * D, D)
            p[f"{d}.{k}.bias_ih_l0"] = f32(4 * D)
            p[f"{d}.{k}.bias_hh_l0"] = f32(4 * D)
    p["out.weight"], p["out.bias"] = f32(V, D), f32(V)
    ids = rng.integers(0, V, size=(B, T)).astype(np.int64)
    lens = [5, 3]
    g = f32(B, T, 2 * D)
    t = {n: torch.tensor(a, requires_grad=True) for n, a in p.items()}

    def lstm(prefix):
        m = nn.LSTM(D, D)
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            setattr(m, f"{n}_l0", nn.Parameter(t[f"{prefix}.{n}_l0"].detach().clone()))
        return m

    mods = {f"{d}.{k}": lstm(f"{d}.{k}") for d in ("fwd", "bwd") for k in range(L)}

    def run(m, x):
        out, _ = m(pack_padded_sequence(x, torch.tensor(lens), enforce_sorted=False))
        return pad_packed_sequence(out, total_length=T)[0]

    e = t["emb.weight"][torch.tensor(ids.T)]  # [T, B, D]
    f, b = e, rev(e, lens)
    reps = [torch.cat([e, e], -1)]
    for k in range(L):
        f = run(mods[f"fwd.{k}"], f)
        b = run(mods[f"bwd.{k}"], b)
        reps.append(torch.cat([f, rev(b, lens)], -1))
    reps = [r.transpose(0, 1) for r in reps]
    bl = rev(b, lens)
    fl = f.transpose(0, 1) @ t["out.weight"].T + t["out.bias"]
    bwl = bl.transpose(0, 1) @ t["out.weight"].T + t["out.bias"]
    ft = np.full((B, T), -100)
    bt = np.full((B, T), -100)
    for i, n in enumerate(lens):
        ft[i, : n - 1] = ids[i, 1:n]
        bt[i, 1:n] = ids[i, : n - 1]
    loss = 0.5 * (
        Fn.cross_entropy(
            fl.reshape(-1, V), torch.tensor(ft).reshape(-1), ignore_index=-100
        )
        + Fn.cross_entropy(
            bwl.reshape(-1, V), torch.tensor(bt).reshape(-1), ignore_index=-100
        )
    )
    s = torch.tensor([0.3, -0.2, 0.5], requires_grad=True)
    gamma = torch.tensor([1.5], requires_grad=True)
    w = torch.softmax(s, 0)
    mix = gamma * sum(w[k] * reps[k] for k in range(L + 1))
    (loss + (mix * torch.tensor(g)).sum()).backward()
    arrays = {f"param.{n}": a for n, a in p.items()}
    arrays.update(
        ids=ids,
        lengths=np.array(lens, dtype=np.int64),
        g=g,
        loss=np.array(loss.item(), dtype=np.float32),
        fwd_logits=fl.detach().numpy(),
        bwd_logits=bwl.detach().numpy(),
        mix=mix.detach().numpy(),
        s=s.detach().numpy(),
        gamma=gamma.detach().numpy(),
        **{"grad.s": s.grad.numpy(), "grad.gamma": gamma.grad.numpy()},
    )
    for k, r in enumerate(reps):
        arrays[f"layer{k}"] = r.detach().numpy()
    for n, v in t.items():
        if v.grad is not None:
            arrays[f"grad.{n}"] = v.grad.numpy()
    for key, m in mods.items():
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            arrays[f"grad.{key}.{n}_l0"] = getattr(m, f"{n}_l0").grad.numpy()
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L3.5/elmo_torch.py",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "sizes": {"V": V, "d": D, "n_layers": L, "B": B, "T": T},
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L3.5/elmo_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
