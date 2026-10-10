# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L4.1 torch golden fixture (encoder-decoder).

The model of contracts/py/tinyllm/seq2seq/model.pyi built from torch modules
with the course's weights copied in: nn.Embedding, a bidirectional nn.GRU
over pack_padded_sequence (the course's enc_fwd is torch's *_l0 weights,
enc_bwd the *_l0_reverse ones), the bridge, nn.GRUCell or nn.LSTMCell, the
attention formulas of L4.2 and L4.3, and the output layer. Teacher forcing;
logits [B, T, Vt] and the gradients of sum(logits * g) for every parameter.

Cases: none/gru, bahdanau/gru, luong-concat/gru, luong-dot/lstm.

    uv run --offline --python 3.12 --script course/oracle/L4.1/seq2seq_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

OUT = Path("course/fixtures/L4.1/seq2seq_torch.npz")
SEED = 20261012
rng = np.random.default_rng(SEED)
VS, VT, E, H, A, B, S, T = 7, 8, 4, 6, 5, 3, 5, 4
H2 = H // 2


def f32(*shape):
    return rng.uniform(-0.8, 0.8, size=shape).astype(np.float32)


def run(attn: str, cell: str):
    p = {}
    p["src_emb.weight"] = f32(VS, E)
    for d in ("enc_fwd", "enc_bwd"):
        p[f"{d}.weight_ih_l0"] = f32(3 * H2, E)
        p[f"{d}.weight_hh_l0"] = f32(3 * H2, H2)
        p[f"{d}.bias_ih_l0"] = f32(3 * H2)
        p[f"{d}.bias_hh_l0"] = f32(3 * H2)
    p["bridge.weight"], p["bridge.bias"] = f32(H, H), f32(H)
    p["tgt_emb.weight"] = f32(VT, E)
    G = 3 if cell == "gru" else 4
    d_in = E + (H if attn != "none" else 0)
    p["cell.weight_ih"], p["cell.weight_hh"] = f32(G * H, d_in), f32(G * H, H)
    p["cell.bias_ih"], p["cell.bias_hh"] = f32(G * H), f32(G * H)
    if attn == "bahdanau":
        p["attention.query.weight"] = f32(A, H)
        p["attention.key.weight"], p["attention.key.bias"] = f32(A, H), f32(A)
        p["attention.v.weight"] = f32(1, A)
    elif attn == "luong-concat":
        p["attention.score_proj.weight"] = f32(H, 2 * H)
        p["attention.v.weight"] = f32(1, H)
        p["attention.combine.weight"] = f32(H, 2 * H)
    elif attn == "luong-dot":
        p["attention.combine.weight"] = f32(H, 2 * H)
    p["out.weight"] = f32(VT, 2 * H if attn == "bahdanau" else H)
    p["out.bias"] = f32(VT)
    lens = np.array([5, 2, 4], dtype=np.int64)
    src = rng.integers(0, VS, size=(B, S)).astype(np.int64)
    tgt = rng.integers(0, VT, size=(B, T)).astype(np.int64)
    g = f32(B, T, VT)

    t = {n: torch.tensor(a, requires_grad=True) for n, a in p.items()}
    gru = nn.GRU(E, H2, bidirectional=True)
    with torch.no_grad():
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(gru, f"{n}_l0").copy_(t[f"enc_fwd.{n}_l0"])
            getattr(gru, f"{n}_l0_reverse").copy_(t[f"enc_bwd.{n}_l0"])
    # Route gru's parameters to the course names so gradients land there.
    for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        setattr(gru, f"{n}_l0", nn.Parameter(t[f"enc_fwd.{n}_l0"].detach().clone()))
        setattr(
            gru, f"{n}_l0_reverse", nn.Parameter(t[f"enc_bwd.{n}_l0"].detach().clone())
        )
    gru.flatten_parameters = lambda: None
    x = t["src_emb.weight"][torch.tensor(src.T)]  # [S, B, E]
    packed = pack_padded_sequence(x, torch.tensor(lens), enforce_sorted=False)
    out, h_n = gru(packed)
    out, _ = pad_packed_sequence(out, total_length=S)  # [S, B, H]
    s = torch.tanh(
        torch.cat([h_n[0], h_n[1]], -1) @ t["bridge.weight"].T + t["bridge.bias"]
    )
    keys = out.transpose(0, 1)
    mask = torch.arange(S)[None, :] < torch.tensor(lens)[:, None]
    c = torch.zeros(B, H)
    feed = torch.zeros(B, H)

    def read(q):
        if attn == "bahdanau":
            proj = keys @ t["attention.key.weight"].T + t["attention.key.bias"]
            e = (
                torch.tanh(proj + (q @ t["attention.query.weight"].T)[:, None, :])
                @ t["attention.v.weight"].T
            ).squeeze(-1)
        elif attn == "luong-concat":
            hq = q[:, None, :].expand(B, S, H)
            e = (
                torch.tanh(
                    torch.cat([hq, keys], -1) @ t["attention.score_proj.weight"].T
                )
                @ t["attention.v.weight"].T
            ).squeeze(-1)
        else:
            e = torch.einsum("bsd,bd->bs", keys, q)
        a = torch.softmax(e.masked_fill(~mask, float("-inf")), -1)
        return torch.bmm(a[:, None, :], keys).squeeze(1)

    def step(xin, s, c):
        if cell == "gru":
            return torch._VF.gru_cell(
                xin,
                s,
                t["cell.weight_ih"],
                t["cell.weight_hh"],
                t["cell.bias_ih"],
                t["cell.bias_hh"],
            ), None
        h2, c2 = torch._VF.lstm_cell(
            xin,
            (s, c),
            t["cell.weight_ih"],
            t["cell.weight_hh"],
            t["cell.bias_ih"],
            t["cell.bias_hh"],
        )
        return h2, c2

    logits = []
    for j in range(T):
        e_ = t["tgt_emb.weight"][torch.tensor(tgt[:, j])]
        if attn == "none":
            s, c = step(e_, s, c)
            logits.append(s @ t["out.weight"].T + t["out.bias"])
        elif attn == "bahdanau":
            ctx = read(s)
            s, c = step(torch.cat([e_, ctx], -1), s, c)
            logits.append(torch.cat([s, ctx], -1) @ t["out.weight"].T + t["out.bias"])
        else:
            s, c = step(torch.cat([e_, feed], -1), s, c)
            ctx = read(s)
            feed = torch.tanh(torch.cat([ctx, s], -1) @ t["attention.combine.weight"].T)
            logits.append(feed @ t["out.weight"].T + t["out.bias"])
    L = torch.stack(logits, 1)
    (L * torch.tensor(g)).sum().backward()
    grads = {n: v.grad.numpy() for n, v in t.items() if v.grad is not None}
    for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        grads[f"enc_fwd.{n}_l0"] = getattr(gru, f"{n}_l0").grad.numpy()
        grads[f"enc_bwd.{n}_l0"] = getattr(gru, f"{n}_l0_reverse").grad.numpy()
    return (
        p,
        {"src": src, "lens": lens, "tgt": tgt, "g": g, "logits": L.detach().numpy()},
        grads,
    )


def main() -> None:
    arrays = {}
    cases = [
        ("none", "gru"),
        ("bahdanau", "gru"),
        ("luong-concat", "gru"),
        ("luong-dot", "lstm"),
    ]
    for attn, cell in cases:
        key = f"{attn}.{cell}"
        p, io, grads = run(attn, cell)
        arrays.update({f"{key}.param.{n}": a for n, a in p.items()})
        arrays.update({f"{key}.{n}": a for n, a in io.items()})
        arrays.update({f"{key}.grad.{n}": a for n, a in grads.items()})
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L4.1/seq2seq_torch.py",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "cases": [f"{a}.{c}" for a, c in cases],
                "sizes": {
                    "Vs": VS,
                    "Vt": VT,
                    "d_emb": E,
                    "d_h": H,
                    "d_attn": A,
                    "B": B,
                    "S": S,
                    "T": T,
                },
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L4.1/seq2seq_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
