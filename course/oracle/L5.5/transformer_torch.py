# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L5.5 torch golden fixture (encoder-decoder).

A tiny transformer built from torch.nn.TransformerEncoderLayer and
TransformerDecoderLayer (batch_first, ReLU, dropout 0), d_model 8, 2 heads,
d_ff 16, 2 encoder and 2 decoder layers, one shared vocabulary of 11 ids
(0 = pad) tied across the source embedding, the target embedding, and the
output layer, embeddings scaled by sqrt(d_model), sinusoidal positions
(float64 angles, as contracts/py/tinyllm/xfmr/pos.pyi). Once with
norm_first=False (post-LN, the 2017 paper: no final LayerNorm) and once with
norm_first=True (pre-LN: TransformerEncoder/Decoder get a final LayerNorm).

Masks: src lengths 5, 3 (S = 5); tgt_in lengths 4, 2 (T = 4); torch gets the
causal mask, the key padding masks, and memory_key_padding_mask. Stored per
norm: the parameters under the course's names (in_proj split into q, k, v),
the logits [2, 4, 11], the label-smoothed loss (eps 0.1, pad ignored), and
the gradients of sum(logits * g) for every parameter.

    uv run --offline --python 3.12 --script course/oracle/L5.5/transformer_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path("course/fixtures/L5.5/transformer_torch.npz")
SEED = 20261012
V, D, H, FF, NL, PAD = 11, 8, 2, 16, 2, 0
S, T = 5, 4


def pe(n: int, d: int) -> np.ndarray:
    w = 10000.0 ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    a = np.arange(n, dtype=np.float64)[:, None] * w[None, :]
    out = np.empty((n, d))
    out[:, 0::2], out[:, 1::2] = np.sin(a), np.cos(a)
    return out.astype(np.float32)


def mha_names(prefix: str, m: nn.MultiheadAttention) -> dict:
    out = {}
    for j, n in enumerate(("q_proj", "k_proj", "v_proj")):
        out[f"{prefix}.{n}.weight"] = m.in_proj_weight[j * D : (j + 1) * D]
        out[f"{prefix}.{n}.bias"] = m.in_proj_bias[j * D : (j + 1) * D]
    out[f"{prefix}.out_proj.weight"] = m.out_proj.weight
    out[f"{prefix}.out_proj.bias"] = m.out_proj.bias
    return out


def run(norm_first: bool, rng: np.random.Generator, src, tgt_in, tgt_out, g) -> dict:
    emb = nn.Embedding(V, D)
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            D, H, FF, 0.0, batch_first=True, norm_first=norm_first
        ),
        NL,
        norm=nn.LayerNorm(D) if norm_first else None,
        enable_nested_tensor=False,
    )
    dec = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
            D, H, FF, 0.0, batch_first=True, norm_first=norm_first
        ),
        NL,
        norm=nn.LayerNorm(D) if norm_first else None,
    )
    mods = nn.ModuleList([emb, enc, dec])
    with torch.no_grad():
        for p in mods.parameters():
            p.copy_(
                torch.tensor(
                    rng.uniform(-0.5, 0.5, size=tuple(p.shape)).astype(np.float32)
                )
            )
    mods.train()  # dropout is 0; train mode keeps torch off its fused fast path
    P = torch.tensor(pe(16, D))
    src_t, tin = torch.tensor(src), torch.tensor(tgt_in)
    xs = emb(src_t) * math.sqrt(D) + P[:S]
    xt = emb(tin) * math.sqrt(D) + P[:T]
    src_pad, tgt_pad = src_t == PAD, tin == PAD
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    mem = enc(xs, src_key_padding_mask=src_pad)
    y = dec(
        xt,
        mem,
        tgt_mask=causal,
        tgt_key_padding_mask=tgt_pad,
        memory_key_padding_mask=src_pad,
    )
    logits = y @ emb.weight.T
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, V),
        torch.tensor(tgt_out).reshape(-1),
        ignore_index=PAD,
        label_smoothing=0.1,
    )
    (logits * torch.tensor(g)).sum().backward()
    names = {"src_emb.weight": emb.weight}
    for i, L in enumerate(enc.layers):
        p = f"encoder.{i}"
        names.update(mha_names(p + ".self_attn", L.self_attn))
        for n in ("linear1", "linear2", "norm1", "norm2"):
            names[f"{p}.{n}.weight"], names[f"{p}.{n}.bias"] = (
                getattr(L, n).weight,
                getattr(L, n).bias,
            )
    for i, L in enumerate(dec.layers):
        p = f"decoder.{i}"
        names.update(mha_names(p + ".self_attn", L.self_attn))
        names.update(mha_names(p + ".cross_attn", L.multihead_attn))
        for n in ("linear1", "linear2", "norm1", "norm2", "norm3"):
            names[f"{p}.{n}.weight"], names[f"{p}.{n}.bias"] = (
                getattr(L, n).weight,
                getattr(L, n).bias,
            )
    if norm_first:
        names["enc_norm.weight"], names["enc_norm.bias"] = (
            enc.norm.weight,
            enc.norm.bias,
        )
        names["dec_norm.weight"], names["dec_norm.bias"] = (
            dec.norm.weight,
            dec.norm.bias,
        )
    key = "pre" if norm_first else "post"
    out = {
        f"{key}.logits": logits.detach().numpy(),
        f"{key}.loss": np.array(loss.item(), dtype=np.float64),
    }
    for n, t in names.items():
        out[f"{key}.param.{n}"] = t.detach().numpy().copy()
    # grads: a slice of a packed parameter's grad is the grad of the slice
    grads = {"src_emb.weight": emb.weight.grad}
    for n, t in names.items():
        if n == "src_emb.weight":
            continue
        out[f"{key}.grad.{n}"] = None
    for i, L in enumerate(enc.layers):
        _grads(
            grads,
            f"encoder.{i}",
            L,
            ("self_attn",),
            ("linear1", "linear2", "norm1", "norm2"),
        )
    for i, L in enumerate(dec.layers):
        _grads(
            grads,
            f"decoder.{i}",
            L,
            ("self_attn", "multihead_attn"),
            ("linear1", "linear2", "norm1", "norm2", "norm3"),
        )
    if norm_first:
        grads["enc_norm.weight"], grads["enc_norm.bias"] = (
            enc.norm.weight.grad,
            enc.norm.bias.grad,
        )
        grads["dec_norm.weight"], grads["dec_norm.bias"] = (
            dec.norm.weight.grad,
            dec.norm.bias.grad,
        )
    for n in list(out):
        if out[n] is None:
            out[n] = grads[n.split(".grad.", 1)[1]].numpy().copy()
    out[f"{key}.grad.src_emb.weight"] = emb.weight.grad.numpy().copy()
    return out


def _grads(grads: dict, prefix: str, L, attns, plain) -> None:
    for a in attns:
        m = getattr(L, a)
        ours = "cross_attn" if a == "multihead_attn" else a
        for j, n in enumerate(("q_proj", "k_proj", "v_proj")):
            grads[f"{prefix}.{ours}.{n}.weight"] = m.in_proj_weight.grad[
                j * D : (j + 1) * D
            ]
            grads[f"{prefix}.{ours}.{n}.bias"] = m.in_proj_bias.grad[
                j * D : (j + 1) * D
            ]
        grads[f"{prefix}.{ours}.out_proj.weight"] = m.out_proj.weight.grad
        grads[f"{prefix}.{ours}.out_proj.bias"] = m.out_proj.bias.grad
    for n in plain:
        grads[f"{prefix}.{n}.weight"], grads[f"{prefix}.{n}.bias"] = (
            getattr(L, n).weight.grad,
            getattr(L, n).bias.grad,
        )


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    src = np.array([[3, 7, 4, 9, 5], [6, 2, 8, PAD, PAD]], dtype=np.int64)
    tgt_in = np.array([[1, 4, 10, 6], [1, 9, PAD, PAD]], dtype=np.int64)
    tgt_out = np.array([[4, 10, 6, 2], [9, 2, PAD, PAD]], dtype=np.int64)
    g = rng.normal(size=(2, T, V)).astype(np.float32)
    arrays = {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out, "g": g}
    arrays.update(run(False, rng, src, tgt_in, tgt_out, g))
    arrays.update(run(True, rng, src, tgt_in, tgt_out, g))
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L5.5/transformer_torch.py",
                "torch": torch.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "shapes": {
                    "vocab": V,
                    "d_model": D,
                    "n_heads": H,
                    "d_ff": FF,
                    "n_enc": NL,
                    "n_dec": NL,
                    "pad": PAD,
                },
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L5.5/transformer_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
