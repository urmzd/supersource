# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.2.6"]
# ///
"""Maintainer generator for craft.22: the tiny trained model whose evals the
learner writes, and its held-out text.

The model is a two-layer Llama-style decoder (RMSNorm, RoPE theta 10000 in
the half layout, causal attention with 4 heads, a SwiGLU MLP, tied
embeddings) over a character-level BPE of 96 merges, trained with AdamW on
the synthetic stories of course/fixtures/MS-corpus/sources/stories.jsonl
(the first 146 documents; the last 16 are the held-out text). The numpy
forward in course/ref/primers/craft.22/tinymodel.py is the same math; this
script checks that the two agree on the held-out text before writing.

    uv run --offline --script course/oracle/craft.22/train_tinylm.py

Run from the repo root; it prints the MANIFEST rows to paste into
course/fixtures/MANIFEST.tsv.

    tinylm.npz   embed [V, 48] and per layer l: l<l>.{wq,wk,wv,wo} [48, 48],
                 l<l>.{w_gate,w_up} [128, 48], l<l>.w_down [48, 128],
                 l<l>.{g_attn,g_mlp} [48], g_final [48] (all float16);
                 chars [B] (uint32 code points of the base vocabulary, sorted),
                 merges [96, 2] (int32 ids, rank order), __meta__ (uint8 JSON)
    val.txt      the held-out stories, one per line
    words.txt    the lowercase words of the training text, sorted
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SRC = Path("course/fixtures/MS-corpus/sources/stories.jsonl")
OUT = Path("course/fixtures/craft.22")
SEED = 20261022
D, H, LAYERS, FF, CTX, MERGES = 48, 4, 2, 128, 64, 96
THETA, EPS = 10000.0, 1e-5
STEPS, BATCH, LR = 2500, 32, 3e-3

torch.manual_seed(SEED)
torch.set_num_threads(1)
docs = [
    json.loads(line)["text"] for line in SRC.read_text().splitlines() if line.strip()
]
train_docs, val_docs = docs[:146], docs[146:]
train_text = "\n".join(train_docs)
val_text = "\n".join(val_docs)

# -- character BPE: base = sorted chars of the whole file; merges by count,
#    ties to the smallest (a, b) pair of ids.
chars = sorted(set("\n".join(docs)))
cid = {c: i for i, c in enumerate(chars)}
seq = [cid[c] for c in train_text]
merges: list[tuple[int, int]] = []
for r in range(MERGES):
    counts = Counter(zip(seq, seq[1:]))
    (a, b), _ = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    new = len(chars) + r
    merges.append((a, b))
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    seq = out
V = len(chars) + MERGES


def encode(text: str) -> list[int]:
    ids = [cid[c] for c in text]
    rank = {p: r for r, p in enumerate(merges)}
    while len(ids) > 1:
        best = min(
            ((rank.get(p, 1 << 30), i) for i, p in enumerate(zip(ids, ids[1:]))),
            default=(1 << 30, -1),
        )
        if best[0] == 1 << 30:
            break
        a, b = merges[best[0]]
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(len(chars) + best[0])
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


assert encode(train_text) == seq, "encode must reproduce the training merges"


# -- the model
def rmsnorm(x, g):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * g


hd = D // H
inv_freq = 1.0 / THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)


def rope(x, T):
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


P: dict[str, torch.nn.Parameter] = {}


def param(name, *shape, std=0.02, one=False):
    t = torch.ones(*shape) if one else torch.randn(*shape) * std
    P[name] = torch.nn.Parameter(t)


param("embed", V, D)
for l in range(LAYERS):
    for w in ("wq", "wk", "wv"):
        param(f"l{l}.{w}", D, D)
    param(f"l{l}.wo", D, D, std=0.02 / math.sqrt(2 * LAYERS))
    param(f"l{l}.w_gate", FF, D)
    param(f"l{l}.w_up", FF, D)
    param(f"l{l}.w_down", D, FF, std=0.02 / math.sqrt(2 * LAYERS))
    param(f"l{l}.g_attn", D, one=True)
    param(f"l{l}.g_mlp", D, one=True)
param("g_final", D, one=True)


def forward(ids):  # ids [B, T] -> logits [B, T, V]
    B, T = ids.shape
    x = P["embed"][ids]
    mask = torch.full((T, T), float("-inf")).triu(1)
    for l in range(LAYERS):
        h = rmsnorm(x, P[f"l{l}.g_attn"])
        q = (h @ P[f"l{l}.wq"].T).view(B, T, H, hd).transpose(1, 2)
        k = (h @ P[f"l{l}.wk"].T).view(B, T, H, hd).transpose(1, 2)
        v = (h @ P[f"l{l}.wv"].T).view(B, T, H, hd).transpose(1, 2)
        q, k = rope(q, T), rope(k, T)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd) + mask, -1)
        o = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + o @ P[f"l{l}.wo"].T
        h = rmsnorm(x, P[f"l{l}.g_mlp"])
        x = (
            x
            + (
                torch.nn.functional.silu(h @ P[f"l{l}.w_gate"].T)
                * (h @ P[f"l{l}.w_up"].T)
            )
            @ P[f"l{l}.w_down"].T
        )
    return rmsnorm(x, P["g_final"]) @ P["embed"].T


data = torch.tensor(seq)
opt = torch.optim.AdamW(P.values(), lr=LR, weight_decay=0.01)
gen = torch.Generator().manual_seed(SEED)
for step in range(STEPS):
    lr = LR * min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
    for g in opt.param_groups:
        g["lr"] = lr
    ix = torch.randint(0, len(data) - CTX - 1, (BATCH,), generator=gen)
    x = torch.stack([data[i : i + CTX] for i in ix])
    y = torch.stack([data[i + 1 : i + CTX + 1] for i in ix])
    loss = torch.nn.functional.cross_entropy(forward(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(P.values(), 1.0)
    opt.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step} loss {loss.item():.4f}", file=sys.stderr)

# -- write
OUT.mkdir(parents=True, exist_ok=True)
arrays = {k: v.detach().numpy().astype(np.float16) for k, v in P.items()}
meta = {
    "generator": "course/oracle/craft.22/train_tinylm.py",
    "seed": SEED,
    "d": D,
    "heads": H,
    "layers": LAYERS,
    "ff": FF,
    "theta": THETA,
    "eps": EPS,
    "steps": STEPS,
    "batch": BATCH,
    "ctx": CTX,
    "source": str(SRC),
    "train_docs": 146,
    "val_docs": len(val_docs),
    "torch": torch.__version__,
}
arrays["chars"] = np.array([ord(c) for c in chars], dtype=np.uint32)
arrays["merges"] = np.array(merges, dtype=np.int32)
arrays["__meta__"] = np.frombuffer(
    json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8
)
np.savez_compressed(OUT / "tinylm.npz", **arrays)
(OUT / "val.txt").write_text(val_text + "\n")
words = sorted(set(re.findall(r"[a-z]+", train_text.lower())))
(OUT / "words.txt").write_text("\n".join(words) + "\n")

# -- the numpy kata must agree with torch on the held-out text (f16 weights)
with torch.no_grad():
    for k, v in P.items():
        v.copy_(torch.from_numpy(arrays[k].astype(np.float32)))
    ids = encode(val_text[:200])
    ref = torch.log_softmax(forward(torch.tensor([ids])), -1)[0].numpy()
sys.path.insert(0, "course/ref/primers/craft.22")
import tinymodel  # noqa: E402

m = tinymodel.load(OUT / "tinylm.npz")
assert m.encode(val_text[:200]) == ids
got = m.log_probs(ids)
err = float(np.abs(got - ref).max())
print(f"numpy vs torch max |log p| difference: {err:.2e}", file=sys.stderr)
assert err < 1e-4, err
for f in ("tinylm.npz", "val.txt", "words.txt"):
    p = OUT / f
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(
        f"course/fixtures/craft.22/{f}\t{h}\t{p.stat().st_size}\tcourse/oracle/craft.22/train_tinylm.py\t"
        f"torch=={torch.__version__.split('+')[0]}\t-\tApache-2.0"
    )
