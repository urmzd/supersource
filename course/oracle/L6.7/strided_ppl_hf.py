# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L6.7 golden fixture (strided perplexity).

A tiny random Hugging Face GPT2LMHeadModel (2 layers, d 16, 2 heads, 16
positions, vocabulary 50) scores a 40-token sequence with the course's
strided evaluation (contracts/py/tinyllm/eval/lm.pyi, token_nlls): windows
start at 0, stride, 2 stride, ...; a window [s, e) with e = min(s + ctx_len,
n) scores the tokens t in [max(prev_e, 1), e) from its own logits at t - 1 - s;
every token 1..n-1 exactly once. Recorded per setting: the per-token NLLs in
float64 (from the float32 logits) and their mean, plus the HF state dict that
L6.1's load_hf_gpt2 reads.

    uv run --offline --python 3.12 --script course/oracle/L6.7/strided_ppl_hf.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel

OUT = Path("course/fixtures/L6.7/strided_ppl_hf.npz")
SEED = 20261013
CFG = dict(
    vocab_size=50,
    n_positions=16,
    n_embd=16,
    n_layer=2,
    n_head=2,
    n_inner=64,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    layer_norm_epsilon=1e-5,
    activation_function="gelu_new",
    initializer_range=0.3,
)
SETTINGS = [(8, 4), (8, 7), (16, 15), (16, 1)]


def token_nlls(model, ids: np.ndarray, ctx: int, stride: int) -> np.ndarray:
    n = len(ids)
    out = np.full(n, np.nan)
    prev = 0
    for s in range(0, n, stride):
        e = min(s + ctx, n)
        with torch.no_grad():
            logits = model(torch.tensor(ids[s:e])[None]).logits[0].double()
        lsm = torch.log_softmax(logits, dim=-1).numpy()
        for t in range(max(prev, 1), e):
            out[t] = -lsm[t - 1 - s, ids[t]]
        prev = e
        if e == n:
            break
    return out[1:]


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = GPT2LMHeadModel(GPT2Config(**CFG)).eval()
    ids = rng.integers(0, 50, size=40).astype(np.int64)
    arrays = {
        f"param.{k}": v.detach().numpy().astype(np.float32)
        for k, v in model.state_dict().items()
    }
    arrays["ids"] = ids
    for ctx, stride in SETTINGS:
        nll = token_nlls(model, ids, ctx, stride)
        arrays[f"nll.{ctx}.{stride}"] = nll
        arrays[f"mean.{ctx}.{stride}"] = np.array(nll.mean())
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L6.7/strided_ppl_hf.py",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "config": CFG,
                "settings": SETTINGS,
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L6.7/strided_ppl_hf.py\t"
        f"torch=={torch.__version__},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
