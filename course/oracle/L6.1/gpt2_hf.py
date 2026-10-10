# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L6.1 golden fixture: a tiny random GPT-2.

transformers.GPT2LMHeadModel with GPT2Config(vocab_size=50, n_positions=16,
n_embd=16, n_layer=2, n_head=2, activation "gelu_new", every dropout 0,
layer_norm_epsilon 1e-5), random weights from torch.manual_seed (biases
and LayerNorm gains perturbed so every tensor matters), eager attention,
float32. Stored: the state dict under Hugging Face's names
(Conv1D weights as HF keeps them, [in, out]; the tied lm_head.weight and the
causal-mask buffers left out), ids [2, 8], the logits [2, 8, 50], HF's loss
for labels = ids (HF shifts inside), and the gradient of that loss for every
stored tensor.

    uv run --offline --python 3.12 --script course/oracle/L6.1/gpt2_hf.py

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

OUT = Path("course/fixtures/L6.1/gpt2_tiny_hf.npz")
SEED = 20261013


def main() -> None:
    torch.manual_seed(SEED)
    cfg = GPT2Config(
        vocab_size=50,
        n_positions=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(cfg)
    model.train()  # dropout is 0 everywhere
    with (
        torch.no_grad()
    ):  # HF starts biases at 0 and LayerNorm gains at 1: perturb them
        for n, p in model.named_parameters():
            if n.endswith("bias"):
                p.normal_(0.0, 0.1)
            elif ".ln_" in n or n.startswith("transformer.ln_f"):
                p.add_(torch.randn_like(p) * 0.1)
    ids = torch.tensor(
        np.random.default_rng(SEED).integers(0, 50, size=(2, 8)), dtype=torch.long
    )
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    arrays = {
        "ids": ids.numpy(),
        "logits": out.logits.detach().numpy(),
        "loss": np.array(out.loss.item(), dtype=np.float64),
    }
    for n, p in model.named_parameters():
        if n == "lm_head.weight":
            continue  # tied: the same tensor as transformer.wte.weight
        arrays[f"sd.{n}"] = p.detach().numpy().copy()
        arrays[f"grad.{n}"] = p.grad.numpy().copy()
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L6.1/gpt2_hf.py",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "config": {
                    "vocab_size": 50,
                    "n_positions": 16,
                    "n_embd": 16,
                    "n_layer": 2,
                    "n_head": 2,
                    "n_inner": 64,
                    "activation_function": "gelu_new",
                    "layer_norm_epsilon": 1e-5,
                },
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L6.1/gpt2_hf.py\t"
        f"torch=={torch.__version__},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
