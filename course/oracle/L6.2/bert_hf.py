# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L6.2 golden fixture: a tiny random BERT MLM.

transformers.BertForMaskedLM with BertConfig(vocab_size=40, hidden_size=16,
num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
max_position_embeddings=16, type_vocab_size=2, hidden_act "gelu" (exact),
every dropout 0, layer_norm_eps 1e-12), random weights from
torch.manual_seed (biases and LayerNorm gains perturbed), eager attention, float32. Inputs: ids [2, 7], the second
row padded after 4 tokens (attention_mask 0 there), token types 0 then 1,
and labels -100 except at three picked positions. Stored: the state dict
under Hugging Face's names (the tied decoder weight left out), the inputs,
the logits [2, 7, 40], HF's loss, and the gradient of that loss for every
stored tensor.

    uv run --offline --python 3.12 --script course/oracle/L6.2/bert_hf.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import BertConfig, BertForMaskedLM

OUT = Path("course/fixtures/L6.2/bert_tiny_hf.npz")
SEED = 20261014


def main() -> None:
    torch.manual_seed(SEED)
    cfg = BertConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
        type_vocab_size=2,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        attn_implementation="eager",
    )
    model = BertForMaskedLM(cfg)
    model.train()  # dropout is 0 everywhere
    with (
        torch.no_grad()
    ):  # HF zero-initializes biases and LayerNorm shifts: make them nonzero
        for n, p in model.named_parameters():
            if n.endswith("bias"):
                p.normal_(0.0, 0.1)
            elif "LayerNorm.weight" in n:
                p.add_(torch.randn_like(p) * 0.1)
    ids = torch.tensor([[2, 11, 25, 4, 17, 30, 3], [2, 9, 4, 3, 0, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    types = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0, 0]])
    labels = torch.full((2, 7), -100)
    labels[0, 2], labels[0, 4], labels[1, 1] = 12, 17, 9
    out = model(input_ids=ids, attention_mask=attn, token_type_ids=types, labels=labels)
    out.loss.backward()
    arrays = {
        "ids": ids.numpy(),
        "attention_mask": attn.numpy().astype(bool),
        "token_type_ids": types.numpy(),
        "labels": labels.numpy(),
        "logits": out.logits.detach().numpy(),
        "loss": np.array(out.loss.item(), dtype=np.float64),
    }
    for n, p in model.named_parameters():
        if n == "cls.predictions.decoder.weight":
            continue
        arrays[f"sd.{n}"] = p.detach().numpy().copy()
        arrays[f"grad.{n}"] = p.grad.numpy().copy()
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L6.2/bert_hf.py",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "config": {
                    "vocab_size": 40,
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "intermediate_size": 32,
                    "max_position_embeddings": 16,
                    "type_vocab_size": 2,
                    "hidden_act": "gelu",
                    "layer_norm_eps": 1e-12,
                },
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L6.2/bert_hf.py\t"
        f"torch=={torch.__version__},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
