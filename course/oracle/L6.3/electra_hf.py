# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L6.3 golden fixture (ELECTRA discriminator).

A tiny random Hugging Face ElectraForPreTraining (embedding_size equal to
hidden_size, so no embeddings_project; dropout 0; initializer_range 0.2 so
the logits are not all near 0) on a padded batch: its state dict, the
replaced-token logits [B, T], the loss HF computes (BCEWithLogitsLoss over the
positions whose attention_mask is 1), and the gradient of that loss with
respect to the discriminator head and the word embeddings. The course loads
the state dict through L6.2's hf_bert_encoder_sd (after dropping
"electra.") and compares.

    uv run --offline --python 3.12 --script course/oracle/L6.3/electra_hf.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import ElectraConfig, ElectraForPreTraining

OUT = Path("course/fixtures/L6.3/electra_hf.npz")
SEED = 20261010
CFG = dict(
    vocab_size=40,
    embedding_size=16,
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=16,
    type_vocab_size=2,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    initializer_range=0.2,
    layer_norm_eps=1e-12,
)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = ElectraForPreTraining(ElectraConfig(**CFG)).eval()
    ids = rng.integers(0, 40, size=(2, 7)).astype(np.int64)
    types = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1]], dtype=np.int64)
    mask = np.array([[1] * 7, [1] * 5 + [0] * 2], dtype=np.int64)
    labels = rng.integers(0, 2, size=(2, 7)).astype(np.int64)
    out = model(
        input_ids=torch.tensor(ids),
        token_type_ids=torch.tensor(types),
        attention_mask=torch.tensor(mask),
        labels=torch.tensor(labels),
    )
    out.loss.backward()
    arrays = {
        f"param.{k}": v.detach().numpy().astype(np.float32)
        for k, v in model.state_dict().items()
    }
    arrays.update(
        {
            "ids": ids,
            "types": types,
            "mask": mask,
            "labels": labels,
            "logits": out.logits.detach().numpy(),
            "loss": np.array(out.loss.item(), dtype=np.float64),
            "grad.dense.weight": model.discriminator_predictions.dense.weight.grad.numpy(),
            "grad.dense_prediction.weight": model.discriminator_predictions.dense_prediction.weight.grad.numpy(),
            "grad.word_emb.weight": model.electra.embeddings.word_embeddings.weight.grad.numpy(),
        }
    )
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L6.3/electra_hf.py",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "config": CFG,
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L6.3/electra_hf.py\t"
        f"torch=={torch.__version__},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
