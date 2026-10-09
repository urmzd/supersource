# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L6.4 golden fixture (T5 relative positions).

Hugging Face's T5Attention._relative_position_bucket on every relative
position in -300..300 for five (bidirectional, num_buckets, max_distance)
settings, and T5Attention.compute_bias with copied random bias weights for
an encoder (bidirectional) and a decoder (causal, with past_seen_tokens),
so the course's t5_relative_bucket and T5RelativeBias are compared with the
code T5 checkpoints were trained with (torch float32 log, truncation).

    uv run --offline --python 3.12 --script course/oracle/L6.4/t5_golden.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Attention

OUT = Path("course/fixtures/L6.4/t5_buckets.npz")
SEED = 20261009
SETTINGS = [
    (True, 32, 128),
    (False, 32, 128),
    (True, 8, 20),
    (False, 16, 64),
    (True, 64, 256),
]
REL = np.arange(-300, 301, dtype=np.int64)


def main() -> None:
    arrays = {"rel": REL}
    for bi, nb, md in SETTINGS:
        b = T5Attention._relative_position_bucket(
            torch.tensor(REL), bidirectional=bi, num_buckets=nb, max_distance=md
        )
        arrays[f"bucket.{int(bi)}.{nb}.{md}"] = b.numpy().astype(np.int64)
    rng = np.random.default_rng(SEED)
    cases = {"encoder": (False, 5, 7, 0), "decoder": (True, 3, 9, 6)}
    for name, (is_dec, q, k, past) in cases.items():
        cfg = T5Config(
            d_model=8,
            d_kv=4,
            num_heads=3,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=20,
            is_decoder=is_dec,
        )
        att = T5Attention(cfg, has_relative_attention_bias=True)
        w = rng.uniform(-1, 1, size=(8, 3)).astype(np.float32)
        with torch.no_grad():
            att.relative_attention_bias.weight.copy_(torch.tensor(w))
            bias = att.compute_bias(q, k, past_seen_tokens=past)[0].numpy()
        arrays[f"{name}.weight"] = w
        arrays[f"{name}.bias"] = bias
        arrays[f"{name}.shape"] = np.array([q, k, past], dtype=np.int64)
    arrays["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L6.4/t5_golden.py",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "seed": SEED,
                "settings": [list(s) for s in SETTINGS],
                "cases": {
                    k: {
                        "is_decoder": v[0],
                        "q_len": v[1],
                        "k_len": v[2],
                        "past_seen_tokens": v[3],
                    }
                    for k, v in cases.items()
                },
                "bias_config": {
                    "num_heads": 3,
                    "relative_attention_num_buckets": 8,
                    "relative_attention_max_distance": 20,
                },
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L6.4/t5_golden.py\t"
        f"torch=={torch.__version__},transformers=={transformers.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
