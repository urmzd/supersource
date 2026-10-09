# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the M00.3 golden frequency ladders.

inv_freq: Hugging Face transformers' default RoPE initialisation
(LlamaRotaryEmbedding.compute_default_rope_parameters), computed by torch in
float32, for real head sizes and bases (SmolLM2-135M: head_dim 64, theta 1e5;
Llama 2: 128, 1e4; Llama 3: 128, 5e5; a partial rotary dimension of 32).

alibi: transformers' BLOOM build_alibi_tensor (float32), the slopes of Press
et al. including head counts that are not powers of two. The slope of head
h is the bias it adds one position away, alibi[h, 0, 1].

    uv run --script course/oracle/M00.3/ladders_golden.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import LlamaConfig
from transformers.models.bloom.modeling_bloom import build_alibi_tensor
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

OUT = Path("course/fixtures/M00.3/ladders.json")

ROPE = [  # name, head_dim, rope_theta
    ("smollm2-135m", 64, 100000.0),
    ("llama-2", 128, 10000.0),
    ("llama-3", 128, 500000.0),
    ("partial-rotary-32", 32, 10000.0),
    ("chapter-example", 8, 10000.0),
]
HEADS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]


def main() -> None:
    rope = []
    for name, d, theta in ROPE:
        cfg = LlamaConfig(
            hidden_size=d, num_attention_heads=1, head_dim=d, rope_theta=theta
        )
        inv, _ = LlamaRotaryEmbedding.compute_default_rope_parameters(cfg)
        assert inv.dtype == torch.float32 and inv.shape == (d // 2,)
        rope.append({"name": name, "d_rot": d, "base": theta, "inv_freq": inv.tolist()})
    alibi = []
    for n in HEADS:
        a = build_alibi_tensor(torch.ones(1, 2, dtype=torch.int64), n, torch.float32)
        alibi.append({"n_heads": n, "slopes": a[:, 0, 1].tolist()})
    doc = {
        "generator": "course/oracle/M00.3/ladders_golden.py",
        "library": f"torch=={torch.__version__} transformers=={transformers.__version__}",
        "dtype": "float32",
        "rope_inv_freq": rope,
        "alibi_slopes": alibi,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M00.3/ladders_golden.py",
                f"torch=={torch.__version__},transformers=={transformers.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
