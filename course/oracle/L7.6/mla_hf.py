# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.6 golden fixture: DeepSeek-V3 multi-head
latent attention (transformers 5.19.0 DeepseekV3Attention, eager, float32).

Two tiny configurations, weights drawn from numpy (seed below) and stored
under HF's state_dict keys with a prefix:

  A.*   d 32, 4 heads, q_lora_rank None (a plain q_proj), kv_lora_rank 16,
        qk_nope 8, qk_rope 4, v 6, rope_interleave True, no bias,
        positions 0..4 shared by the batch
  B.*   the same widths with q_lora_rank 12 (q_a_proj, q_a_layernorm,
        q_b_proj), attention_bias True, rope_interleave False (half layout),
        per-row positions (row 0: 0..4, row 1: 3..7)

For each: x [2, 5, 32], positions, out (the attention output under the
causal mask), and what DeepseekV3Attention hands its cache: c (the
normalized latent [2, 5, 16]) and, for B only, k_rope (the rotated rope
key [2, 5, 4]; with rope_interleave HF returns it de-interleaved, so A's is
not comparable entry by entry). inv_freq is HF's float32 table.

    uv run --offline --script course/oracle/L7.6/mla_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RotaryEmbedding,
)

OUT = Path("course/fixtures/L7.6/mla_hf.npz")
SEED = 20261012
rng = np.random.default_rng(SEED)


class Recorder:
    """Stands in for HF's Cache: records what the layer would store."""

    def __init__(self):
        self.c = self.k = None

    def update(self, kv_nope, k_rot, layer_idx):
        self.c, self.k = kv_nope.detach().clone(), k_rot.detach().clone()
        return kv_nope, k_rot


def make(
    prefix: str, q_lora_rank, interleave: bool, bias: bool, pos: np.ndarray, out: dict
) -> None:
    cfg = DeepseekV3Config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=6,
        rope_interleave=interleave,
        attention_bias=bias,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    cfg._attn_implementation = "eager"
    attn = DeepseekV3Attention(cfg, layer_idx=0).eval()
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if name.endswith("layernorm.weight"):
                v = 1.0 + 0.2 * rng.standard_normal(p.shape)
            elif name.endswith("bias"):
                v = 0.1 * rng.standard_normal(p.shape)
            else:
                v = rng.standard_normal(p.shape) / np.sqrt(p.shape[1])
            p.copy_(torch.tensor(v, dtype=torch.float32))
            out[f"{prefix}.{name}"] = p.numpy().copy()
    rot = DeepseekV3RotaryEmbedding(cfg)
    B, T = 2, 5
    x = torch.tensor(rng.standard_normal((B, T, 32)), dtype=torch.float32)
    pid = torch.tensor(np.broadcast_to(pos, (B, T)).copy())
    cos, sin = rot(x, pid)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None].expand(B, 1, T, T)
    rec = Recorder()
    with torch.no_grad():
        y, _ = attn(x, (cos, sin), mask, past_key_values=rec)
    out[f"{prefix}.x"] = x.numpy()
    out[f"{prefix}.positions"] = pos.astype(np.int64)
    out[f"{prefix}.out"] = y.numpy()
    out[f"{prefix}.c"] = rec.c[:, 0].numpy()
    if not interleave:
        out[f"{prefix}.k_rope"] = rec.k[:, 0].numpy()
    out[f"{prefix}.inv_freq"] = rot.inv_freq.numpy().copy()
    out[f"{prefix}.softmax_scale"] = np.array(attn.scaling, dtype=np.float64)


def main() -> None:
    out: dict = {}
    make("A", None, True, False, np.arange(5), out)
    make("B", 12, False, True, np.array([[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]]), out)
    out["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L7.6/mla_hf.py",
                "seed": SEED,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            }
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/L7.6/mla_hf.py",
                f"torch=={torch.__version__},transformers=={transformers.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
