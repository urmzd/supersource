# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.7 golden fixture: sliding-window attention
(Mistral) and learned attention sinks (gpt-oss), transformers 5.19.0, eager,
float32.

  mistral.*   one MistralAttention layer (d 24, 4 query heads, 2 kv heads,
              head_dim 6, rope_theta 1e4, sliding_window 3) under HF's
              sliding-window causal mask (kv_idx > q_idx - 3), weights under
              HF's state_dict keys, x [1, 9, 24] at positions 0..8, out
  gptoss.*    gpt-oss's eager_attention_forward (repeat_kv groups of 2,
              learned sinks [4], scaling 5 ** -0.5) on random q [2, 4, 7, 5],
              k, v [2, 2, 7, 5]:
                win3:  all 7 queries, causal and window 3
                chunk: the last 3 queries (q_offset 4) over all 7 keys,
                       causal and window 3
                full:  all 7 queries, causal only
              out [B, H, Tq, 5] (transposed back from HF's [B, Tq, H, 5])
              and lse [B, H, Tq] = logsumexp over the visible scores and
              the sink logit

    uv run --offline --script course/oracle/L7.7/window_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import MistralConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    eager_attention_forward as gptoss_eager,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRotaryEmbedding,
)

OUT = Path("course/fixtures/L7.7/window_hf.npz")
SEED = 20261013
rng = np.random.default_rng(SEED)


def additive(Tq: int, Tk: int, window, q_offset: int = 0) -> torch.Tensor:
    i = np.arange(Tq)[:, None] + q_offset
    j = np.arange(Tk)[None, :]
    vis = j <= i
    if window is not None:
        vis &= j > i - window
    return torch.tensor(np.where(vis, 0.0, -np.inf), dtype=torch.float32)[None, None]


def mistral(out: dict) -> None:
    cfg = MistralConfig(
        hidden_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        sliding_window=3,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    cfg._attn_implementation = "eager"
    attn = MistralAttention(cfg, layer_idx=0).eval()
    with torch.no_grad():
        for name, p in attn.named_parameters():
            p.copy_(
                torch.tensor(
                    rng.standard_normal(p.shape) / np.sqrt(p.shape[1]),
                    dtype=torch.float32,
                )
            )
            out[f"mistral.{name}"] = p.numpy().copy()
    rot = MistralRotaryEmbedding(cfg)
    T = 9
    x = torch.tensor(rng.standard_normal((1, T, 24)), dtype=torch.float32)
    cos, sin = rot(x, torch.arange(T)[None])
    with torch.no_grad():
        y, _ = attn(x, (cos, sin), additive(T, T, 3))
    out["mistral.x"], out["mistral.out"] = x.numpy(), y.numpy()
    out["mistral.inv_freq"] = rot.inv_freq.numpy().copy()


def gptoss(out: dict) -> None:
    B, H, Hkv, T, dh = 2, 4, 2, 7, 5
    q = torch.tensor(rng.standard_normal((B, H, T, dh)), dtype=torch.float32)
    k = torch.tensor(rng.standard_normal((B, Hkv, T, dh)), dtype=torch.float32)
    v = torch.tensor(rng.standard_normal((B, Hkv, T, dh)), dtype=torch.float32)
    sinks = torch.tensor(rng.standard_normal(H), dtype=torch.float32)
    mod = SimpleNamespace(num_key_value_groups=H // Hkv, sinks=sinks, training=False)
    out.update(
        {
            "gptoss.q": q.numpy(),
            "gptoss.k": k.numpy(),
            "gptoss.v": v.numpy(),
            "gptoss.sinks": sinks.numpy(),
        }
    )
    scale = dh**-0.5
    for name, qq, window, off in (
        ("win3", q, 3, 0),
        ("chunk", q[:, :, 4:], 3, 4),
        ("full", q, None, 0),
    ):
        mask = additive(qq.shape[2], T, window, off)
        o, _ = gptoss_eager(mod, qq, k, v, mask, scaling=scale)
        kk = k.repeat_interleave(H // Hkv, dim=1)
        s = qq @ kk.transpose(2, 3) * scale + mask
        comb = torch.cat(
            [s, sinks.reshape(1, H, 1, 1).expand(B, H, qq.shape[2], 1)], dim=-1
        )
        out[f"gptoss.{name}.out"] = o.transpose(1, 2).numpy()
        out[f"gptoss.{name}.lse"] = torch.logsumexp(comb, dim=-1).numpy()


def main() -> None:
    out: dict = {}
    mistral(out)
    gptoss(out)
    out["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L7.7/window_hf.py",
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
                "course/oracle/L7.7/window_hf.py",
                f"torch=={torch.__version__},transformers=={transformers.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
