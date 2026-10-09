# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L5.2 torch golden fixture (attention masks).

Records, float64 unless noted:
  upper_left.<Tq>x<Tk>   torch.nn.attention.bias.causal_upper_left(Tq, Tk), materialized (bool)
  lower_right.<Tq>x<Tk>  torch.nn.attention.bias.causal_lower_right(Tq, Tk), materialized (bool)
  sdpa.*                 torch.nn.functional.scaled_dot_product_attention outputs on
                         q, k, v [2, 2, Tq, Tk-or-Tq, 4] for: is_causal=True (square),
                         the lower-right causal bias (Tq = 3, Tk = 7, a decode chunk),
                         and a bool key-padding mask [B, 1, 1, T] AND causal
                         (lengths 5 and 3), all float64
  window.*               a sliding-window (window 3) causal attention computed by
                         flex_attention's create_mask from the mask_mod
                         "kv <= q and q - kv < 3" on a 6 x 6 grid (bool)

    uv run --offline --python 3.12 --script course/oracle/L5.2/masks_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.nn.attention.bias import causal_lower_right, causal_upper_left

OUT = Path("course/fixtures/L5.2/masks_torch.npz")
torch.manual_seed(0)
gen = np.random.default_rng(20261009)


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    for tq, tk in [(4, 4), (3, 7), (1, 5), (5, 3)]:
        arrs[f"upper_left.{tq}x{tk}"] = causal_upper_left(tq, tk)._materialize().numpy()
        arrs[f"lower_right.{tq}x{tk}"] = (
            causal_lower_right(tq, tk)._materialize().numpy()
        )

    def qkv(tq, tk):
        q = gen.standard_normal((2, 2, tq, 4))
        k = gen.standard_normal((2, 2, tk, 4))
        v = gen.standard_normal((2, 2, tk, 4))
        return q, k, v

    q, k, v = qkv(5, 5)
    out = Fn.scaled_dot_product_attention(
        *map(torch.from_numpy, (q, k, v)), is_causal=True
    )
    arrs.update(
        {
            "sdpa.causal.q": q,
            "sdpa.causal.k": k,
            "sdpa.causal.v": v,
            "sdpa.causal.out": out.numpy(),
        }
    )

    q, k, v = qkv(3, 7)
    out = Fn.scaled_dot_product_attention(
        *map(torch.from_numpy, (q, k, v)), attn_mask=causal_lower_right(3, 7)
    )
    arrs.update(
        {
            "sdpa.decode.q": q,
            "sdpa.decode.k": k,
            "sdpa.decode.v": v,
            "sdpa.decode.out": out.numpy(),
        }
    )

    q, k, v = qkv(5, 5)
    lengths = np.array([5, 3])
    key = torch.arange(5)[None, :] < torch.from_numpy(lengths)[:, None]
    m = key[:, None, None, :] & torch.ones(5, 5, dtype=torch.bool).tril()
    out = Fn.scaled_dot_product_attention(
        *map(torch.from_numpy, (q, k, v)), attn_mask=m
    )
    arrs.update(
        {
            "sdpa.padded.q": q,
            "sdpa.padded.k": k,
            "sdpa.padded.v": v,
            "sdpa.padded.lengths": lengths,
            "sdpa.padded.out": out.numpy(),
        }
    )

    from torch.nn.attention.flex_attention import create_mask

    def window_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (q_idx - kv_idx < 3)

    arrs["window.6x6.w3"] = create_mask(window_mod, 1, 1, 6, 6, device="cpu")[
        0, 0
    ].numpy()

    meta = {
        "generator": "course/oracle/L5.2/masks_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L5.2/masks_torch.py\t"
        f"torch=={torch.__version__.split('+')[0]},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
