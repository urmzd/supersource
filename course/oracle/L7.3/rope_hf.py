# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.3 golden fixture: rotary embeddings.

All float32. cos and sin tables are stored one value per pair ([..., r/2]),
the first half of HF's duplicated [..., r] tables.

  table        LlamaRotaryEmbedding (head_dim 16, rope_theta 1e4) at positions
               0, 1, 2, 7, 100, 4095, 4096, 8191, 65535: cos, sin, and HF's
               float32 inv_freq (computed as a float32 power; the float64
               ladder rounded to float32 can differ by one ulp, which moves
               the angle at position 65535 by about 1e-4)
  table-yarn   the same with rope_parameters yarn (factor 4, original 2048):
               cos and sin already multiplied by attention_scaling
  half         q, k [2, 3, 5, 16] through transformers' apply_rotary_pos_emb
               with per-row positions (row 0: 0..4, row 1: 1000..1004)
  interleaved  q [2, 5, 3, 16] (Meta's [B, T, H, dh] layout) through Meta's
               llama apply_rotary_emb: torch.polar frequencies, complex
               product (pairs (x[2i], x[2i+1])), positions 0..4; stored
               transposed to [B, H, T, dh]
  partial      q, k [2, 3, 5, 16] through GPT-NeoX's apply_rotary_pos_emb
               with rotary_dim 8 (the last 8 entries pass through)

    uv run --offline --script course/oracle/L7.3/rope_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import LlamaConfig
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb as neox_apply
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

OUT = Path("course/fixtures/L7.3/rope_hf.npz")
SEED = 20261011
rng = np.random.default_rng(SEED)
DH, THETA = 16, 10000.0
POS = np.array([0, 1, 2, 7, 100, 4095, 4096, 8191, 65535], dtype=np.int64)


def rotary(**rope) -> LlamaRotaryEmbedding:
    params = {"rope_type": "default", "rope_theta": THETA}
    params.update(rope)
    cfg = LlamaConfig(hidden_size=DH * 2, num_attention_heads=2, head_dim=DH, max_position_embeddings=2048,
                      rope_parameters=params)
    return LlamaRotaryEmbedding(cfg)


def table(emb, pos: np.ndarray):
    x = torch.zeros(1, dtype=torch.float32)
    cos, sin = emb(x, torch.tensor(pos)[None])
    h = cos.shape[-1] // 2
    assert torch.equal(cos[..., :h], cos[..., h:])
    return cos[0, :, :h].numpy(), sin[0, :, :h].numpy()


def meta_interleaved(xq: torch.Tensor, theta: float, T: int) -> torch.Tensor:
    # Meta llama/model.py: precompute_freqs_cis and apply_rotary_emb
    dim = xq.shape[-1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(T, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    fc = freqs_cis.view(1, T, 1, dim // 2)
    return torch.view_as_real(xq_ * fc).flatten(3)


def main() -> None:
    out: dict = {}
    base = rotary()
    c, s = table(base, POS)
    out.update({"table.positions": POS, "table.cos": c, "table.sin": s, "table.inv_freq": base.inv_freq.numpy()})
    emb = rotary(rope_type="yarn", factor=4.0, original_max_position_embeddings=2048)
    c, s = table(emb, POS)
    out.update({"table-yarn.cos": c, "table-yarn.sin": s, "table-yarn.inv_freq": emb.inv_freq.numpy(),
                "table-yarn.attention_scaling": np.array(emb.attention_scaling, dtype=np.float64)})

    pos = np.array([[0, 1, 2, 3, 4], [1000, 1001, 1002, 1003, 1004]], dtype=np.int64)
    q = rng.normal(size=(2, 3, 5, DH)).astype(np.float32)
    k = rng.normal(size=(2, 3, 5, DH)).astype(np.float32)
    cos, sin = rotary()(torch.zeros(1), torch.tensor(pos))
    qe, ke = apply_rotary_pos_emb(torch.tensor(q), torch.tensor(k), cos, sin)
    out.update({"half.positions": pos, "half.q": q, "half.k": k, "half.q_out": qe.numpy(), "half.k_out": ke.numpy()})

    xq = rng.normal(size=(2, 5, 3, DH)).astype(np.float32)
    xo = meta_interleaved(torch.tensor(xq), THETA, 5)
    out.update({"interleaved.q": xq.transpose(0, 2, 1, 3).copy(),
                "interleaved.q_out": xo.numpy().transpose(0, 2, 1, 3).copy()})

    cfg = LlamaConfig(hidden_size=16, num_attention_heads=2, head_dim=8, rope_parameters={"rope_type": "default", "rope_theta": THETA})
    cos8, sin8 = LlamaRotaryEmbedding(cfg)(torch.zeros(1), torch.arange(5)[None])
    q = rng.normal(size=(2, 3, 5, DH)).astype(np.float32)
    k = rng.normal(size=(2, 3, 5, DH)).astype(np.float32)
    qe, ke = neox_apply(torch.tensor(q), torch.tensor(k), cos8, sin8)
    out.update({"partial.q": q, "partial.k": k, "partial.q_out": qe.numpy(), "partial.k_out": ke.numpy()})

    out["__meta__"] = np.array(json.dumps({
        "generator": "course/oracle/L7.3/rope_hf.py", "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "seed": SEED,
        "head_dim": DH, "rope_theta": THETA, "partial_rotary_dim": 8,
        "yarn": {"factor": 4.0, "original_max_position_embeddings": 2048},
    }))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    data = OUT.read_bytes()
    print(f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L7.3/rope_hf.py\t"
          f"torch=={torch.__version__},transformers=={transformers.__version__}\t-\tApache-2.0")


main()
