# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.5 golden fixture: grouped-query attention.

d = 24, 6 query heads over 2 kv heads (n_rep 3), head_dim 8 (so H dh = 48 is
not d), B = 2, T = 6, rope_theta 1e4, eager attention, float32. Weights are
uniform(-0.3, 0.3) copied in by HF name; each case records the output and
the gradients of sum(out * g) for x and every parameter.

  llama        transformers LlamaAttention, causal mask, positions 0..5
  llama-mask   the same plus an extra mask hiding key 1 from every query of row 1
  qwen2-bias   transformers Qwen2Attention (q, k, v biases; o_proj without)
  gptoss       transformers GptOssAttention (attention_bias False): learned
               sinks and a sliding window of 3 (query t sees keys t-2..t),
               per-row positions (row 0: 0..5, row 1: 0, 2, 3, 7, 8, 12: gaps,
               so the row really rotates differently)

Masks are HF's additive form (0 or float32 min); every query keeps at least
one visible key, so HF's finite fill and the course's -inf agree.

    uv run --offline --script course/oracle/L7.5/gqa_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GptOssConfig, LlamaConfig, Qwen2Config
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding

OUT = Path("course/fixtures/L7.5/gqa_hf.npz")
SEED = 20261012
rng = np.random.default_rng(SEED)
D, H, HKV, DH, B, T, THETA, WINDOW = 24, 6, 2, 8, 2, 6, 1e4, 3
COMMON = dict(hidden_size=D, num_attention_heads=H, num_key_value_heads=HKV, head_dim=DH,
              max_position_embeddings=64, rope_parameters={"rope_type": "default", "rope_theta": THETA})


def additive(visible: np.ndarray) -> torch.Tensor:
    m = np.where(visible, 0.0, np.finfo(np.float32).min).astype(np.float32)
    return torch.tensor(m)


def causal(window=None) -> np.ndarray:
    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    vis = j <= i
    if window is not None:
        vis &= (i - j) < window
    return np.broadcast_to(vis, (B, 1, T, T)).copy()


def run(name: str, attn, rotary, positions: np.ndarray, visible: np.ndarray, out: dict) -> None:
    attn.config._attn_implementation = "eager"
    with torch.no_grad():
        for pname, p in attn.named_parameters():
            w = rng.uniform(-0.3, 0.3, size=tuple(p.shape)).astype(np.float32)
            p.copy_(torch.tensor(w))
            out[f"{name}.param.{pname}"] = w
    x = rng.normal(size=(B, T, D)).astype(np.float32)
    g = rng.normal(size=(B, T, D)).astype(np.float32)
    tx = torch.tensor(x, requires_grad=True)
    pos = torch.tensor(np.broadcast_to(positions, (B, T)).copy())
    cos, sin = rotary(tx, pos)
    y, _ = attn(tx, position_embeddings=(cos, sin), attention_mask=additive(visible))
    (y * torch.tensor(g)).sum().backward()
    out.update({f"{name}.x": x, f"{name}.g": g, f"{name}.y": y.detach().numpy(), f"{name}.grad.x": tx.grad.numpy(),
                f"{name}.positions": positions, f"{name}.visible": visible,
                f"{name}.inv_freq": rotary.inv_freq.numpy()})
    for pname, p in attn.named_parameters():
        out[f"{name}.grad.{pname}"] = p.grad.numpy()


def main() -> None:
    out: dict = {}
    pos = np.arange(T, dtype=np.int64)
    lc = LlamaConfig(**COMMON)
    run("llama", LlamaAttention(lc, 0), LlamaRotaryEmbedding(lc), pos, causal(), out)
    vis = causal()
    vis[1, :, :, 1] = False
    run("llama-mask", LlamaAttention(lc, 0), LlamaRotaryEmbedding(lc), pos, vis, out)
    qc = Qwen2Config(**COMMON)
    run("qwen2-bias", Qwen2Attention(qc, 0), Qwen2RotaryEmbedding(qc), pos, causal(), out)
    gc = GptOssConfig(**COMMON, attention_bias=False, sliding_window=WINDOW, layer_types=["sliding_attention"],
                      num_hidden_layers=1)
    gpos = np.array([[0, 1, 2, 3, 4, 5], [0, 2, 3, 7, 8, 12]], dtype=np.int64)
    run("gptoss", GptOssAttention(gc, 0), GptOssRotaryEmbedding(gc), gpos, causal(WINDOW), out)
    out["__meta__"] = np.array(json.dumps({
        "generator": "course/oracle/L7.5/gqa_hf.py", "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "seed": SEED,
        "d": D, "n_heads": H, "n_kv_heads": HKV, "d_head": DH, "B": B, "T": T, "rope_theta": THETA,
        "window": WINDOW,
    }))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    data = OUT.read_bytes()
    print(f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L7.5/gqa_hf.py\t"
          f"torch=={torch.__version__},transformers=={transformers.__version__}\t-\tApache-2.0")


main()
