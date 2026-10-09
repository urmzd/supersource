# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.4 golden fixture: scaled RoPE frequencies
and ALiBi.

Each case runs one of transformers' ROPE_INIT_FUNCTIONS (5.19.0, float32) on
a LlamaConfig and records (inv_freq, attention_scaling) with the arguments
of contracts/py/tinyllm/modern/ctxext.pyi that should reproduce it:

  linear-*   "linear"
  ntk-*      "dynamic" evaluated at seq_len = 2 * max_position_embeddings with
             factor 2: base * (2 * 2 - 1) ** (r / (r - 2)), the static NTK rule
             with factor 3
  yarn-*     "yarn" (default betas; Qwen2.5-style betas and sizes; partial rotary)
  llama3-*   "llama3" (Llama 3.1 8B and Llama 3.2 1B values)
  default-*  the default ladder

ALiBi: BLOOM's build_alibi_tensor (float32) for several head counts over 7
key positions, alibi[h, 0, j] = slope_h * j.

    uv run --offline --script course/oracle/L7.4/rope_scaling_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.bloom.modeling_bloom import build_alibi_tensor
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

OUT = Path("course/fixtures/L7.4/rope_scaling_hf.json")

CASES = [
    # name, head_dim, rope_parameters (HF), max_pos, seq_len for dynamic, our kwargs
    ("default-64", 64, {"rope_type": "default", "rope_theta": 1e5}, 2048, None,
     {"d_rot": 64, "base": 1e5, "kind": "default"}),
    ("linear-64-x4", 64, {"rope_type": "linear", "rope_theta": 1e5, "factor": 4.0}, 2048, None,
     {"d_rot": 64, "base": 1e5, "kind": "linear", "factor": 4.0}),
    ("linear-128-x2", 128, {"rope_type": "linear", "rope_theta": 1e4, "factor": 2.0}, 4096, None,
     {"d_rot": 128, "base": 1e4, "kind": "linear", "factor": 2.0}),
    ("ntk-64-x3", 64, {"rope_type": "dynamic", "rope_theta": 1e4, "factor": 2.0}, 2048, 4096,
     {"d_rot": 64, "base": 1e4, "kind": "ntk", "factor": 3.0}),
    ("ntk-128-x3", 128, {"rope_type": "dynamic", "rope_theta": 5e5, "factor": 2.0}, 8192, 16384,
     {"d_rot": 128, "base": 5e5, "kind": "ntk", "factor": 3.0}),
    ("yarn-64-x4", 64, {"rope_type": "yarn", "rope_theta": 1e5, "factor": 4.0,
                        "original_max_position_embeddings": 2048}, 8192, None,
     {"d_rot": 64, "base": 1e5, "kind": "yarn", "factor": 4.0, "original_max_pos": 2048}),
    ("yarn-128-x8-betas", 128, {"rope_type": "yarn", "rope_theta": 1e6, "factor": 8.0, "beta_fast": 16.0,
                                "beta_slow": 2.0, "original_max_position_embeddings": 32768}, 262144, None,
     {"d_rot": 128, "base": 1e6, "kind": "yarn", "factor": 8.0, "original_max_pos": 32768,
      "beta_fast": 16.0, "beta_slow": 2.0}),
    ("yarn-partial-32-x2", 64, {"rope_type": "yarn", "rope_theta": 1e4, "factor": 2.0, "partial_rotary_factor": 0.5,
                                "original_max_position_embeddings": 4096}, 8192, None,
     {"d_rot": 32, "base": 1e4, "kind": "yarn", "factor": 2.0, "original_max_pos": 4096}),
    ("llama3-128-x8", 128, {"rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0, "low_freq_factor": 1.0,
                            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}, 131072, None,
     {"d_rot": 128, "base": 5e5, "kind": "llama3", "factor": 8.0, "original_max_pos": 8192,
      "low_freq_factor": 1.0, "high_freq_factor": 4.0}),
    ("llama3-64-x32", 64, {"rope_type": "llama3", "rope_theta": 5e5, "factor": 32.0, "low_freq_factor": 1.0,
                           "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}, 131072, None,
     {"d_rot": 64, "base": 5e5, "kind": "llama3", "factor": 32.0, "original_max_pos": 8192,
      "low_freq_factor": 1.0, "high_freq_factor": 4.0}),
]
HEADS = [1, 3, 6, 8, 12]


def main() -> None:
    rope = []
    for name, dh, params, max_pos, seq_len, ours in CASES:
        cfg = LlamaConfig(hidden_size=dh * 2, num_attention_heads=2, head_dim=dh,
                          max_position_embeddings=max_pos, rope_parameters=dict(params))
        if params["rope_type"] == "default":
            inv, scale = LlamaRotaryEmbedding.compute_default_rope_parameters(cfg)
        else:
            fn = ROPE_INIT_FUNCTIONS[params["rope_type"]]
            inv, scale = fn(cfg, None, seq_len) if seq_len else fn(cfg, None)
        assert inv.dtype == torch.float32 and inv.shape == (ours["d_rot"] // 2,), (name, inv.shape)
        rope.append({"name": name, "hf_rope_parameters": params, "max_position_embeddings": max_pos,
                     "seq_len": seq_len, "args": ours, "inv_freq": inv.tolist(),
                     "attention_scaling": float(scale)})
    alibi = []
    for n in HEADS:
        a = build_alibi_tensor(torch.ones(1, 7, dtype=torch.int64), n, torch.float32)  # [n, 1, 7]
        alibi.append({"n_heads": n, "bias": a[:, 0, :].tolist()})
    doc = {
        "generator": "course/oracle/L7.4/rope_scaling_hf.py",
        "library": f"torch=={torch.__version__} transformers=={transformers.__version__}",
        "rope": rope,
        "alibi_bloom": alibi,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L7.4/rope_scaling_hf.py\t"
          f"torch=={torch.__version__},transformers=={transformers.__version__}\t-\tApache-2.0")


main()
