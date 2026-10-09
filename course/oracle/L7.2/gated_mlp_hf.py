# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.2 golden fixture: Hugging Face gated MLPs.

Three cases, float32, d = 8, d_ff = 12, x [2, 3, 8], forward and the
gradients of sum(y * g) for x and every parameter (HF names):

  swiglu       transformers LlamaMLP, hidden_act "silu", mlp_bias False
  swiglu-bias  LlamaMLP with mlp_bias True
  geglu        transformers GemmaMLP, hidden_act "gelu_pytorch_tanh"

    uv run --offline --script course/oracle/L7.2/gated_mlp_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GemmaConfig, LlamaConfig
from transformers.models.gemma.modeling_gemma import GemmaMLP
from transformers.models.llama.modeling_llama import LlamaMLP

OUT = Path("course/fixtures/L7.2/gated_mlp_hf.npz")
SEED = 20261010
rng = np.random.default_rng(SEED)
D, FF = 8, 12


def case(name: str, mod, out: dict) -> None:
    with torch.no_grad():
        for pname, p in mod.named_parameters():
            w = rng.uniform(-0.5, 0.5, size=tuple(p.shape)).astype(np.float32)
            p.copy_(torch.tensor(w))
            out[f"{name}.param.{pname}"] = w
    x = rng.normal(size=(2, 3, D)).astype(np.float32)
    g = rng.normal(size=(2, 3, D)).astype(np.float32)
    tx = torch.tensor(x, requires_grad=True)
    y = mod(tx)
    (y * torch.tensor(g)).sum().backward()
    out.update({f"{name}.x": x, f"{name}.g": g, f"{name}.y": y.detach().numpy(), f"{name}.grad.x": tx.grad.numpy()})
    for pname, p in mod.named_parameters():
        out[f"{name}.grad.{pname}"] = p.grad.numpy()


def main() -> None:
    out: dict = {}
    case("swiglu", LlamaMLP(LlamaConfig(hidden_size=D, intermediate_size=FF, hidden_act="silu", num_attention_heads=2, num_key_value_heads=2)), out)
    case("swiglu-bias", LlamaMLP(LlamaConfig(hidden_size=D, intermediate_size=FF, hidden_act="silu", mlp_bias=True, num_attention_heads=2, num_key_value_heads=2)), out)
    case("geglu", GemmaMLP(GemmaConfig(hidden_size=D, intermediate_size=FF, hidden_act="gelu_pytorch_tanh", num_attention_heads=2, num_key_value_heads=2, head_dim=4)), out)
    out["__meta__"] = np.array(json.dumps({
        "generator": "course/oracle/L7.2/gated_mlp_hf.py", "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "seed": SEED,
        "cases": {"swiglu": {"act": "silu", "bias": False}, "swiglu-bias": {"act": "silu", "bias": True},
                  "geglu": {"act": "gelu_tanh", "bias": False}}, "d": D, "d_ff": FF,
    }))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    data = OUT.read_bytes()
    print(f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L7.2/gated_mlp_hf.py\t"
          f"torch=={torch.__version__},transformers=={transformers.__version__}\t-\tApache-2.0")


main()
