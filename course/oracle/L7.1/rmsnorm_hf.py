# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.1 golden fixture: Hugging Face RMSNorm.

Three cases, float32, forward and the gradients of sum(y * g) for x and the
weight:

  llama        transformers LlamaRMSNorm(16, eps=1e-5), random weight (offset 0)
  gemma        transformers GemmaRMSNorm(16, eps=1e-6), random weight: the gain
               is 1 + weight (offset 1)
  llama-small  LlamaRMSNorm(16, eps=1e-5) on inputs of size 1e-3, where
               mean(x^2) ~ 1e-6 is smaller than eps: the place of eps (inside
               the root) decides the output

    uv run --offline --script course/oracle/L7.1/rmsnorm_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

OUT = Path("course/fixtures/L7.1/rmsnorm_hf.npz")
SEED = 20261009
rng = np.random.default_rng(SEED)
D = 16


def case(name: str, mod, x: np.ndarray, out: dict) -> None:
    w = rng.uniform(-1.0, 1.0, size=D).astype(np.float32)
    g = rng.normal(size=x.shape).astype(np.float32)
    with torch.no_grad():
        mod.weight.copy_(torch.tensor(w))
    tx = torch.tensor(x, requires_grad=True)
    y = mod(tx)
    (y * torch.tensor(g)).sum().backward()
    out.update({f"{name}.x": x, f"{name}.weight": w, f"{name}.g": g, f"{name}.y": y.detach().numpy(),
                f"{name}.grad.x": tx.grad.numpy(), f"{name}.grad.weight": mod.weight.grad.numpy()})


def main() -> None:
    out: dict = {}
    case("llama", LlamaRMSNorm(D, eps=1e-5), rng.normal(size=(2, 3, D)).astype(np.float32), out)
    case("gemma", GemmaRMSNorm(D, eps=1e-6), rng.normal(size=(2, 3, D)).astype(np.float32), out)
    case("llama-small", LlamaRMSNorm(D, eps=1e-5), (1e-3 * rng.normal(size=(4, D))).astype(np.float32), out)
    out["__meta__"] = np.array(json.dumps({
        "generator": "course/oracle/L7.1/rmsnorm_hf.py", "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "seed": SEED,
        "cases": {"llama": {"eps": 1e-5, "offset": 0.0}, "gemma": {"eps": 1e-6, "offset": 1.0},
                  "llama-small": {"eps": 1e-5, "offset": 0.0}},
    }))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    data = OUT.read_bytes()
    print(f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L7.1/rmsnorm_hf.py\t"
          f"torch=={torch.__version__},transformers=={transformers.__version__}\t-\tApache-2.0")


main()
