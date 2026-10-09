# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.2.6"]
# ///
"""Maintainer generator for the craft.05 golden fixture: forward values and
torch autograd gradients, float64, of the four ops of the craft.05 kata
(linear, RMSNorm, the SwiGLU gate, scaled dot-product attention), for a
given upstream gradient g of the output.

  linear.*        x [3, 4], W [5, 4], b [5], g [3, 5]: y = x W^T + b; gx, gW, gb
  linear_sq.*     the same with a square W [4, 4] (a transposed weight
                  gradient has the right shape here)
  rmsnorm.*       x [3, 6], w [6], eps 1e-6, g [3, 6]: y = x / sqrt(mean(x^2) + eps) * w; gx, gw
  swiglu.*        a, b [3, 5], g: h = silu(a) * b; ga, gb
  attn.*          q [2, 3, 4], k [2, 5, 4], v [2, 5, 2], g [2, 3, 2]:
                  o = softmax(q k^T / 2) v; gq, gk, gv
  attn_causal.*   q, k [2, 4, 4], v [2, 4, 2], the causal mask (key j <= query i)

    uv run --offline --script course/oracle/craft.05/oracles_torch.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/craft.05/oracles.npz")
SEED = 20261016
rng = np.random.default_rng(SEED)


def t(*shape):
    return torch.tensor(
        rng.standard_normal(shape), dtype=torch.float64, requires_grad=True
    )


def record(out: dict, prefix: str, inputs: dict, y: torch.Tensor) -> None:
    g = torch.tensor(rng.standard_normal(tuple(y.shape)), dtype=torch.float64)
    y.backward(g)
    for k, v in inputs.items():
        out[f"{prefix}.{k}"] = v.detach().numpy()
        out[f"{prefix}.g{k}"] = v.grad.numpy()
    out[f"{prefix}.y"], out[f"{prefix}.g"] = y.detach().numpy(), g.numpy()


def main() -> None:
    out: dict = {}
    for prefix, n_out in (("linear", 5), ("linear_sq", 4)):
        x, W, b = t(3, 4), t(n_out, 4), t(n_out)
        record(out, prefix, {"x": x, "W": W, "b": b}, x @ W.T + b)
    x, w = t(3, 6), t(6)
    record(
        out,
        "rmsnorm",
        {"x": x, "w": w},
        x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w,
    )
    a, b = t(3, 5), t(3, 5)
    record(out, "swiglu", {"a": a, "b": b}, torch.nn.functional.silu(a) * b)
    q, k, v = t(2, 3, 4), t(2, 5, 4), t(2, 5, 2)
    record(
        out,
        "attn",
        {"q": q, "k": k, "v": v},
        torch.nn.functional.scaled_dot_product_attention(q, k, v),
    )
    q, k, v = t(2, 4, 4), t(2, 4, 4), t(2, 4, 2)
    record(
        out,
        "attn_causal",
        {"q": q, "k": k, "v": v},
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
    )
    out["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/craft.05/oracles_torch.py",
                "seed": SEED,
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
                "course/oracle/craft.05/oracles_torch.py",
                f"torch=={torch.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
