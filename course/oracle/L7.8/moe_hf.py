# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.8 golden fixture: mixture-of-experts
blocks of transformers 5.19.0, float32.

HF 5.x stores the experts fused (gate_up_proj [E, 2f, d], down_proj
[E, d, f]); the fixture splits them into the per-expert keys tinyllm's MoE
registers (experts.<e>.gate_proj.weight = gate_up_proj[e, :f],
experts.<e>.up_proj.weight = gate_up_proj[e, f:],
experts.<e>.down_proj.weight = down_proj[e]), which are also the keys of
DeepSeek and Qwen checkpoints on the Hub.

  mixtral.*   MixtralSparseMoeBlock, d 16, 4 experts of f 12, top 2,
              softmax then top-k then renormalize (router softmax_topk,
              norm_topk); x [2, 5, 16]; out, router logits [10, 4],
              topk_idx and topk_w [10, 2]; aux = HF load_balancing_loss_func
              on the logits (top 2, no mask)
  qwen3.*     Qwen3MoeSparseMoeBlock, d 16, 4 experts of f 8, top 2,
              norm_topk_prob False (weights are the raw softmax values)
  deepseek.*  DeepseekV3MoE, d 16, 8 routed experts of f 6, top 2, 2 shared
              experts (one MLP of width 12), sigmoid router with a nonzero
              e_score_correction_bias, n_group 1, norm_topk_prob True,
              routed_scaling_factor 2.5

    uv run --offline --script course/oracle/L7.8/moe_hf.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import DeepseekV3Config, MixtralConfig, Qwen3MoeConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.mixtral.modeling_mixtral import (
    MixtralSparseMoeBlock,
    load_balancing_loss_func,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

OUT = Path("course/fixtures/L7.8/moe_hf.npz")
SEED = 20261014
rng = np.random.default_rng(SEED)


def fill(block: torch.nn.Module, prefix: str, out: dict, f: int) -> None:
    with torch.no_grad():
        for name, p in block.named_parameters():
            fan_in = p.shape[-1]
            p.copy_(
                torch.tensor(
                    rng.standard_normal(p.shape) / np.sqrt(fan_in), dtype=torch.float32
                )
            )
            a = p.numpy().copy()
            if name == "experts.gate_up_proj":
                for e in range(a.shape[0]):
                    out[f"{prefix}.experts.{e}.gate_proj.weight"] = a[e, :f]
                    out[f"{prefix}.experts.{e}.up_proj.weight"] = a[e, f:]
            elif name == "experts.down_proj":
                for e in range(a.shape[0]):
                    out[f"{prefix}.experts.{e}.down_proj.weight"] = a[e]
            else:
                out[f"{prefix}.{name}"] = a


def run(block, prefix: str, out: dict, x: torch.Tensor) -> None:
    with torch.no_grad():
        logits, w, idx = block.gate(x.reshape(-1, x.shape[-1]))
        y = block(x)
    order = np.argsort(
        idx.numpy(), axis=-1, kind="stable"
    )  # topk order is unspecified; store by expert id
    out[f"{prefix}.x"], out[f"{prefix}.out"] = x.numpy(), y.numpy()
    out[f"{prefix}.router_logits"] = logits.numpy()
    out[f"{prefix}.topk_idx"] = np.take_along_axis(idx.numpy(), order, -1).astype(
        np.int64
    )
    out[f"{prefix}.topk_w"] = np.take_along_axis(w.numpy(), order, -1)


def main() -> None:
    out: dict = {}
    x = torch.tensor(rng.standard_normal((2, 5, 16)), dtype=torch.float32)

    mc = MixtralConfig(
        hidden_size=16, intermediate_size=12, num_local_experts=4, num_experts_per_tok=2
    )
    mix = MixtralSparseMoeBlock(mc).eval()
    fill(mix, "mixtral", out, 12)
    run(mix, "mixtral", out, x)
    aux = load_balancing_loss_func((torch.tensor(out["mixtral.router_logits"]),), 4, 2)
    out["mixtral.aux"] = np.array(float(aux), dtype=np.float64)

    qc = Qwen3MoeConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
    )
    qw = Qwen3MoeSparseMoeBlock(qc).eval()
    fill(qw, "qwen3", out, 8)
    run(qw, "qwen3", out, x)

    dc = DeepseekV3Config(
        hidden_size=16,
        moe_intermediate_size=6,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
    )
    ds = DeepseekV3MoE(dc).eval()
    fill(ds, "deepseek", out, 6)
    with torch.no_grad():
        ds.gate.e_score_correction_bias.copy_(
            torch.tensor(0.3 * rng.standard_normal(8), dtype=torch.float32)
        )
    out["deepseek.gate.e_score_correction_bias"] = (
        ds.gate.e_score_correction_bias.numpy().copy()
    )
    run(ds, "deepseek", out, x)

    out["__meta__"] = np.array(
        json.dumps(
            {
                "generator": "course/oracle/L7.8/moe_hf.py",
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
                "course/oracle/L7.8/moe_hf.py",
                f"torch=={torch.__version__},transformers=={transformers.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
