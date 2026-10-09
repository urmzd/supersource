# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the M05.1 golden parameter counts.

For each architecture, Hugging Face transformers builds the model on the meta
device (no memory, no weights) and reports `num_parameters()`, plus the same
total split by parameter name into the five components of
contracts/py/tinyllm/accounting.pyi (embed, attn, mlp, norm, lm_head):

  smollm2-135m      the pinned SmolLM2-135M-Instruct config.json (same
                    architecture as the base model); its total is also
                    re-counted from the safetensors header of the pinned
                    checkpoint when the file is in the local HF cache
  llama-2-7b        LlamaConfig() defaults (Llama 2 7B: MHA, untied)
  mixtral-8x7b      MixtralConfig() defaults (GQA 32/8, 8 experts, top 2)
  deepseek-v3       DeepseekV3Config() defaults (MLA with q_lora_rank 1536,
                    3 dense layers then 256 routed + 1 shared experts, top 8)
  tiny-*            small random configs covering qkv bias, untied heads,
                    MLA without q compression, and MoE with shared experts

Nothing is downloaded: the meta device needs only the configs.

    uv run --offline --script course/oracle/M05.1/hf_param_counts.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    DeepseekV3Config,
    LlamaConfig,
    MixtralConfig,
    Qwen2Config,
)

OUT = Path("course/fixtures/M05.1/hf_param_counts.json")
SMOL_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/"
    "12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
SMOL_CONFIG_SHA = "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a"  # ASSETS.tsv


def component(name: str) -> str:
    if "embed_tokens" in name:
        return "embed"
    if name.startswith("lm_head"):
        return "lm_head"
    if "norm" in name:  # input_layernorm, post_attention_layernorm, model.norm, q_a/kv_a_layernorm
        return "norm"
    if ".self_attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    raise ValueError(name)


def count(cfg) -> dict:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    parts = {"embed": 0, "attn": 0, "mlp": 0, "norm": 0, "lm_head": 0}
    for name, p in model.named_parameters():  # tied weights appear once
        parts[component(name)] += p.numel()
    parts["total"] = model.num_parameters()
    assert parts["total"] == sum(v for k, v in parts.items() if k != "total"), parts
    return parts


def safetensors_total(path: str) -> int:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    total = 0
    for k, v in header.items():
        if k != "__metadata__":
            size = 1
            for s in v["shape"]:
                size *= s
            total += size
    return total


def fields(cfg, keys) -> dict:
    return {k: getattr(cfg, k, None) for k in keys}


LLAMA_KEYS = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "tie_word_embeddings", "attention_bias", "mlp_bias"]
MIXTRAL_KEYS = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "tie_word_embeddings", "num_local_experts", "num_experts_per_tok"]
DS_KEYS = ["vocab_size", "hidden_size", "intermediate_size", "moe_intermediate_size", "num_hidden_layers",
           "num_attention_heads", "num_key_value_heads", "n_shared_experts", "n_routed_experts",
           "num_experts_per_tok", "first_k_dense_replace", "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim",
           "qk_nope_head_dim", "v_head_dim", "tie_word_embeddings", "attention_bias"]


def main() -> None:
    cases = []

    raw = Path(SMOL_SNAPSHOT, "config.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SMOL_CONFIG_SHA, "config.json is not the pinned revision"
    smol = LlamaConfig(**{k: v for k, v in json.loads(raw).items() if k not in ("transformers.js_config",)})
    c = {"name": "smollm2-135m", "model_type": "llama", "config": fields(smol, LLAMA_KEYS), "hf": count(smol)}
    st = glob.glob(os.path.join(SMOL_SNAPSHOT, "model.safetensors"))
    if st:
        c["safetensors_total"] = safetensors_total(st[0])
        assert c["safetensors_total"] == c["hf"]["total"]
    cases.append(c)

    l2 = LlamaConfig()
    cases.append({"name": "llama-2-7b", "model_type": "llama", "config": fields(l2, LLAMA_KEYS), "hf": count(l2)})
    mx = MixtralConfig()
    cases.append({"name": "mixtral-8x7b", "model_type": "mixtral", "config": fields(mx, MIXTRAL_KEYS), "hf": count(mx)})
    ds = DeepseekV3Config()
    cases.append({"name": "deepseek-v3", "model_type": "deepseek_v3", "config": fields(ds, DS_KEYS), "hf": count(ds)})

    tq = Qwen2Config(vocab_size=101, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                     num_attention_heads=6, num_key_value_heads=2, tie_word_embeddings=False)
    qkeys = [k for k in LLAMA_KEYS if k not in ("attention_bias", "mlp_bias")]
    cases.append({"name": "tiny-qwen2-bias", "model_type": "qwen2", "config": fields(tq, qkeys), "hf": count(tq)})
    tl = LlamaConfig(vocab_size=97, hidden_size=40, intermediate_size=112, num_hidden_layers=2,
                     num_attention_heads=5, num_key_value_heads=5, head_dim=12, tie_word_embeddings=False)
    cases.append({"name": "tiny-llama-mha-headdim", "model_type": "llama", "config": fields(tl, LLAMA_KEYS), "hf": count(tl)})
    tm = MixtralConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=1, num_local_experts=4,
                       num_experts_per_tok=2, tie_word_embeddings=True)
    cases.append({"name": "tiny-mixtral-mqa-tied", "model_type": "mixtral", "config": fields(tm, MIXTRAL_KEYS), "hf": count(tm)})
    td = DeepseekV3Config(vocab_size=80, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
                          num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                          n_shared_experts=2, n_routed_experts=6, num_experts_per_tok=2,
                          first_k_dense_replace=1, kv_lora_rank=12, q_lora_rank=None,
                          qk_rope_head_dim=4, qk_nope_head_dim=8, v_head_dim=6, n_group=1, topk_group=1,
                          tie_word_embeddings=False)
    cases.append({"name": "tiny-deepseek-mla-moe", "model_type": "deepseek_v3", "config": fields(td, DS_KEYS), "hf": count(td)})
    td2 = DeepseekV3Config(vocab_size=50, hidden_size=24, intermediate_size=40, moe_intermediate_size=8,
                           num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=3,
                           n_shared_experts=1, n_routed_experts=4, num_experts_per_tok=1,
                           first_k_dense_replace=0, kv_lora_rank=10, q_lora_rank=6,
                           qk_rope_head_dim=2, qk_nope_head_dim=4, v_head_dim=4, n_group=1, topk_group=1,
                           tie_word_embeddings=True)
    cases.append({"name": "tiny-deepseek-qlora-tied", "model_type": "deepseek_v3", "config": fields(td2, DS_KEYS), "hf": count(td2)})

    doc = {
        "generator": "course/oracle/M05.1/hf_param_counts.py",
        "library": f"torch=={torch.__version__} transformers=={transformers.__version__}",
        "upstream": "HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac (config.json)",
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M05.1/hf_param_counts.py\t"
        f"torch=={torch.__version__},transformers=={transformers.__version__}\t"
        "HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac\tApache-2.0"
    )
    for c in cases:
        print(c["name"], c["hf"])


if __name__ == "__main__":
    main()
