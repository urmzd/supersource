# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L7.9 golden fixtures: tiny random
Llama-family models saved with transformers 5.19.0 save_pretrained, their
float32 logits, parameter counts, and margin-filtered greedy continuations.

Model directories (course/fixtures/L7.9/<name>/config.json, model.safetensors):

  tiny-llama-2l     LlamaForCausalLM, vocab 256 with the byte tokenizer
                    (config.json gains tl_arch "llama", tl_tokenizer
                    "bytes"), d 32, 2 layers, 4 query and 2 kv heads of 8,
                    d_ff 64, tied embeddings, rope_theta 1e4, rms 1e-5,
                    weights saved as BF16 (the MS-L7 smoke model)
  tiny-mistral-swa  MistralForCausalLM, vocab 64, d 24, MQA (4 query heads,
                    1 kv head of 6), sliding_window 4, untied, F32
  tiny-qwen2        Qwen2ForCausalLM (q, k, v biases), vocab 64, d 16, 2
                    heads of 8 (MHA), tied, F32
  tiny-llama3-rope  LlamaForCausalLM with rope_parameters llama3 (factor 8,
                    original 16), F32
  tiny-yarn         LlamaForCausalLM with rope_parameters yarn (factor 4,
                    original 16), F32

hf_logits.npz: per model `<name>.ids` [2, T] and `<name>.logits` [2, T, V]
(HF float32 forward of the saved weights; T = 12, or 40 for the two rope
scaling models so positions pass the original context).
hf_params.json: HF num_parameters() per model.
margin-prompts.txt, tiny_greedy_32.json, tiny_logits.json: prompts for
tiny-llama-2l whose greedy top-2 logit margin stays >= 1e-3 for 32 steps
(DESIGN 5.7 near-tie rule), their generated ids concatenated in file order
with the margin at every step, and the next-token logits of the first two
prompts (the MS-L7 numeric step).

    uv run --offline --script course/oracle/L7.9/llama_hf.py

Run from the repo root and paste the printed rows into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

DIR = Path("course/fixtures/L7.9")
SEED = 20261015
rng = np.random.default_rng(SEED)
VERS = f"torch=={torch.__version__},transformers=={transformers.__version__}"
GEN = "course/oracle/L7.9/llama_hf.py"
CANDIDATES = [
    "Once upon a time",
    "The cat sat on the",
    "Hello, world",
    "In the beginning",
    "A small model",
    "def main():",
    "1, 2, 3, 4,",
    "The quick brown fox",
    "She said that",
    "Rain fell on",
    "We are building",
    "abc",
    "Tiny llamas",
    "The end.",
]


def randomize(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                v = 1.0 + 0.1 * rng.standard_normal(p.shape)
            elif name.endswith("bias"):
                v = 0.2 * rng.standard_normal(p.shape)
            elif "embed_tokens" in name:
                v = rng.standard_normal(p.shape)
            else:
                v = rng.standard_normal(p.shape) * (1.5 / np.sqrt(p.shape[-1]))
            p.copy_(torch.tensor(v, dtype=torch.float32))


def save(model, name: str, dtype, extra: dict | None = None) -> Path:
    d = DIR / name
    d.mkdir(parents=True, exist_ok=True)
    model.to(dtype).save_pretrained(d, safe_serialization=True)
    for junk in ("generation_config.json",):
        (d / junk).unlink(missing_ok=True)
    if extra:
        cfg = json.loads((d / "config.json").read_text())
        cfg.update(extra)
        (d / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return d


def reload(cls, d: Path):
    return cls.from_pretrained(d, dtype=torch.float32).eval()


def main() -> None:
    logits: dict = {}
    params: dict = {}
    specs = [
        (
            "tiny-llama-2l",
            LlamaForCausalLM,
            LlamaConfig(
                vocab_size=256,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                rms_norm_eps=1e-5,
                tie_word_embeddings=True,
                max_position_embeddings=256,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            ),
            torch.bfloat16,
            {"tl_arch": "llama", "tl_tokenizer": "bytes"},
            12,
        ),
        (
            "tiny-mistral-swa",
            MistralForCausalLM,
            MistralConfig(
                vocab_size=64,
                hidden_size=24,
                intermediate_size=40,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=6,
                sliding_window=4,
                tie_word_embeddings=False,
                max_position_embeddings=64,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            ),
            torch.float32,
            None,
            12,
        ),
        (
            "tiny-qwen2",
            Qwen2ForCausalLM,
            Qwen2Config(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                tie_word_embeddings=True,
                max_position_embeddings=64,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            ),
            torch.float32,
            None,
            12,
        ),
        (
            "tiny-llama3-rope",
            LlamaForCausalLM,
            LlamaConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
                max_position_embeddings=128,
                rope_parameters={
                    "rope_type": "llama3",
                    "rope_theta": 10000.0,
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 16,
                },
            ),
            torch.float32,
            None,
            40,
        ),
        (
            "tiny-yarn",
            LlamaForCausalLM,
            LlamaConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
                max_position_embeddings=64,
                rope_parameters={
                    "rope_type": "yarn",
                    "rope_theta": 10000.0,
                    "factor": 4.0,
                    "original_max_position_embeddings": 16,
                },
            ),
            torch.float32,
            None,
            40,
        ),
    ]
    for name, cls, cfg, dtype, extra, T in specs:
        torch.manual_seed(0)
        m = cls(cfg)
        randomize(m)
        d = save(m, name, dtype, extra)
        m = reload(cls, d)
        params[name] = int(m.num_parameters())
        ids = rng.integers(0, cfg.vocab_size, size=(2, T))
        with torch.no_grad():
            out = m(torch.tensor(ids)).logits
        logits[f"{name}.ids"], logits[f"{name}.logits"] = (
            ids.astype(np.int64),
            out.numpy(),
        )
        if name == "tiny-llama-2l":
            tiny = m
    np.savez(DIR / "hf_logits.npz", **logits)
    (DIR / "hf_params.json").write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n"
    )

    # Margin-filtered greedy continuations of tiny-llama-2l (byte prompts).
    kept, all_ids, all_margins, per_prompt, first_logits = [], [], [], [], []
    for p in CANDIDATES:
        ids = list(p.encode("utf-8"))
        gen, margins = [], []
        with torch.no_grad():
            for step in range(32):
                row = tiny(torch.tensor([ids + gen])).logits[0, -1]
                if step == 0:
                    first = row.numpy().copy()
                top = torch.topk(row, 2).values
                margins.append(float(top[0] - top[1]))
                gen.append(int(torch.argmax(row)))
        if min(margins) >= 1e-3:
            kept.append(p)
            all_ids += gen
            all_margins += margins
            per_prompt.append(gen)
            first_logits.append([float(x) for x in first])
        if len(kept) == 4:
            break
    assert len(kept) == 4, kept
    (DIR / "margin-prompts.txt").write_text("\n".join(kept) + "\n")
    (DIR / "tiny_greedy_32.json").write_text(
        json.dumps(
            {
                "prompts": kept,
                "ids": all_ids,
                "margins": all_margins,
                "per_prompt": per_prompt,
            }
        )
        + "\n"
    )
    (DIR / "tiny_logits.json").write_text(
        json.dumps({"prompts": kept[:2], "logits": first_logits[:2]}) + "\n"
    )

    for f in sorted(DIR.rglob("*")):
        if f.is_file():
            data = f.read_bytes()
            print(
                "\t".join(
                    [
                        str(f),
                        hashlib.sha256(data).hexdigest(),
                        str(len(data)),
                        GEN,
                        VERS,
                        "-",
                        "Apache-2.0",
                    ]
                )
            )


if __name__ == "__main__":
    main()
