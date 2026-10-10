# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "transformers==5.19.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the MS-L7 nightly fixture smollm2-parity:
HuggingFaceTB/SmolLM2-135M-Instruct at revision
12fd25f77366fa6b3b4b768ec3050bf629380bac (the smollm2-135m-instruct row of
course/fixtures/ASSETS.tsv; Apache-2.0), read from the local Hugging Face
cache with HF_HUB_OFFLINE=1, run in float32 (the BF16 weights upcast).

From the candidate prompts it keeps the first 8 whose greedy continuation
has a top-2 logit margin >= 1e-3 at each of 32 steps (DESIGN 5.7), and writes
course/fixtures/smollm2-parity/:

  prompts.txt       the 8 prompts, one per line
  prompts.jsonl     {"prompt", "ids"}: HF's token ids of each prompt (no
                    special tokens added)
  greedy_32.json    {"ids": the 8 continuations concatenated in file order,
                     "margins": the oracle's top-2 margin at every step,
                     "per_prompt": [[...] x 8]}
  logits_top.npz    per prompt i: top{i}.ids and top{i}.values, the 256
                    largest next-token logits after the prompt (float32)
  params.json       {"params": num_parameters()}

    HF_HUB_OFFLINE=1 uv run --offline --script course/oracle/MS-L7/smollm2_parity.py

Run from the repo root and paste the printed rows into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

REPO, REV = (
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "12fd25f77366fa6b3b4b768ec3050bf629380bac",
)
OUT = Path("course/fixtures/smollm2-parity")
CANDIDATES = [
    "The capital of France is",
    "Once upon a time, there was a",
    "The three primary colors are",
    "Water boils at",
    "def fibonacci(n):",
    "The largest planet in our solar system is",
    "To make a cup of tea, first",
    "Photosynthesis is the process by which",
    "In 1492, Columbus",
    "The quick brown fox",
    "My favorite book is",
    "A neural network is",
    "The opposite of hot is",
    "Monday, Tuesday, Wednesday,",
    "import numpy as np\n",
    "Hello! How are",
]


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = LlamaForCausalLM.from_pretrained(
        REPO, revision=REV, dtype=torch.float32
    ).eval()
    kept, rows, ids_all, margins_all, per, tops = [], [], [], [], [], {}
    for p in CANDIDATES:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        gen, margins = [], []
        with torch.no_grad():
            out = model(torch.tensor([ids]), use_cache=True)
            past, row = out.past_key_values, out.logits[0, -1]
            first = row.clone()
            for _ in range(32):
                top = torch.topk(row, 2).values
                margins.append(float(top[0] - top[1]))
                nxt = int(torch.argmax(row))
                gen.append(nxt)
                out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
                past, row = out.past_key_values, out.logits[0, -1]
        if min(margins) < 1e-3:
            continue
        i = len(kept)
        kept.append(p)
        rows.append({"prompt": p, "ids": ids})
        ids_all += gen
        margins_all += margins
        per.append(gen)
        v, ix = torch.topk(first, 256)
        tops[f"top{i}.ids"], tops[f"top{i}.values"] = (
            ix.numpy().astype(np.int64),
            v.numpy(),
        )
        if len(kept) == 8:
            break
    assert len(kept) == 8, kept
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prompts.txt").write_text(
        "\n".join(json.dumps(p)[1:-1] if "\n" in p else p for p in kept) + "\n"
    )
    (OUT / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "greedy_32.json").write_text(
        json.dumps({"ids": ids_all, "margins": margins_all, "per_prompt": per}) + "\n"
    )
    np.savez(OUT / "logits_top.npz", **tops)
    (OUT / "params.json").write_text(
        json.dumps(
            {"params": int(model.num_parameters()), "repo": REPO, "revision": REV}
        )
        + "\n"
    )
    vers = f"torch=={torch.__version__},transformers=={transformers.__version__}"
    for f in sorted(OUT.iterdir()):
        data = f.read_bytes()
        print(
            "\t".join(
                [
                    str(f),
                    hashlib.sha256(data).hexdigest(),
                    str(len(data)),
                    "course/oracle/MS-L7/smollm2_parity.py",
                    vers,
                    f"{REPO}@{REV}",
                    "Apache-2.0",
                ]
            )
        )


if __name__ == "__main__":
    main()
