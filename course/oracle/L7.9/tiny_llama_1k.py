# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for course/fixtures/L7.9/tiny-llama-2l-1k: the
weights of tiny-llama-2l unchanged, with max_position_embeddings raised from
256 to 1024. The model uses RoPE (no learned position table), so the same
weights serve the longer context and every output inside the first 256
positions is the same.

Why: the engine's PR conformance includes tool calls, whose prompt carries
the tool's JSON schema and the call format in the system message. With the
byte tokenizer that prompt is about 470 tokens, and the case asks for up to
64 more, past the 256-position context of tiny-llama-2l. MS-L10 and the
reference system serve this copy; everything that pins outputs of the
original model keeps using course/fixtures/L7.9/tiny-llama-2l.

    uv run --script course/oracle/L7.9/tiny_llama_1k.py

Run from the repo root and paste the printed rows into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

SRC = Path("course/fixtures/L7.9/tiny-llama-2l")
DST = Path("course/fixtures/L7.9/tiny-llama-2l-1k")
CONTEXT = 1024


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "config.json").read_text())
    cfg["max_position_embeddings"] = CONTEXT
    (DST / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(SRC / "model.safetensors", DST / "model.safetensors")
    for name in ("config.json", "model.safetensors"):
        p = DST / name
        data = p.read_bytes()
        print(
            "\t".join(
                [
                    str(p),
                    hashlib.sha256(data).hexdigest(),
                    str(len(data)),
                    "course/oracle/L7.9/tiny_llama_1k.py",
                    "torch==2.14.1,transformers==5.19.0",
                    "-",
                    "Apache-2.0",
                ]
            )
        )


if __name__ == "__main__":
    main()
