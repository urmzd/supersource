# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers==4.46.3"]
# ///
"""Maintainer generator for the M05.2 golden byte map.

Writes course/fixtures/M05.2/gpt2_bytes_to_unicode.json: the GPT-2 byte map
as produced by Hugging Face transformers (`bytes_to_unicode` in
models/gpt2/tokenization_gpt2.py, the port of openai/gpt-2 src/encoder.py),
cross-checked against the 256-character alphabet of the Rust `tokenizers`
ByteLevel pre-tokenizer. No PyTorch is needed.

    uv run --script course/oracle/M05.2/gpt2_byte_map.py

Run from the repo root, then update the row in course/fixtures/MANIFEST.tsv
with the sha256 and size the script prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tokenizers
import transformers
from tokenizers.pre_tokenizers import ByteLevel
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

OUT = Path("course/fixtures/M05.2/gpt2_bytes_to_unicode.json")


def main() -> None:
    table = bytes_to_unicode()
    assert sorted(table) == list(range(256)), "domain is not 0..255"
    alphabet = set(ByteLevel.alphabet())
    assert set(table.values()) == alphabet, "transformers and tokenizers disagree"
    doc = {
        "generator": "course/oracle/M05.2/gpt2_byte_map.py",
        "upstream": f"transformers=={transformers.__version__} bytes_to_unicode; "
        f"tokenizers=={tokenizers.__version__} ByteLevel.alphabet()",
        "note": "code_points[b] is the code point of the character byte b maps to",
        "code_points": [ord(table[b]) for b in range(256)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1) + "\n").encode()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\t"
        f"course/oracle/M05.2/gpt2_byte_map.py\ttransformers=={transformers.__version__}"
    )


if __name__ == "__main__":
    main()
