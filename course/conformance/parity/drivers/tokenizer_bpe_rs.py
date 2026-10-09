"""Parity driver for `tokenizer.bpe`, implementation `rust-via-pyo3` (L1.5):
reads one JSON case per stdin line ({"text": ...}) and prints the GPT-2 ids
of that text as a JSON list, computed by tinyllm_rs.Bpe (your tl-tok through
your tl-py) loaded from $TINYLLM_FIXTURES/tok-gpt2/tokenizer.json.
"""

import json
import os
import sys

import tinyllm_rs

tok = tinyllm_rs.Bpe.from_hf_json(
    os.path.join(os.environ["TINYLLM_FIXTURES"], "tok-gpt2", "tokenizer.json")
)
for line in sys.stdin:
    if line.strip():
        print(json.dumps(tok.encode(json.loads(line)["text"])), flush=False)
sys.stdout.flush()
