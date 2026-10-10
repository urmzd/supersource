"""Parity driver for `tokenizer.bpe`, implementation `python` (L1.2): reads
one JSON case per stdin line ({"text": ...}) and prints the GPT-2 ids of that
text as a JSON list, computed by your tinyllm.tok.bpe.BPETokenizer loaded from
$TINYLLM_FIXTURES/tok-gpt2/tokenizer.json. Python is the specification (P6):
the Rust port (L1.5) is compared with these ids in --fuzz mode.
"""

import json
import os
import sys

from tinyllm.tok.bpe import BPETokenizer

tok = BPETokenizer.from_hf_json(
    os.path.join(os.environ["TINYLLM_FIXTURES"], "tok-gpt2", "tokenizer.json")
)
for line in sys.stdin:
    if line.strip():
        print(json.dumps(tok.encode(json.loads(line)["text"])))
sys.stdout.flush()
