"""Maintainer generator for course/fixtures/MS-L6/sst2-train.bin: the train
split's sentences of course/fixtures/small-corpora/sst2-2k.tsv, one per line,
as a formats/tokens-bin.md shard of byte ids (version 1, uint16, vocab 256).
MS-L6 pretrains its BERT and ELECTRA on it before the LoRA fine-tune: a short,
in-domain pretraining corpus, so the 400-step byte models already know the
words the classifier needs (DEVIATIONS B73-06). Stdlib and numpy only.

    course/harness/.venv/bin/python course/oracle/MS-L6/sst2_train_bin.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

SRC = Path("course/fixtures/small-corpora/sst2-2k.tsv")
OUT = Path("course/fixtures/MS-L6/sst2-train.bin")


def main() -> None:
    rows = [
        line.split("\t") for line in SRC.read_text().splitlines()[1:] if line.strip()
    ]
    text = ("\n".join(r[1] for r in rows if r[0] == "train") + "\n").encode("ascii")
    header = struct.pack("<256i", *([20240520, 1, len(text), 256] + [0] * 252))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(
        header + np.frombuffer(text, dtype=np.uint8).astype("<u2").tobytes()
    )
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/MS-L6/sst2_train_bin.py\t"
        f"numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
