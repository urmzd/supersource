# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the MS-L0 milestone fixtures and bounds.

1. course/fixtures/MS-L0/corpus.bin: the bytes of course/fixtures/MS-P1/corpus.txt
   as a formats/tokens-bin.md file (version 1, vocab_size 256). The kill-and-
   resume steps read it through the learner's TokenStream (L0.6).
2. The count-MLE bound of the autograd-bigram step. Gradient descent on the
   mean cross-entropy of every (byte, next byte) pair minimizes

       L(W) = -(1/N) sum_t log softmax(W[x_t])[x_{t+1}]

   whose infimum over all [256, 256] tables is reached by the unsmoothed count
   model, p(j | i) = count(i, j) / count(i, *). Its NLL is the conditional
   entropy of the corpus,

       H = -(1/N) sum_{i,j} count(i, j) log(count(i, j) / count(i, *)),

   computed here with exact integer counts and math.fsum. No finite table
   reaches H (unseen pairs would need logits of minus infinity), so the
   milestone asks for nll <= H + 1e-3, rounded down to 5 decimals: within
   1e-3 nats of the count MLE.

    uv run --script course/oracle/MS-L0/fixtures.py

Run from the repo root. It rewrites corpus.bin and prints the bound for
MS-L0.toml and the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

SRC = Path("course/fixtures/MS-P1/corpus.txt")
OUT = Path("course/fixtures/MS-L0")
MAGIC, VERSION, VOCAB = 20240520, 1, 256


def tokens_bin(ids: bytes) -> bytes:
    header = [0] * 256
    header[0], header[1], header[2], header[3] = MAGIC, VERSION, len(ids), VOCAB
    return struct.pack("<256i", *header) + struct.pack(f"<{len(ids)}H", *ids)


def main() -> None:
    data = SRC.read_bytes()
    OUT.mkdir(parents=True, exist_ok=True)
    blob = tokens_bin(data)
    assert len(blob) == 1024 + 2 * len(data)
    (OUT / "corpus.bin").write_bytes(blob)

    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(data, data[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    rows: dict[int, int] = {}
    for (a, _), c in counts.items():
        rows[a] = rows.get(a, 0) + c
    n = len(data) - 1
    h = -math.fsum(c * math.log(c / rows[a]) for (a, _), c in counts.items()) / n
    print(f"pairs = {n}, distinct pairs = {len(counts)}, contexts = {len(rows)}")
    print(f"count-MLE NLL (conditional entropy) H = {h:.9f} nats/byte")
    print(
        f"MS-L0 autograd-bigram bound: nll <= {math.floor((h + 1e-3) * 1e5) / 1e5:.5f}"
    )
    raw = (OUT / "corpus.bin").read_bytes()
    print(
        "\t".join(
            [
                (OUT / "corpus.bin").as_posix(),
                hashlib.sha256(raw).hexdigest(),
                str(len(raw)),
                "course/oracle/MS-L0/fixtures.py",
                "-",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
