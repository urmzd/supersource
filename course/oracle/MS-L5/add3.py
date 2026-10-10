# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Maintainer generator for the MS-L5 addition task (stdlib only).

Pairs a + b with a, b in 0..999, written LEAST significant digit first, the
way a decoder that writes left to right can carry: the source is the three
digits of a reversed, "+", the three digits of b reversed (zero padded), and
the target is the four digits of a + b reversed (zero padded). For example
a = 123, b = 456: source "321+654", target "9750" (579 reversed and padded).

    small-corpora/add3.tsv        4000 training pairs
    small-corpora/add3-test.tsv    500 test pairs, none of them in training

Pairs are drawn without replacement from all 10^6 by random.Random(SEED).

    uv run --offline --python 3.12 --script course/oracle/MS-L5/add3.py

Run from the repo root, then update the two MANIFEST.tsv rows it prints.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

SEED = 20261015
N_TRAIN, N_TEST = 4000, 500
OUT = Path("course/fixtures/small-corpora")


def row(k: int) -> str:
    a, b = divmod(k, 1000)
    return f"{a:03d}"[::-1] + "+" + f"{b:03d}"[::-1] + "\t" + f"{a + b:04d}"[::-1]


def main() -> None:
    picks = random.Random(SEED).sample(range(1000 * 1000), N_TRAIN + N_TEST)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ks in (("add3.tsv", picks[:N_TRAIN]), ("add3-test.tsv", picks[N_TRAIN:])):
        p = OUT / name
        p.write_text("".join(row(k) + "\n" for k in ks))
        data = p.read_bytes()
        print(
            f"{p}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/MS-L5/add3.py\t"
            f"python-stdlib\t-\tApache-2.0"
        )


main()
