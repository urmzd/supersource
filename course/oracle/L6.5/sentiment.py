# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Maintainer generator for course/fixtures/small-corpora/sst2-2k.tsv: a
synthetic, binary sentiment set in the shape of SST-2 (one short lowercase
sentence and a 0/1 label per row), the stand-in for the design's "SST-2 2k
subset" (DEVIATIONS B73-05: the SST-2 data has no clear license to commit).

Rows are drawn from templates over small word lists, so a model must read
more than single words to be right: negation ("is not good" is negative),
contrast ("dull but the ending is wonderful": the clause after "but"
decides), and verbs of liking. 2000 rows, columns split (train: the first
1600, val: the last 400), sentence, label (1 positive, 0 negative); both
labels equally often in each split. ASCII only, so the byte tokenizer (D32)
reads it and the bytes 0 to 4 stay free for special tokens.

    uv run --offline --python 3.12 --script course/oracle/L6.5/sentiment.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

OUT = Path("course/fixtures/small-corpora/sst2-2k.tsv")
SEED = 20261011
N, N_TRAIN = 2000, 1600

SUBJ = [
    "the film",
    "this movie",
    "the plot",
    "the acting",
    "the story",
    "the cast",
    "the ending",
    "the script",
    "the music",
    "the pacing",
    "the dialogue",
    "the director's work",
]
POS = [
    "good",
    "great",
    "wonderful",
    "delightful",
    "charming",
    "brilliant",
    "moving",
    "funny",
    "gripping",
    "beautiful",
    "clever",
    "warm",
]
NEG = [
    "bad",
    "dull",
    "boring",
    "awful",
    "tedious",
    "clumsy",
    "flat",
    "weak",
    "lifeless",
    "messy",
    "shallow",
    "predictable",
]
INT = ["", "very ", "really ", "truly ", "quite ", "so "]
LIKE = ["loved", "enjoyed", "adored", "admired"]
DISLIKE = ["hated", "disliked", "regretted", "resented"]


def sentence(rng: np.random.Generator, label: int) -> str:
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    good, bad = (POS, NEG) if label == 1 else (NEG, POS)
    t = int(rng.integers(5))
    if t == 0:
        return f"{pick(SUBJ)} is {pick(INT)}{pick(good)}"
    if t == 1:
        return f"{pick(SUBJ)} is not {pick(bad)}"
    if t == 2:
        return (
            f"{pick(SUBJ)} is {pick(bad)} but {pick(SUBJ)} is {pick(INT)}{pick(good)}"
        )
    if t == 3:
        return f"i {pick(LIKE if label == 1 else DISLIKE)} {pick(SUBJ)}"
    return f"{pick(SUBJ)} was {pick(good)} and {pick(good)}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = ["split\tsentence\tlabel"]
    for i in range(N):
        label = i % 2
        split = "train" if i < N_TRAIN else "val"
        rows.append(f"{split}\t{sentence(rng, label)}\t{label}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n")
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L6.5/sentiment.py\t"
        f"numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
