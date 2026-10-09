# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the M06.2 golden segmentations.

Writes course/fixtures/M06.2/segmentation.json: a small subword vocabulary,
test strings, and for each string (a) the greedy longest-match segmentation
WordPiece uses (L1.3) and (b) every vocabulary match at every position, the
edges of the Unigram lattice (L1.4). Both are computed by brute force over
the whole vocabulary with str.startswith, never with a trie, so the course
test compares the learner's trie against an independent oracle.

    uv run --script course/oracle/M06.2/segment_golden.py

Run from the repo root, then update the row in course/fixtures/MANIFEST.tsv
with the sha256 and size the script prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("course/fixtures/M06.2/segmentation.json")

VOCAB = [
    "a",
    "ab",
    "abl",
    "able",
    "b",
    "be",
    "bel",
    "believ",
    "believe",
    "c",
    "d",
    "e",
    "ed",
    "er",
    "ers",
    "f",
    "h",
    "i",
    "in",
    "ing",
    "l",
    "li",
    "lie",
    "n",
    "o",
    "r",
    "re",
    "s",
    "t",
    "th",
    "the",
    "u",
    "un",
    "unbe",
    "unbeliev",
    "v",
    "w",
    "x",
    "y",
    "z",
    "é",
    "café",
    "é",
    "\U0001f642",
    "\U0001f642\U0001f642",
    " ",
    "  ",
    " the",
]
STRINGS = [
    "unbelievable",
    "believers",
    "the",
    " the",
    "  the",
    "thethe",
    "unbe",
    "unbelie",
    "café",
    "café",
    "\U0001f642\U0001f642\U0001f642",
    "ableable",
    "inline",
    "relied",
    "zzz",
    "qq",
    "",
    "a",
    "lie lie",
    "rebelling",
    "ununun",
    "théâtre",
    "x y z",
    "believe",
    "believ",
    "beliee",
]


def greedy(s: str, vocab: dict[str, int]) -> list[list[int]]:
    """[[length, id]] pieces; an unmatched position is [1, -1]."""
    out, i = [], 0
    while i < len(s):
        best = max((w for w in vocab if s.startswith(w, i)), key=len, default=None)
        if best is None:
            out.append([1, -1])
            i += 1
        else:
            out.append([len(best), vocab[best]])
            i += len(best)
    return out


def lattice(s: str, vocab: dict[str, int]) -> list[list[list[int]]]:
    """For each start position, every [length, id] match, shortest first."""
    return [
        sorted([len(w), vocab[w]] for w in vocab if s.startswith(w, i))
        for i in range(len(s) + 1)
    ]


def main() -> None:
    vocab = {w: i for i, w in enumerate(VOCAB)}
    assert len(vocab) == len(VOCAB), "duplicate vocabulary entry"
    cases = [
        {"s": s, "greedy": greedy(s, vocab), "lattice": lattice(s, vocab)}
        for s in STRINGS
    ]
    doc = {
        "generator": "course/oracle/M06.2/segment_golden.py",
        "note": "ids are positions in vocab; greedy [1, -1] marks a character no key matches",
        "vocab": VOCAB,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M06.2/segment_golden.py"
    )


if __name__ == "__main__":
    main()
