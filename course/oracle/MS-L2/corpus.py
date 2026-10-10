# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the MS-L2, L2.2, and L2.3 fixtures.

The design's MS-L2 trains on `course-corpora/ts-train.bin`, a TinyStories
slice tokenized by the learner's BPE and published as a release asset. That
asset is not published (nothing is published from this batch), so the
milestone, the NPLM learning test (L2.2), and the word2vec tests (L2.3) use a
small synthetic stand-in instead (DEVIATIONS B54-01): short children's
stories in the TinyStories style, generated here from a fixed grammar by a
PCG32 (spec/pcg32.md), so the text is original, Apache-2.0, and reproducible
byte for byte.

The grammar puts words of one semantic class in the same slots (animals do
animal things, foods are eaten and baked, places are gone to), which is what
makes distributional embeddings separate the classes. The same class table
gives L2.3 its word-similarity gold scores:

    3  same class (cat, dog)
    1  same part of speech, different class (cat, apple: both nouns)
    0  different part of speech (cat, red)

Outputs (byte tokens, formats/tokens-bin.md version 1, vocab_size 256; one
story per line, stories joined by "\\n"):

    course/fixtures/MS-L2/train.bin   the first TRAIN_STORIES stories
    course/fixtures/MS-L2/val.bin     the next VAL_STORIES stories (disjoint)
    course/fixtures/L2.3/wordsim.json the classes and the gold pairs

It also prints the MS-L2 bigram bar: the mean NLL per byte on val.bin of the
add-one count bigram of L0.0 (alpha 1, V 256) fitted on train.bin, in exact
rational arithmetic for the counts and math.fsum for the logs, and the
MANIFEST.tsv rows.

    uv run --script course/oracle/MS-L2/corpus.py

Run from the repo root.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

TRAIN_STORIES, VAL_STORIES = 560, 60
SEED = 20261009
MASK64 = (1 << 64) - 1


class PCG32:
    """spec/pcg32.md: PCG-XSH-RR 64/32, pcg32_srandom_r seeding."""

    def __init__(self, seed: int, seq: int = 54) -> None:
        self.state, self.inc = 0, ((seq << 1) | 1) & MASK64
        self.next_u32()
        self.state = (self.state + seed) & MASK64
        self.next_u32()

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & MASK64
        xs = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & 0xFFFFFFFF

    def below(self, n: int) -> int:
        t = ((1 << 32) - n) % n
        while True:
            r = self.next_u32()
            if r >= t:
                return r % n

    def pick(self, xs):
        return xs[self.below(len(xs))]


# word class -> (part of speech, words). Every word is lower case ASCII.
CLASSES = {
    "animal": ("noun", ["cat", "dog", "bird", "frog", "duck", "bunny", "bear", "fox"]),
    "food": (
        "noun",
        ["apple", "cake", "cookie", "bread", "soup", "pie", "carrot", "cheese"],
    ),
    "toy": (
        "noun",
        ["ball", "doll", "kite", "truck", "drum", "train", "boat", "blocks"],
    ),
    "place": (
        "noun",
        ["park", "garden", "forest", "beach", "farm", "school", "river", "hill"],
    ),
    "color": (
        "adj",
        ["red", "blue", "green", "yellow", "pink", "purple", "brown", "orange"],
    ),
    "feeling": (
        "adj",
        ["happy", "sad", "scared", "excited", "tired", "proud", "angry", "calm"],
    ),
    "move": (
        "verb",
        ["ran", "walked", "jumped", "hopped", "raced", "skipped", "marched", "danced"],
    ),
    "eat": (
        "verb",
        ["ate", "baked", "shared", "cooked", "tasted", "bought", "found", "packed"],
    ),
}
GIRLS = ["Lily", "Mia", "Anna", "Sue", "Lucy", "Emma", "Rose", "Zoe"]
BOYS = ["Tom", "Ben", "Max", "Sam", "Tim", "Jack", "Leo", "Finn"]
SIZES = ["big", "small", "little", "tiny"]
TASTES = ["yummy", "sweet", "warm", "soft"]
SOUNDS = {
    "cat": "meow",
    "dog": "woof",
    "bird": "tweet",
    "frog": "ribbit",
    "duck": "quack",
    "bunny": "squeak",
    "bear": "growl",
    "fox": "yip",
}


def art(word: str) -> str:
    """The indefinite article and the word: "an apple", "a cake"."""
    return ("an " if word[0] in "aeiou" else "a ") + word


def story(g: PCG32) -> str:
    """One story: an opening, four to seven middle sentences, an ending."""
    girl = g.below(2) == 0
    name = g.pick(GIRLS if girl else BOYS)
    friend = g.pick(BOYS if girl else GIRLS)
    he, his = ("she", "her") if girl else ("he", "his")
    W = {k: v[1] for k, v in CLASSES.items()}
    animal, toy, place, color = (
        g.pick(W["animal"]),
        g.pick(W["toy"]),
        g.pick(W["place"]),
        g.pick(W["color"]),
    )
    s = [
        g.pick(
            [
                f"Once upon a time, there was a {g.pick(SIZES)} {'girl' if girl else 'boy'} named {name}.",
                f"One day, {name} went to the {place} with {his} mom.",
                f"There was a {g.pick(SIZES)} {animal} who lived near the {place}.",
            ]
        )
    ]
    middles = [
        lambda: f"{name} had {art(color)} {toy}.",
        lambda: f"{name} saw {art(g.pick(W['animal']))} in the {g.pick(W['place'])}.",
        lambda: f"The {animal} {g.pick(W['move'])} to the {g.pick(W['place'])}.",
        lambda: f'The {animal} said, "{SOUNDS[animal].capitalize()}!"',
        lambda: f"{name} {g.pick(W['eat'])} {art(g.pick(W['food']))} with {friend}.",
        lambda: f"The {g.pick(W['food'])} was {g.pick(TASTES)}.",
        lambda: (
            f"{name} and {friend} played with the {g.pick(W['color'])} {g.pick(W['toy'])}."
        ),
        lambda: (
            f"{he.capitalize()} {g.pick(W['move'])} home and felt {g.pick(W['feeling'])}."
        ),
        lambda: f"{name} gave the {g.pick(W['animal'])} some {g.pick(W['food'])}.",
        lambda: (
            f"The {g.pick(W['animal'])} was {g.pick(W['feeling'])} because it lost its {g.pick(W['toy'])}."
        ),
        lambda: (
            f"{friend} {g.pick(W['eat'])} the {g.pick(W['food'])} at the {g.pick(W['place'])}."
        ),
        lambda: f'"I love my {g.pick(W["color"])} {g.pick(W["toy"])}!" said {name}.',
        lambda: f"They {g.pick(W['move'])} around the {g.pick(W['place'])} all day.",
        lambda: f"{name} felt {g.pick(W['feeling'])} and {g.pick(W['feeling'])}.",
    ]
    for _ in range(4 + g.below(4)):
        s.append(g.pick(middles)())
    s.append(
        g.pick(
            [
                f"{name} was {g.pick(W['feeling'])} and went to sleep.",
                "The end.",
                f"From that day on, {name} and the {animal} were best friends.",
                f"{name} hugged {his} {color} {toy} and smiled.",
            ]
        )
    )
    return " ".join(s)


def tokens_bin(path: Path, data: bytes) -> bytes:
    head = struct.pack("<4i", 20240520, 1, len(data), 256) + b"\0" * (1024 - 16)
    blob = head + struct.pack(f"<{len(data)}H", *data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob


def bigram_nll(train: bytes, val: bytes) -> float:
    """Mean NLL per predicted byte on val of L0.0's add-one bigram fitted on train."""
    V, alpha = 256, 1
    counts: dict[tuple[int, int], int] = {}
    rows = [0] * V
    for a, b in zip(train, train[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
        rows[a] += 1
    terms = [
        -(math.log(counts.get((a, b), 0) + alpha) - math.log(rows[a] + alpha * V))
        for a, b in zip(val, val[1:])
    ]
    return math.fsum(terms) / len(terms)


def wordsim() -> dict:
    """Gold pairs: every same-class pair, plus a seeded sample of the others."""
    g = PCG32(SEED, seq=7)
    words = [(w, k, v[0]) for k, v in CLASSES.items() for w in v[1]]
    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            (a, ka, pa), (b, kb, pb) = words[i], words[j]
            score = 3 if ka == kb else (1 if pa == pb else 0)
            pairs.append([a, b, score])
    # A fixed subsample keeps the file small and the three levels balanced.
    keep = [p for p in pairs if p[2] == 3]
    ones = [p for p in pairs if p[2] == 1]
    zeros = [p for p in pairs if p[2] == 0]
    keep += [ones[g.below(len(ones))] for _ in range(len(keep))]
    keep += [zeros[g.below(len(zeros))] for _ in range(len(keep) // 2)]
    seen, out = set(), []
    for p in keep:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            out.append(p)
    return {
        "generator": "course/oracle/MS-L2/corpus.py",
        "scores": {
            "3": "same class",
            "1": "same part of speech, different class",
            "0": "different part of speech",
        },
        "classes": {k: {"pos": v[0], "words": v[1]} for k, v in CLASSES.items()},
        "pairs": out,
    }


def main() -> None:
    g = PCG32(SEED)
    stories = [story(g) for _ in range(TRAIN_STORIES + VAL_STORIES)]
    train = ("\n".join(stories[:TRAIN_STORIES]) + "\n").encode("ascii")
    val = ("\n".join(stories[TRAIN_STORIES:]) + "\n").encode("ascii")
    rows = []
    for rel, data in (("MS-L2/train.bin", train), ("MS-L2/val.bin", val)):
        blob = tokens_bin(Path("course/fixtures") / rel, data)
        rows.append((rel, blob))
    ws = (json.dumps(wordsim(), indent=1) + "\n").encode()
    p = Path("course/fixtures/L2.3/wordsim.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(ws)
    rows.append(("L2.3/wordsim.json", ws))
    print(f"train {len(train)} bytes, val {len(val)} bytes")
    print(f"bigram (add-one, train -> val) nll per byte: {bigram_nll(train, val):.9f}")
    for rel, blob in rows:
        print(
            "\t".join(
                [
                    f"course/fixtures/{rel}",
                    hashlib.sha256(blob).hexdigest(),
                    str(len(blob)),
                    "course/oracle/MS-L2/corpus.py",
                    "-",
                    "-",
                    "Apache-2.0",
                ]
            )
        )


if __name__ == "__main__":
    main()
