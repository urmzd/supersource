"""Maintainer generator for course/fixtures/L3.6/corpus.txt (stdlib only).

An original synthetic English text from a small grammar: short sentences
with subject-verb agreement, quoted speech, and numbered lines, so a
character-level model has structure to learn at several ranges (spelling,
word order, matching quotes, the counter). Public domain by construction
(CC0-1.0), 24 000 characters, deterministic for the seed.

    python3 course/oracle/L3.6/corpus.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

import hashlib
import random
from pathlib import Path

OUT = Path("course/fixtures/L3.6/corpus.txt")
rng = random.Random(20261009)

SING = [
    "the fox",
    "a girl",
    "the old man",
    "my cat",
    "the baker",
    "a small bird",
    "the king",
]
PLUR = [
    "the foxes",
    "two girls",
    "the old men",
    "our cats",
    "the bakers",
    "many birds",
    "the kings",
]
VERB_S = ["sees", "likes", "finds", "calls", "helps", "watches", "follows"]
VERB_P = ["see", "like", "find", "call", "help", "watch", "follow"]
OBJ = [
    "the river",
    "a red apple",
    "the green hill",
    "a quiet song",
    "the long road",
    "a warm fire",
]
SAY = ["hello", "come here", "look at that", "not today", "we are late", "it is cold"]
WHEN = ["in the morning", "at night", "after the rain", "every day", "before dinner"]


def sentence() -> str:
    one = rng.random() < 0.5
    if one:
        s = f"{rng.choice(SING)} {rng.choice(VERB_S)} {rng.choice(OBJ)}"
    else:
        s = f"{rng.choice(PLUR)} {rng.choice(VERB_P)} {rng.choice(OBJ)}"
    if rng.random() < 0.3:
        s += " " + rng.choice(WHEN)
    if rng.random() < 0.25:
        s += f', and {"says" if one else "say"} "{rng.choice(SAY)}"'
    return s[0].upper() + s[1:] + "."


def main() -> None:
    lines, n = [], 1
    while sum(len(x) + 1 for x in lines) < 24000:
        lines.append(f"{n}. " + " ".join(sentence() for _ in range(rng.randint(1, 3))))
        n += 1
    text = "\n".join(lines)[:24000].rsplit("\n", 1)[0] + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L3.6/corpus.py\tpython-stdlib\t-\tCC0-1.0"
    )


main()
