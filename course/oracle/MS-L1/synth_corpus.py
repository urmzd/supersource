"""Maintainer generator for course/fixtures/MS-L1/train.txt (MS-L1 step 1).

    uv run --project course/harness python course/oracle/MS-L1/synth_corpus.py [--check]

A synthetic stand-in for the TinyStories 5 MB slice of DESIGN 4.3 MS-L1,
which lives in the unpublished `course-corpora` asset (DEVIATIONS B43-07):
800 short stories from a small seeded grammar, original text written for
the course (Apache-2.0), about 280 KB. Pure stdlib and a pure function of
SEED, so the bytes never change.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

SEED = 20261009
N = 800
NAMES = ["Lily", "Tom", "Mia", "Ben", "Sue", "Max", "Ana", "Leo", "Zoe", "Sam"]
ANIMALS = ["cat", "dog", "bird", "frog", "bunny", "fox", "bear", "duck"]
THINGS = ["ball", "kite", "box", "hat", "cake", "book", "boat", "drum", "cup", "shell"]
PLACES = [
    "the park",
    "the garden",
    "the beach",
    "the forest",
    "the house",
    "school",
    "the hill",
]
ADJ = ["big", "small", "red", "blue", "happy", "sad", "shiny", "soft", "old", "funny"]
FEEL = ["happy", "sad", "scared", "proud", "tired", "excited", "angry", "calm"]
VERB = ["found", "saw", "lost", "wanted", "made", "shared", "dropped", "liked"]


def story(rng: random.Random) -> str:
    a, b = rng.sample(NAMES, 2)
    an, th, pl = rng.choice(ANIMALS), rng.choice(THINGS), rng.choice(PLACES)
    s = [f"Once upon a time, there was a {rng.choice(ADJ)} {an} named {a}."]
    s.append(f"{a} liked to play in {pl} with a {rng.choice(ADJ)} {th}.")
    for _ in range(rng.randint(2, 6)):
        k = rng.randrange(5)
        if k == 0:
            s.append(
                f"One day, {a} {rng.choice(VERB)} a {rng.choice(ADJ)} {rng.choice(THINGS)}."
            )
        elif k == 1:
            s.append(
                f'"{rng.choice(["Look", "Wow", "Oh no", "Yay", "Help"])}!" said {a}. "I can\'t find my {th}."'
            )
        elif k == 2:
            s.append(
                f"{b} came to {pl} and they {rng.choice(VERB)} {rng.randint(2, 12)} {rng.choice(THINGS)}s."
            )
        elif k == 3:
            s.append(f"{a} felt {rng.choice(FEEL)}, but {b} was {rng.choice(FEEL)}.")
        else:
            s.append(
                f"They didn't know that it was {rng.randint(1, 9)} o'clock and time to go home."
            )
    s.append(
        f"In the end, {a} and {b} were {rng.choice(FEEL)} and they played until the sun went down."
    )
    return " ".join(s)


def main() -> int:
    rng = random.Random(SEED)
    text = "\n".join(story(rng) for _ in range(N)) + "\n"
    path = Path.cwd() / "course" / "fixtures" / "MS-L1" / "train.txt"
    if "--check" in sys.argv[1:]:
        same = path.is_file() and path.read_text() == text
        print(("same  " if same else "DIFF  ") + str(path))
        return 0 if same else 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path} ({len(text.encode())} bytes, {N} stories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
