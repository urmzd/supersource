# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the MS-corpus fixtures (course/fixtures/MS-corpus/).

Synthetic children's stories from templates and word lists (no third-party
text; Apache-2.0 like the repository), with every pipeline stage given
something to do, planted by construction:

  sources/stories.jsonl   JSON Lines, CC0-1.0: 140 stories of 100 to 160
                          words, plus 4 verbatim copies (exact dedup), 5
                          stories with one word changed (near dedup), 5 with
                          an email or phone number (PII), 3 that embed a
                          15-word span of a protected story (decontamination),
                          2 in French (language filter), 2 too short (length
                          and Gopher), 1 that repeats itself (repetition)
  sources/notes.txt       plain text, CC-BY-4.0: 24 notes separated by blank
                          lines
  protected/stories-val.jsonl
                          8 formats/eval-case.schema.json rows: the held-out
                          stories no training document may overlap
  corpus.toml             the corpus config (formats/corpus-config.schema.json):
                          relative urls and protected paths resolve against
                          this file's directory (MS-corpus header)
  bad-license.toml        the same sources, with notes declared GPL-3.0-only:
                          the run must stop with exit 65 at ledger verify

    uv run --script course/oracle/MS-corpus/make_fixtures.py

Run from the repo root; prints the MANIFEST.tsv rows. The expected counts in
course/milestones/MS-corpus.toml come from a run of the reference CLI
(course/ref/entry/python/corpus/__main__.py) on these files.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

OUT = Path("course/fixtures/MS-corpus")

NAMES = "Tom Mia Ben Lily Sam Zoe Max Ava Leo Ruby Finn Nora Jack Ella Owen Lucy Theo Iris Hugo Maya".split()
ANIMALS = "dog cat rabbit fox owl duck frog bear mouse goat horse turtle squirrel lamb pony".split()
PLACES = "park garden forest river beach farm hill school kitchen meadow village lake barn library pond".split()
THINGS = "ball kite hat boat cake book drum lamp apple box shell map key bell blanket".split()
ADJ = "little big happy sleepy brave quiet funny shiny tiny old kind clever gentle bright".split()
FEEL = "happy sad surprised proud scared excited calm curious tired glad".split()
VERBS = "found lost painted carried shared hid cleaned fixed opened dropped".split()
TEMPLATES = [
    "One day {n} went to the {p} with a {a} {an}.",
    "{n} saw a {a} {t} near the {p}.",
    "The {an} said the {t} was too {a} to carry.",
    "{n} {v} the {t} and felt {f}.",
    "They walked to the {p} to look for a {a} {an}.",
    "When the sun went down, {n} sat by the {p}.",
    "It was a {a} day, so {n} took the {t} outside.",
    "{n} asked the {an} about the {t} in the {p}.",
    "The {a} {an} laughed at the {t}.",
    "At night {n} told {n2} about the {a} {t}.",
    "{n2} smiled and said the {an} was {f}.",
    "{n} learned to be {f} and kind to the {an}.",
    "The {an} {v} a {t} under a tree by the {p}.",
    "After lunch {n} and {n2} played with the {t}.",
    "{n} felt {f} when the {t} was gone.",
    "A {a} {an} came out of the {p}.",
    "{n2} {v} a {a} {t} for {n}.",
    "The {p} was {a} and full of {an}s.",
    "Then the {an} ran to the {p} with the {t}.",
    "{n} and the {an} became {a} friends.",
    "Every morning {n2} visits the {p}.",
    "The {t} rolled into the {p}, and {n} ran after it.",
    "{n} gave the {an} a {a} {t}.",
    "Nobody knew why the {an} was so {f}.",
    "Later, {n2} {v} the {t} again.",
]
FRENCH = [
    "Le petit chat est dans le jardin avec une balle rouge et il joue avec elle pendant que la fille regarde le ciel bleu.",
    "Il était une fois une fille qui avait un chien et elle allait au parc avec lui pour jouer dans le sable et sur la pelouse.",
]


def story(rng: random.Random, lo: int = 100, hi: int = 160) -> str:
    out: list[str] = []
    order = rng.sample(
        TEMPLATES, len(TEMPLATES)
    )  # each template at most once per story
    while sum(len(s.split()) for s in out) < lo:
        t = order[len(out) % len(order)]
        out.append(
            t.format(
                n=rng.choice(NAMES),
                n2=rng.choice(NAMES),
                an=rng.choice(ANIMALS),
                p=rng.choice(PLACES),
                t=rng.choice(THINGS),
                a=rng.choice(ADJ),
                f=rng.choice(FEEL),
                v=rng.choice(VERBS),
            )
        )
    text = " ".join(out)
    words = text.split()
    return " ".join(words[:hi]) if len(words) > hi else text


def main() -> None:
    rng = random.Random(31)
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    (OUT / "protected").mkdir(exist_ok=True)

    val = [story(rng, 60, 90) for _ in range(8)]
    suite = [
        {
            "case_id": f"stories-val-{i}",
            "input": v,
            "ground_truth": "",
            "tags": ["story"],
            "scorer_args": {},
        }
        for i, v in enumerate(val)
    ]
    (OUT / "protected" / "stories-val.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in suite)
    )

    base = [story(rng) for _ in range(140)]
    extra = []
    for i in range(4):  # verbatim copies
        extra.append(base[i * 7])
    for i in range(5):  # one word changed
        w = base[i * 11 + 3].split()
        k = len(w) // 2
        w[k] = "zebra" if w[k] != "zebra" else "camel"
        extra.append(" ".join(w))
    pii = [
        "ana.lopez@example.com",
        "(212) 555-0147",
        "bo.kim+story@example.org",
        "+44 20 7946 0321",
        "dara@example.net",
    ]
    for i, p in enumerate(pii):
        extra.append(story(rng) + f" Write to {p} to hear the next story.")
    for i in range(3):  # a 15-word span of a protected story
        w = val[i].split()
        extra.append(
            story(rng, 50, 70) + " " + " ".join(w[5:20]) + " " + story(rng, 40, 60)
        )
    extra += FRENCH
    extra += ["Tom ran.", "The end of the story is here and it is short."]
    extra.append(" ".join(["the dog and the cat went to the park"] * 12))
    docs = base + extra
    rng.shuffle(docs)
    stories = "".join(json.dumps({"text": t}) + "\n" for t in docs).encode()
    (OUT / "sources" / "stories.jsonl").write_bytes(stories)

    notes = "\n\n".join(story(rng, 60, 90) for _ in range(24)).encode() + b"\n"
    (OUT / "sources" / "notes.txt").write_bytes(notes)

    def config(notes_license: str) -> str:
        return f'''# MS-corpus fixture config (formats/corpus-config.schema.json). Relative
# source urls and protected paths resolve against this file's directory.
dataset = "small"
version = "v1"

[[sources]]
id = "stories"
url = "sources/stories.jsonl"
sha256 = "{hashlib.sha256(stories).hexdigest()}"
license_spdx = "CC0-1.0"
format = "jsonl"

[[sources]]
id = "notes"
url = "sources/notes.txt"
sha256 = "{hashlib.sha256(notes).hexdigest()}"
license_spdx = "{notes_license}"
format = "text"

[filters]
min_chars = 50
langs = ["en"]

[dedup]
jaccard_threshold = 0.8
num_perm = 128
bands = 16
shingle = 5
seed = 0

[decontam]
protected = ["protected/stories-val.jsonl"]
ngram = 13

[shard]
shard_rows = 64
val_permille = 100

[tokenizer]
id = "bytes"
'''

    (OUT / "corpus.toml").write_text(config("CC-BY-4.0"))
    (OUT / "bad-license.toml").write_text(config("GPL-3.0-only"))
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            b = p.read_bytes()
            print(
                f"{p}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/MS-corpus/make_fixtures.py\t-\t-\tApache-2.0"
            )


if __name__ == "__main__":
    main()
