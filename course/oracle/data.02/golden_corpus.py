#!/usr/bin/env python3
"""Generate the data.02 golden corpus and the data.03 duplicate corpus.

Maintainer-only, standard library plus the frozen PCG32 (course/tests/_lib).
Every text is synthetic and original (templates below), so the fixtures carry
no third-party license. Labels come from construction, not from the
reference: a "keep" document is built far inside every threshold, a "drop"
document is built to fail one named rule by a wide margin, and the expected
output text of a messy document is the clean text it was made from.

    uv run python course/oracle/data.02/golden_corpus.py [--check]

writes (or with --check, compares against):
    course/fixtures/data.02/golden_in.jsonl    200 docs: id, source_id, text, expect, why
    course/fixtures/data.02/golden_out.jsonl   the kept docs: id, lang, sha256 of the output text
    course/fixtures/data.03/dupes.jsonl        docs with planted exact duplicates
    course/fixtures/data.03/expected.json      counts and kept output, by construction
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from _lib.pcg32 import PCG32  # noqa: E402

NAMES = [
    "Lily",
    "Tom",
    "Mia",
    "Ben",
    "Sam",
    "Max",
    "Ana",
    "Leo",
    "Ivy",
    "Kai",
    "Nora",
    "Finn",
    "Ruby",
    "Omar",
    "Ella",
    "Jack",
]
ACCENTED = ["Zoë", "José", "Chloé", "Renée", "Noël", "Inés"]
ANIMALS = [
    "cat",
    "dog",
    "rabbit",
    "duck",
    "frog",
    "horse",
    "bird",
    "fox",
    "mouse",
    "goat",
    "turtle",
    "puppy",
]
OBJECTS = [
    "ball",
    "kite",
    "box",
    "boat",
    "drum",
    "hat",
    "book",
    "cup",
    "sock",
    "train",
    "shell",
    "lamp",
]
PLACES = [
    "park",
    "garden",
    "forest",
    "kitchen",
    "beach",
    "farm",
    "school",
    "river",
    "hill",
    "market",
]
ADJS = [
    "red",
    "big",
    "small",
    "shiny",
    "soft",
    "happy",
    "quiet",
    "brave",
    "funny",
    "green",
    "tiny",
    "warm",
]
FEELINGS = [
    "happy",
    "sad",
    "proud",
    "sleepy",
    "curious",
    "excited",
    "calm",
    "surprised",
]

SENTENCES = [
    "Once upon a time, {n} had a {a} {o} that {p1} loved very much.",
    "Every morning {n} walked to the {pl} with a little {an}.",
    "The {an} wanted to play with the {a} {o}, but it was too high.",
    "{n} looked under the bed and found an old {o} covered in dust.",
    "One day the sky turned dark and {n} heard a loud noise from the {pl}.",
    "{n} asked {m} for help, and together they carried the {o} home.",
    "The {a} {an} jumped over a puddle and laughed at the rain.",
    "At lunch {n} shared a warm piece of bread with the hungry {an}.",
    "{m} said that sharing is a good thing to do with friends.",
    "After a while the {an} fell asleep next to the {o}.",
    "{n} felt {f} because the day had been long and full of games.",
    "They built a tower of sticks by the {pl} and named it the castle.",
    "When the sun went down, {n} counted the stars one by one.",
    "The wind took the {o} away, so {n} ran after it as fast as possible.",
    "A kind old man gave {n} a map of the {pl} with a big cross on it.",
    "{n} and {m} promised to meet again the next day at the {pl}.",
    "Mom smiled and told {n} to wash before dinner.",
    "Inside the {o} there was a note that said thank you.",
    "The {an} was {f}, so {n} sang a soft song to it.",
    "In the end everyone went home tired and very {f}.",
    "{n} painted the {o} {a} and showed it to the whole class.",
    "Grandpa told a story about a {a} {an} who could fly.",
    "The little {an} hid behind a tree and waited for {n} to find it.",
    "{n} learned that being patient is better than being fast.",
]

FOREIGN = {
    "de": [
        "Der kleine Hund spielt im Garten und die Katze schläft auf dem Sofa.",
        "Am Morgen geht {n} mit der Mutter zum Markt und kauft ein Brot.",
        "Es ist ein schöner Tag, und die Kinder lachen auf der Wiese.",
        "Die Oma erzählt eine Geschichte von einem Fuchs, der sehr klug war.",
        "Nach dem Essen ist der Junge müde und will nicht mehr spielen.",
        "Im Winter liegt viel Schnee, und wir bauen einen großen Mann aus Schnee.",
    ],
    "fr": [
        "Le petit chat dort sur le lit et la fille joue dans le jardin.",
        "Chaque matin, {n} va au marché avec sa mère pour acheter du pain.",
        "Il fait beau aujourd'hui et les enfants chantent dans la cour.",
        "La grand-mère raconte une histoire sur un renard qui était très malin.",
        "Après le repas, le garçon est fatigué et ne veut pas jouer.",
        "En hiver, il y a de la neige et nous faisons un bonhomme dans le parc.",
    ],
    "es": [
        "El perro pequeño juega en el jardín y la gata duerme en la cama.",
        "Cada mañana {n} va al mercado con su madre para comprar pan.",
        "Hoy hace sol y los niños cantan en el patio de la escuela.",
        "La abuela cuenta una historia de un zorro que era muy listo.",
        "Después de la comida, el niño está cansado y no quiere jugar.",
        "En invierno hay mucha nieve y hacemos un muñeco en el parque.",
    ],
    "nl": [
        "De kleine hond speelt in de tuin en de kat slaapt op de bank.",
        "Elke ochtend gaat {n} met haar moeder naar de markt voor brood.",
        "Het is mooi weer en de kinderen zingen op het plein.",
        "Oma vertelt een verhaal over een vos die heel slim was.",
        "Na het eten is de jongen moe en wil hij niet meer spelen.",
        "In de winter ligt er veel sneeuw en maken wij een sneeuwpop.",
    ],
}


class Gen:
    def __init__(self, seed: int) -> None:
        self.r = PCG32(seed=seed)

    def pick(self, xs):
        return xs[self.r.below(len(xs))]

    def sample(self, xs, k):
        pool = list(xs)
        out = []
        for _ in range(k):
            out.append(pool.pop(self.r.below(len(pool))))
        return out

    def fill(self, tmpl: str, names) -> str:
        n, m = names
        return tmpl.format(
            n=n,
            m=m,
            p1="she" if self.r.below(2) else "he",
            a=self.pick(ADJS),
            o=self.pick(OBJECTS),
            an=self.pick(ANIMALS),
            pl=self.pick(PLACES),
            f=self.pick(FEELINGS),
        )

    def story(self, n_sent: int = 9, names=None) -> str:
        names = names or self.sample(NAMES, 2)
        sents = [self.fill(t, names) for t in self.sample(SENTENCES, n_sent)]
        k = min(self.r.below(2) + 2, n_sent)  # 2 or 3 paragraphs
        cuts = sorted(self.sample(range(1, n_sent), k - 1))
        paras, prev = [], 0
        for c in cuts + [n_sent]:
            paras.append(" ".join(sents[prev:c]))
            prev = c
        return "\n\n".join(paras)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> int:
    return len(text.split())


def golden() -> tuple[list[dict], list[dict]]:
    g = Gen(seed=20261009)
    docs: list[dict] = []

    def add(text: str, expect: str, why: str, lang: str = "en", out: str | None = None):
        k = len(docs)
        docs.append(
            {
                "id": f"golden:{k}",
                "source_id": "golden",
                "text": text,
                "expect": expect,
                "why": why,
                "_out": out,
                "_lang": lang,
            }
        )

    plan = (
        ["good"] * 110
        + ["messy"] * 15
        + ["short_words"] * 12
        + ["short_chars"] * 5
        + ["empty"] * 5
        + ["foreign"] * 15
        + ["bullets"] * 8
        + ["hashtags"] * 8
        + ["ellipsis"] * 7
        + ["alpha"] * 7
        + ["repetition"] * 8
    )
    order = g.sample(plan, len(plan))  # shuffled, so kinds interleave
    langs = sorted(FOREIGN)
    for i, kind in enumerate(order):
        if kind == "good":
            t = g.story(g.r.below(4) + 8)
            add(t, "keep", "good story", out=t)
        elif kind == "messy":
            names = [g.pick(ACCENTED), g.pick(NAMES)]
            clean = g.story(9, names)
            assert unicodedata.is_normalized("NFC", clean)
            messy = unicodedata.normalize("NFD", clean)
            lines = messy.split("\n")
            lines = [x + (" " * (1 + g.r.below(3)) if x else "\t ") for x in lines]
            messy = "\r\n".join(lines)
            messy = messy.replace(
                "\r\n\t \r\n", "\r\n\r\n\r\n \r\n"
            )  # extra blank lines between paragraphs
            messy = "\ufeff" + messy.replace(". ", ".\x07 ", 1) + "\x00\r\n\r\n"
            add(
                messy,
                "keep",
                "noisy bytes around a good story: CRLF, NFD, BOM, controls, trailing blanks",
                out=clean,
            )
        elif kind == "short_words":
            t = g.story(3)
            assert words(t) < 45
            add(t, "drop", "gopher:words (fewer than 50 words)")
        elif kind == "short_chars":
            add(
                g.pick(["The end.", "Hi there.", "Yes.", "So it goes.", "Bye now."]),
                "drop",
                "length (fewer than 20 characters)",
            )
        elif kind == "empty":
            add(
                g.pick(["", "   ", "\n\n\t\n", "\ufeff", "\r\n \r\n"]),
                "drop",
                "normalize (empty after normalization)",
            )
        elif kind == "foreign":
            lang = langs[i % len(langs)]
            name = g.pick(NAMES)
            sents = [s.format(n=name) for s in g.sample(FOREIGN[lang], 6)]
            add(
                " ".join(sents[:3]) + "\n\n" + " ".join(sents[3:]),
                "drop",
                f"lang (written in {lang})",
                lang=lang,
            )
        elif kind == "bullets":
            items = [g.fill(t, g.sample(NAMES, 2)) for t in g.sample(SENTENCES, 8)]
            add(
                "\n".join("- " + x for x in items),
                "drop",
                "gopher:bullet_lines (every line is a bullet)",
            )
        elif kind == "hashtags":
            t = g.story(9)
            ws = t.split(" ")
            for j in range(0, len(ws), 4):
                ws[j] = ws[j] + " #" + g.pick(ANIMALS + OBJECTS)
            add(" ".join(ws), "drop", "gopher:symbol_ratio (one # per four words)")
        elif kind == "ellipsis":
            sents = [g.fill(t, g.sample(NAMES, 2)) for t in g.sample(SENTENCES, 9)]
            lines = [s[:-1] + "..." if j % 2 == 0 else s for j, s in enumerate(sents)]
            add(
                "\n".join(lines),
                "drop",
                "gopher:ellipsis_lines (half the lines trail off)",
            )
        elif kind == "alpha":
            t = g.story(9)
            ws = t.split(" ")
            nums = [str(1000 + g.r.below(9000)) for _ in range(len(ws) // 2)]
            mixed = [x for pair in zip(ws, nums) for x in pair] + ws[len(nums) :]
            add(
                " ".join(mixed),
                "drop",
                "gopher:alpha_words (a third of the words are numbers)",
            )
        elif kind == "repetition":
            s = g.fill(g.pick(SENTENCES), g.sample(NAMES, 2))
            t = g.story(5)
            add(
                t + "\n\n" + " ".join([s] * 6),
                "drop",
                "repetition (one sentence six times)",
            )
    out = [
        {"id": d["id"], "lang": "en", "sha256": sha(d["_out"])}
        for d in docs
        if d["expect"] == "keep"
    ]
    for d in docs:
        del d["_out"], d["_lang"]
    return docs, out


def dupes() -> tuple[list[dict], dict]:
    """Docs of 2 to 4 unique paragraphs, then planted duplicates:
    whole-document copies and documents that reuse earlier paragraphs.
    Exact dedup keeps the first occurrence of every paragraph."""
    g = Gen(seed=31337)
    seen: set[str] = set()
    pool: list[str] = []  # every unique paragraph, in order of first use
    docs: list[dict] = []

    def fresh_para() -> str:
        while True:
            p = " ".join(g.fill(t, g.sample(NAMES, 2)) for t in g.sample(SENTENCES, 3))
            if p not in seen:
                seen.add(p)
                pool.append(p)
                return p

    def add(paras: list[str], why: str) -> None:
        docs.append(
            {
                "id": f"dupes:{len(docs)}",
                "source_id": "dupes",
                "text": "\n\n".join(paras),
                "why": why,
            }
        )

    for i in range(200):
        r = g.r.below(10)
        if i >= 20 and r == 0:
            # an exact copy of an earlier document
            src = docs[g.r.below(len(docs))]
            docs.append(
                {
                    "id": f"dupes:{len(docs)}",
                    "source_id": "dupes",
                    "text": src["text"],
                    "why": f"copy of {src['id']}",
                }
            )
        elif i >= 20 and r in (1, 2):
            # a new paragraph plus one or two reused ones
            reuse = g.sample(pool, 1 + g.r.below(2))
            paras = [fresh_para()] + reuse
            paras = g.sample(paras, len(paras))
            add(paras, "reuses earlier paragraphs")
        else:
            add([fresh_para() for _ in range(2 + g.r.below(3))], "unique")
    # Expected output by construction: walk in order, keep a paragraph the
    # first time its exact text appears, drop a document left empty.
    kept_seen: set[str] = set()
    kept, n_paras, dup_paras, dropped_docs = [], 0, 0, 0
    for d in docs:
        paras = [p.strip() for p in d["text"].split("\n\n") if p.strip()]
        n_paras += len(paras)
        keep = []
        for p in paras:
            if p in kept_seen:
                dup_paras += 1
            else:
                kept_seen.add(p)
                keep.append(p)
        if keep:
            kept.append({"id": d["id"], "sha256": sha("\n\n".join(keep))})
        else:
            dropped_docs += 1
    expected = {
        "docs": len(docs),
        "paragraphs": n_paras,
        "unique_paragraphs": len(kept_seen),
        "duplicate_paragraphs": dup_paras,
        "dropped_docs": dropped_docs,
        "kept": kept,
    }
    return docs, expected


def dump_jsonl(rows: list[dict]) -> str:
    return "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
    )


def main() -> int:
    check = "--check" in sys.argv
    gin, gout = golden()
    dd, exp = dupes()
    files = {
        ROOT / "fixtures" / "data.02" / "golden_in.jsonl": dump_jsonl(gin),
        ROOT / "fixtures" / "data.02" / "golden_out.jsonl": dump_jsonl(gout),
        ROOT / "fixtures" / "data.03" / "dupes.jsonl": dump_jsonl(dd),
        ROOT / "fixtures" / "data.03" / "expected.json": json.dumps(
            exp, ensure_ascii=False, indent=1
        )
        + "\n",
    }
    bad = 0
    for p, text in files.items():
        if check:
            if not p.is_file() or p.read_text(encoding="utf-8") != text:
                print(f"DIFFERS {p.relative_to(ROOT.parent)}")
                bad += 1
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            print(f"wrote {p.relative_to(ROOT.parent)} ({len(text.encode())} bytes)")
    print(
        f"golden: {len(gin)} in, {len(gout)} kept; dupes: {exp['docs']} docs, "
        f"{exp['paragraphs']} paragraphs, {exp['duplicate_paragraphs']} duplicate, {exp['dropped_docs']} docs dropped"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
