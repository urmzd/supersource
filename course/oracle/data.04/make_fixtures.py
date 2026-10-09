# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the data.04 fixtures (near dedup, decontamination).

Everything is synthetic: pseudo-words built from syllables, so the files
carry no third-party text (license Apache-2.0, like the repository). The
labels come from how each document was built, never from a detector.

course/fixtures/data.04/near-dups.jsonl
    {"id", "text", "group"}: 60 base documents of 160 to 220 words. Groups
    of 2 to 4 near duplicates are made from a base by one word
    substitution, or by dropping the last 3 to 8 words (word 5-shingle
    Jaccard >= 0.93 with the base, five standard deviations of a 128-entry
    estimate above 0.8, so no seed splits a planted cluster). "group" names the planted cluster;
    a document alone in its group has no near duplicate. 12 "borderline"
    documents share a long run with a base but replace a quarter of it,
    so their Jaccard with it is 0.45 to 0.6: they must stay apart at a 0.8
    threshold. Their group is their own id.

course/fixtures/data.04/protected/val-suite.jsonl
    formats/eval-case.schema.json rows: {"case_id", "input", "ground_truth",
    "tags", "scorer_args"}; inputs of 20 to 30 words.
course/fixtures/data.04/protected/heldout.txt
    one plain-text protected document of 60 words.
course/fixtures/data.04/decontam-docs.jsonl
    {"id", "text", "contaminated", "why"}: documents that embed a 13-word
    span of a protected text (at the start, middle, or end; upper-cased or
    with punctuation changed, which words() ignores) are contaminated;
    documents with a 12-word span, or with the case ids and tags of the
    suite, are not.

    uv run --script course/oracle/data.04/make_fixtures.py

Run from the repo root; prints the MANIFEST.tsv rows.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

OUT = Path("course/fixtures/data.04")
SYL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


def vocab(rng: random.Random, n: int) -> list[str]:
    seen: set[str] = set()
    out = []
    while len(out) < n:
        w = "".join(rng.choice(SYL) for _ in range(rng.randint(2, 4)))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def shingles(text: str, k: int = 5) -> set[str]:
    w = words(text)
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}


def jaccard(a: str, b: str) -> float:
    x, y = shingles(a), shingles(b)
    return len(x & y) / len(x | y)


def sentence_case(ws: list[str]) -> str:
    out, i = [], 0
    while i < len(ws):
        n = min(len(ws) - i, 8 + (i % 7))
        s = " ".join(ws[i : i + n])
        out.append(s[0].upper() + s[1:] + ".")
        i += n
    return " ".join(out)


def main() -> None:
    rng = random.Random(20261009)
    V = vocab(rng, 4000)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protected").mkdir(exist_ok=True)

    # -- near duplicates ---------------------------------------------------
    rows = []
    for g in range(60):
        base = [rng.choice(V) for _ in range(rng.randint(160, 220))]
        text = sentence_case(base)
        rows.append({"id": f"nd:{g:03d}:0", "text": text, "group": f"g{g:03d}"})
        n_var = rng.choice([0, 0, 1, 1, 2, 3])
        for v in range(1, n_var + 1):
            while True:  # redraw the rare edit that lands below 0.85
                ws = list(base)
                if rng.random() < 0.6:
                    ws[rng.randrange(len(ws))] = rng.choice(V)
                else:
                    ws = ws[: len(ws) - rng.randint(3, 8)]
                t = sentence_case(ws)
                if jaccard(text, t) >= 0.93:
                    break
            rows.append({"id": f"nd:{g:03d}:{v}", "text": t, "group": f"g{g:03d}"})
        if g < 12:
            ws = list(base)
            lo = rng.randrange(len(ws) // 4)
            for i in range(lo, lo + len(ws) // 4):
                ws[i] = rng.choice(V)
            t = sentence_case(ws)
            assert 0.45 <= jaccard(text, t) <= 0.6, jaccard(text, t)
            rid = f"nd:{g:03d}:b"
            rows.append({"id": rid, "text": t, "group": rid})
    rng.shuffle(rows)
    write_jsonl(OUT / "near-dups.jsonl", rows)

    # -- protected sets ----------------------------------------------------
    suite = []
    for i in range(20):
        inp = sentence_case([rng.choice(V) for _ in range(rng.randint(20, 30))])
        gt = sentence_case([rng.choice(V) for _ in range(rng.randint(14, 20))])
        suite.append(
            {
                "case_id": f"val-{i:02d}",
                "input": inp,
                "ground_truth": gt,
                "tags": ["story", "heldout"],
                "scorer_args": {"metric": "exact"},
            }
        )
    write_jsonl(OUT / "protected" / "val-suite.jsonl", suite)
    heldout = sentence_case([rng.choice(V) for _ in range(60)])
    (OUT / "protected" / "heldout.txt").write_text(heldout + "\n")

    def filler(n: int) -> list[str]:
        return [rng.choice(V) for _ in range(n)]

    docs = []
    sources = (
        [c["input"] for c in suite] + [c["ground_truth"] for c in suite] + [heldout]
    )
    for i in range(36):
        src = words(sources[i % len(sources)])
        at = rng.randrange(len(src) - 13 + 1)
        span = src[at : at + 13]
        where = ["start", "middle", "end"][i % 3]
        style = ["plain", "upper", "punct"][(i // 3) % 3]
        if style == "upper":
            span = [w.upper() for w in span]
        elif style == "punct":
            span = [w + ("," if j % 3 == 2 else "") for j, w in enumerate(span)]
        before = [] if where == "start" else filler(rng.randint(20, 40))
        after = [] if where == "end" else filler(rng.randint(20, 40))
        text = " ".join(before + span + after) + "."
        docs.append(
            {
                "id": f"dc:{i:03d}",
                "text": text,
                "contaminated": True,
                "why": f"13-word span at the {where}, {style}",
            }
        )
    for i in range(36, 60):
        src = words(sources[i % len(sources)])
        at = rng.randrange(len(src) - 12 + 1)
        span = src[at : at + 12]
        text = " ".join(filler(25) + span + filler(25)) + "."
        docs.append(
            {
                "id": f"dc:{i:03d}",
                "text": text,
                "contaminated": False,
                "why": "12-word span only",
            }
        )
    for i in range(60, 66):
        c = suite[i - 60]
        text = (
            " ".join(filler(20))
            + f" {c['case_id']} "
            + " ".join(c["tags"] * 7)
            + " "
            + json.dumps(c["scorer_args"])
            + "."
        )
        docs.append(
            {
                "id": f"dc:{i:03d}",
                "text": text,
                "contaminated": False,
                "why": "case ids, tags, scorer_args are not protected text",
            }
        )
    for i in range(66, 100):
        docs.append(
            {
                "id": f"dc:{i:03d}",
                "text": sentence_case(filler(rng.randint(40, 90))),
                "contaminated": False,
                "why": "unrelated",
            }
        )
    rng.shuffle(docs)
    write_jsonl(OUT / "decontam-docs.jsonl", docs)

    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            b = p.read_bytes()
            print(
                f"{p}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/data.04/make_fixtures.py\t-\t-\tApache-2.0"
            )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    main()
