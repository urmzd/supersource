# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["tokenizers==0.22.1", "tiktoken==0.12.0"]
# ///
"""Maintainer generator for the L1.5 and MS-L1 golden ids, and the parity golden of `tokenizer.bpe`.

    uv run --script course/oracle/L1.5/golden.py            (from the repo root)
    uv run --script course/oracle/L1.5/golden.py --check    (regenerate in memory, compare bytes)

Reads the committed tokenizer files (course/fixtures/tok-gpt2/tokenizer.json,
course/fixtures/tok-smollm2/tokenizer.json, both written unchanged by
course/oracle/tok/golden.py from their pinned revisions) and writes:

  course/fixtures/L1.5/golden.jsonl         10,000 strings: {"text", "gpt2", "smollm2"}
  course/fixtures/MS-L1/texts.jsonl         the first 2,000 strings as {"text"}
  course/fixtures/MS-L1/gpt2.ids.json       {"ids": every gpt2 id of texts.jsonl, in order}
  course/fixtures/MS-L1/smollm2.ids.json    the same for SmolLM2
  course/fixtures/tok-gpt2/parity.json      the first 500 strings as {"cases": [{"input": {"text"}, "output": ids}]}

Every id is computed by Hugging Face `tokenizers` (`encode(text,
add_special_tokens=False)`, which matches added tokens literally) and the
GPT-2 ids are cross-checked against tiktoken's `gpt2` encoding with every
special token allowed. The strings come from `strings()` below, a pure
function of the seed: emoji (ZWJ sequences, skin tones, flags, keycaps),
CJK and other scripts, white-space runs of every Unicode kind, contractions
in every case and quote style, Unicode digits, combining marks, control
characters, added-token texts, and over-long words. Nothing here is
imported by a course test.
"""

from __future__ import annotations

import json
import random
import sys
import unicodedata
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

SEED = 20261009
# Which code points are letters or numbers depends on the Unicode version:
# Python 3.11's unicodedata is Unicode 14.0, 3.12's is 15.0 (tl-tok's tables,
# course/oracle/L1.5/unicode_tables.py), 3.14's is 16.0, and so is Hugging
# Face's regex engine. U+138E9, an Egyptian hieroglyph added in 16.0, is a
# letter to Hugging Face and unassigned to Python 3.12, so the two split
# "\U000138e9't" differently. The random code points are therefore drawn
# only among those already assigned in Unicode 14.0 (run with Python 3.11),
# which every one of those versions classifies alike.
UNICODE = "14.0.0"
N = 10_000
MS_N = 2_000  # the MS-L1 subset

WORDS = [
    "the",
    "a",
    "story",
    "once",
    "upon",
    "time",
    "little",
    "girl",
    "named",
    "Lily",
    "she",
    "liked",
    "to",
    "play",
    "with",
    "her",
    "ball",
    "outside",
    "Tom",
    "dog",
    "happy",
    "sad",
    "because",
    "suddenly",
    "Hello",
    "world",
    "THE",
    "Quick",
    "brown",
    "tokenizer",
    "unbelievable",
    "antidisestablishmentarianism",
    "GPU",
    "API",
    "naïve",
    "café",
    "résumé",
    "Zürich",
    "straße",
    "Ελλάδα",
    "москва",
    "Привет",
    "日本語",
    "中文",
    "東京",
    "한국어",
    "ภาษาไทย",
    "हिन्दी",
    "العربية",
    "עברית",
]
CONTRACTIONS = [
    "'s",
    "'t",
    "'re",
    "'ve",
    "'m",
    "'ll",
    "'d",
    "'S",
    "'T",
    "'RE",
    "'Ll",
    "’s",
    "’t",
    "’re",
    "''",
    "'",
    "'x",
    "n't",
    "N'T",
]
SPACES = [
    " ",
    "  ",
    "   ",
    "\t",
    "\t\t",
    "\n",
    "\n\n",
    "\r\n",
    " \n",
    "\n ",
    " ",
    "　",
    " ",
    " ",
    " ",
    "​",
    "\u0085",
    "\x0b",
    "\x0c",
    " \t \n",
    "    ",
    " ",
    " ",
    " ",
]
DIGITS = [
    "0",
    "7",
    "12",
    "123",
    "2024",
    "3.14159",
    "1,000,000",
    "1e-9",
    "0x1F",
    "٣٤٥",
    "۱۲۳",
    "१२३",
    "½",
    "²",
    "Ⅻ",
    "①",
    "𝟙𝟚",
    "１２３",
    "42nd",
    "v1.2.3",
]
EMOJI = [
    "\U0001f600",
    "\U0001f680",
    "\U0001f9e0",
    "\U0001f308",
    "\U0001f44d\U0001f3fd",
    "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466",
    "\U0001f1e8\U0001f1e6",
    "\U0001f1ef\U0001f1f5",
    "1️⃣",
    "❤️",
    "\U0001f3f3️‍\U0001f308",
    "\U0001f9d1\U0001f3ff‍\U0001f4bb",
    "☺",
    "\U0001fae0",
    "\U0001f642",
]
COMBINING = ["é", "ä", "ñ", "क्ष", "ộ", "Z͑ͫ̓ͪ̂ͫ", "각"]
PUNCT = [
    ".",
    ",",
    "!",
    "?",
    "...",
    "?!",
    ";",
    ":",
    "-",
    "--",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    '"',
    "“",
    "”",
    "«",
    "»",
    "—",
    "…",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "/",
    "\\",
    "|",
    "~",
    "`",
    "^",
    "<",
    ">",
    "=",
    "+",
    "_",
]
CONTROL = ["\x00", "\x01", "\x07", "\x1b", "\x1c", "\x7f", "﻿", "�"]
ADDED = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<repo_name>",
    "<|endoftext",
    "<|im_start|>user\n",
    "<|im_end|><|im_start|>",
]
HAND = [
    "",
    " ",
    "  ",
    "\n",
    "a",
    "Hello world",
    "Hello, world!",
    "don't stop: it's what we'll do, they're sure, I've seen, I'm in, he'd go",
    "DON'T SHOUT: IT'S LOUD",
    "she said ’twas fine",
    "'s 't 're 've 'm 'll 'd",
    "  leading",
    "trailing  ",
    "two  spaces   three    four",
    "tab\tseparated",
    "line\nline\n\nline\r\nend",
    " \n \n ",
    "123456789",
    "1 2 3",
    "2024-03-15",
    "日本語のテキスト",
    "中文字词",
    "\U0001f468‍\U0001f469‍\U0001f467",
    "<|endoftext|>",
    "a<|endoftext|>b",
    "<|im_start|>user\nhi<|im_end|>",
    "x" * 300,
    "ab" * 150,
    " " * 64,
    "\n" * 33,
    "é" * 50,
    "\U0001f600" * 20,
]


def piece(rng: random.Random) -> str:
    k = rng.random()
    if k < 0.30:
        return rng.choice(WORDS)
    if k < 0.45:
        return rng.choice(SPACES)
    if k < 0.53:
        return rng.choice(CONTRACTIONS)
    if k < 0.62:
        return rng.choice(DIGITS)
    if k < 0.70:
        return rng.choice(EMOJI)
    if k < 0.75:
        return rng.choice(COMBINING)
    if k < 0.88:
        return rng.choice(PUNCT)
    if k < 0.91:
        return rng.choice(CONTROL)
    if k < 0.93:
        return rng.choice(ADDED)
    if k < 0.96:  # random code points from the BMP and the astral planes
        out = []
        for _ in range(rng.randint(1, 4)):
            while True:  # assigned in Unicode 14.0 (see UNICODE above)
                cp = rng.choice(
                    [
                        rng.randint(0x20, 0x2FF),
                        rng.randint(0x370, 0x1FFF),
                        rng.randint(0x3000, 0x9FFF),
                        rng.randint(0xAC00, 0xD7A3),
                        rng.randint(0x10000, 0x1FAFF),
                    ]
                )
                if unicodedata.category(chr(cp)) != "Cn":
                    break
            out.append(chr(cp))
        return "".join(out)
    return rng.choice(WORDS) * rng.randint(2, 5)  # over-long words


def strings() -> list[str]:
    rng = random.Random(SEED)
    out = list(HAND)
    while len(out) < N:
        out.append("".join(piece(rng) for _ in range(rng.randint(1, 4))))
    return out


def main() -> int:
    check = "--check" in sys.argv[1:]
    if unicodedata.unidata_version != UNICODE:
        raise SystemExit(
            f"run with Python 3.11 (Unicode {UNICODE}), not {unicodedata.unidata_version}"
        )
    root = Path.cwd()
    fx = root / "course" / "fixtures"
    gpt2 = Tokenizer.from_file(str(fx / "tok-gpt2" / "tokenizer.json"))
    smol = Tokenizer.from_file(str(fx / "tok-smollm2" / "tokenizer.json"))
    tk = tiktoken.get_encoding("gpt2")
    rows, all_g, all_s = [], [], []
    for t in strings():
        g = gpt2.encode(t, add_special_tokens=False).ids
        s = smol.encode(t, add_special_tokens=False).ids
        tt = tk.encode(t, allowed_special="all")
        if g != tt:
            raise SystemExit(f"tokenizers and tiktoken disagree on {t!r}: {g} vs {tt}")
        # decode(encode(x)) == x holds for these byte-level files
        assert gpt2.decode(g, skip_special_tokens=False) == t or "�" in t, t
        rows.append({"text": t, "gpt2": g, "smollm2": s})
        if len(rows) <= MS_N:
            all_g += g
            all_s += s
    files = {
        fx / "L1.5" / "golden.jsonl": "".join(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            for r in rows
        ),
        fx / "MS-L1" / "texts.jsonl": "".join(
            json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n"
            for r in rows[:MS_N]
        ),
        fx / "MS-L1" / "gpt2.ids.json": json.dumps(
            {"ids": all_g}, separators=(",", ":")
        )
        + "\n",
        fx / "MS-L1" / "smollm2.ids.json": json.dumps(
            {"ids": all_s}, separators=(",", ":")
        )
        + "\n",
        fx / "tok-gpt2" / "parity.json": json.dumps(
            {
                "cases": [
                    {"input": {"text": r["text"]}, "output": r["gpt2"]}
                    for r in rows[:500]
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n",
    }
    bad = 0
    for p, text in files.items():
        if check:
            same = p.is_file() and p.read_text(encoding="utf-8") == text
            print(f"{'same' if same else 'DIFF'}  {p.relative_to(root)}")
            bad += not same
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            print(f"wrote {p.relative_to(root)} ({len(text.encode())} bytes)")
    print(
        f"{len(rows)} strings; MS-L1: {MS_N} strings, {len(all_g)} gpt2 ids, {len(all_s)} smollm2 ids"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
