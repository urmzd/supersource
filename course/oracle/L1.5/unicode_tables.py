"""Maintainer generator for the general-category tables in tl-tok's pre-tokenizer (L1.5).

    uv run --project course/harness python course/oracle/L1.5/unicode_tables.py

Prints two Rust `const` tables of inclusive code-point ranges, LETTER_RANGES
(categories Lu Ll Lt Lm Lo, what the GPT-2 regex calls \\p{L}) and
NUMBER_RANGES (Nd Nl No, \\p{N}), computed from Python's `unicodedata`. The
Python BPE (L1.2) classifies characters with the same `unicodedata`, so a
Rust pre-tokenizer reading these tables splits every string exactly as the
Python one does, including code points newer than Rust's own Unicode tables.
The chapter (section 2) shows this script; learners run their own copy.
"""

from __future__ import annotations

import sys
import unicodedata

CATS = {
    "LETTER_RANGES": ("Lu", "Ll", "Lt", "Lm", "Lo"),
    "NUMBER_RANGES": ("Nd", "Nl", "No"),
}


def ranges(cats: tuple[str, ...]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue  # surrogates are not chars in Rust (or in UTF-8 text)
        if unicodedata.category(chr(cp)) in cats:
            if out and out[-1][1] == cp - 1:
                out[-1] = (out[-1][0], cp)
            else:
                out.append((cp, cp))
    return out


def rust(name: str, rs: list[tuple[int, int]]) -> str:
    cells = [f"(0x{a:05X}, 0x{b:05X})," for a, b in rs]
    lines = [" ".join(cells[i : i + 4]) for i in range(0, len(cells), 4)]
    body = "\n".join("    " + ln for ln in lines)
    return f"pub const {name}: &[(u32, u32)] = &[\n{body}\n];\n"


def main() -> int:
    print(f"// Unicode {unicodedata.unidata_version}, from Python {sys.version.split()[0]} unicodedata")
    for name, cats in CATS.items():
        rs = ranges(cats)
        print(f"// {name}: {len(rs)} ranges, categories {' '.join(cats)}")
        print(rust(name, rs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
