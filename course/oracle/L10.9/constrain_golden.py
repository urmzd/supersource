"""Maintainer generator for course/fixtures/L10.9/constrain_golden.json.

L10.9 (the Rust port of L8.7's constrained decoding) is held to the Python
specification on shared inputs. This script runs the reference Python
(course/ref/python, tinyllm.infer.constrain) and records, per case:

* `schema` cases: the JSON-schema subset document (as JSON text, key order
  kept), the regex json_schema_to_regex writes for it, and its canonical DFA;
* `regex` cases: a pattern and its canonical DFA;
* for both: three seeded paths through a fixed byte-token vocabulary with
  the allowed token ids at every step (TokenIndex.mask), whether the prefix
  is complete, and the token taken (EOS once complete, as L8.7's golden);
* `dumps`: JSON texts and Python's compact json.dumps(..., separators=(",",
  ":"), ensure_ascii=False) of them, which enum and const values go through;
* `errors`: schemas and patterns the subset rejects.

A DFA is stored as its accepting flags and, per state, runs [lo, hi, target]
of equal transitions over bytes 0..255 (target -1 = DEAD).

    uv run --project course/harness python course/oracle/L10.9/constrain_golden.py
Run from the repo root; it prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "course" / "ref" / "python"))

from tinyllm.infer.constrain import (  # noqa: E402
    DEAD,
    TokenIndex,
    json_schema_to_regex,
    regex_to_dfa,
)

OUT = ROOT / "course" / "fixtures" / "L10.9" / "constrain_golden.json"

# The vocabulary every path walks: printable ASCII bytes, a few multi-byte
# tokens a JSON tokenizer would have, two UTF-8 tokens, a special (None), an
# empty token, and EOS last.
TOKENS: list[bytes | None] = [bytes([b]) for b in range(0x20, 0x7F)] + [
    b'{"',
    b'":',
    b'","',
    b'"}',
    b'"',
    b"true",
    b"false",
    b"null",
    b"Paris",
    b"city",
    b"name",
    b"arguments",
    b"<tool_call>",
    b"</tool_call>",
    b"\n",
    b"\xc3\xa9",
    b"\xe2\x82\xac",
    b"12",
    b"-0",
    b"1.5",
    b"e+",
    b"\\n",
    b"\\u00e9",
    b"\xc3",
    b"\xa9",
    None,
    b"",
]
EOS = len(TOKENS)
VOCAB = TOKENS + [None]

SCHEMAS = [
    (
        "weather",
        {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    ),
    (
        "optional_props",
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "boolean"},
                "c": {"type": "null"},
            },
            "required": ["b"],
        },
    ),
    (
        "all_optional",
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        },
    ),
    (
        "unsorted",
        {
            "type": "object",
            "properties": {
                "zeta": {"type": "integer"},
                "alpha": {"type": "boolean"},
                "mid": {"type": "string"},
            },
            "required": ["zeta", "alpha"],
        },
    ),
    (
        "enum_mixed",
        {
            "enum": ["red", 1, 2.5, None, True, {"k": [1, "é"]}, 'a"b\\c\n'],
            "description": "ignored",
        },
    ),
    ("const", {"const": "Paris", "title": "ignored"}),
    (
        "array_bounds",
        {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
    ),
    ("array_free", {"type": "array", "items": {"type": "boolean"}}),
    ("array_empty", {"type": "array", "items": {"type": "null"}, "maxItems": 0}),
    ("any_of", {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]}),
    ("number", {"type": "number"}),
    ("string", {"type": "string"}),
    (
        "nested",
        {
            "type": "object",
            "properties": {
                "loc": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number"},
                        "lon": {"type": "number"},
                    },
                    "required": ["lat", "lon"],
                },
                "unit": {"enum": ["c", "f"]},
            },
            "required": ["loc"],
        },
    ),
]

REGEXES = [
    ("hand_example", "(cat|car|dog)s?"),
    ("classes", r"[a-c_]\d{2,3}\w?"),
    ("negated", r"[^\n\"]+\."),
    ("escapes", r"\x41\t\(\)\[\]\{\}\|\*\+\?\.\\\/\"\-\,\:"),
    ("repeat", r"(?:ab|c){0,2}d{3,}"),
    ("dot_and_alt", r".|xy*|(z)+"),
    ("empty_alt", r"a(b|)c"),
    ("bracket_specials", r"[]a-][^]]"),
]

BAD_SCHEMAS = [
    {"type": "string", "pattern": "a+"},
    {"type": "string", "minLength": 2},
    {"$ref": "#/x"},
    {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["b"]},
    {"type": "object", "additionalProperties": True},
    {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 2},
    {"enum": []},
    {"anyOf": []},
    {"type": "date"},
    {},
    ["not", "a", "dict"],
]

BAD_REGEXES = [
    "^a",
    "a$",
    "a*?",
    "(?=a)",
    r"\1",
    "é",
    "[a",
    "(a",
    "a)",
    "a{3,2}",
    "a{257}",
    r"[^\x00-\xff]",
    r"\q",
    "[z-a]",
    "*a",
]

DUMPS = [
    '"plain"',
    '"quote\\" back\\\\ nl\\n tab\\t ctl\\u0001 bs\\b ff\\f cr\\r"',
    '"\\u00e9\\u20ac \\ud83d\\ude00"',
    "1",
    "-0",
    "1.0",
    "2.5",
    "1e16",
    "1.5e-05",
    "0.0001",
    "123456789012345678",
    "-1.25e+100",
    "100000.0",
    "3.141592653589793",
    "true",
    "false",
    "null",
    '[1,"a",[],{}]',
    '{"z":1,"a":{"m":[null,false]},"é":"\\u00e9"}',
]


def runs(row) -> list[list[int]]:
    out, lo = [], 0
    for b in range(1, 257):
        if b == 256 or int(row[b]) != int(row[lo]):
            out.append([lo, b - 1, int(row[lo])])
            lo = b
    return out


def dfa_doc(dfa) -> dict:
    return {
        "n_states": int(dfa.n_states),
        "accept": [bool(x) for x in dfa.accept],
        "trans": [runs(dfa.trans[s]) for s in range(dfa.n_states)],
    }


def paths(dfa, rng: random.Random) -> list[list[dict]]:
    idx = TokenIndex(dfa, VOCAB, eos_id=EOS)
    out = []
    for _ in range(3):
        state, done, path = 0, False, []
        for _ in range(14):
            mask = idx.mask(state)
            allowed = [int(i) for i in mask.nonzero()[0]]
            complete = bool(dfa.accept[state])
            if not allowed:
                break
            tok = EOS if (complete and rng.random() < 0.3) else rng.choice(allowed)
            path.append({"mask": allowed, "complete": complete, "token": tok})
            if tok == EOS:
                done = True
                break
            state = idx.next_state(state, tok)
        if path:
            out.append(path)
        del done
    return out


def main() -> None:
    rng = random.Random(20261009)
    doc = {
        "generator": "course/oracle/L10.9/constrain_golden.py (reference Python L8.7)",
        "vocab": [None if t is None else t.hex() for t in VOCAB],
        "eos_id": EOS,
        "schemas": [],
        "regexes": [],
        "dumps": [],
        "bad_schemas": [json.dumps(s, ensure_ascii=False) for s in BAD_SCHEMAS],
        "bad_regexes": BAD_REGEXES,
    }
    for name, s in SCHEMAS:
        pat = json_schema_to_regex(s)
        dfa = regex_to_dfa(pat)
        doc["schemas"].append(
            {
                "name": name,
                "schema": json.dumps(s, ensure_ascii=False),
                "regex": pat,
                "dfa": dfa_doc(dfa),
                "paths": paths(dfa, rng),
            }
        )
    for name, pat in REGEXES:
        dfa = regex_to_dfa(pat)
        doc["regexes"].append(
            {
                "name": name,
                "pattern": pat,
                "dfa": dfa_doc(dfa),
                "paths": paths(dfa, rng),
            }
        )
    for text in DUMPS:
        doc["dumps"].append(
            {
                "text": text,
                "compact": json.dumps(
                    json.loads(text), separators=(",", ":"), ensure_ascii=False
                ),
            }
        )
    for s in BAD_SCHEMAS:
        try:
            json_schema_to_regex(s)  # type: ignore[arg-type]
            raise SystemExit(f"schema accepted: {s}")
        except ValueError:
            pass
    for p in BAD_REGEXES:
        try:
            regex_to_dfa(p)
            raise SystemExit(f"pattern accepted: {p}")
        except ValueError:
            pass
    assert DEAD == -1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    OUT.write_bytes(data)
    rel = OUT.relative_to(ROOT).as_posix()
    print(
        f"{rel}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L10.9/constrain_golden.py\t-\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
