# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for course/fixtures/L8.7/masks_golden.json.

An independent, stdlib-only oracle for constrained-decoding masks. It shares
no code and no algorithm with the reference (no NFA, no DFA): patterns are
parsed by Python's own regex parser (re._parser), JSON schemas are turned
into the same kind of tree by a transcription of the subset in
contracts/py/tinyllm/infer/constrain.pyi, and a token is allowed after a
prefix w when the Brzozowski derivative of the tree by w + token still has a
non-empty language. EOS is allowed when the derivative by w is nullable (w is
a whole string of the language).

For each case it walks three seeded paths of up to 14 tokens through a small
vocabulary and records, at each step, the allowed token ids, whether the
prefix is complete, and the token taken. L8.7's course test replays the
paths through your TokenIndex; L10.9 (the Rust port) is held to the same
file.

    uv run --script course/oracle/L8.7/masks_golden.py
Run from the repo root; it prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import re._constants as C
import re._parser as P

ALL = frozenset(range(256))
DIGIT = frozenset(range(0x30, 0x3A))
WORD = DIGIT | frozenset(range(0x41, 0x5B)) | frozenset(range(0x61, 0x7B)) | {0x5F}
SPACE = frozenset(b" \t\n\r\f\v")
CATS = {
    C.CATEGORY_DIGIT: DIGIT,
    C.CATEGORY_NOT_DIGIT: ALL - DIGIT,
    C.CATEGORY_WORD: WORD,
    C.CATEGORY_NOT_WORD: ALL - WORD,
    C.CATEGORY_SPACE: SPACE,
    C.CATEGORY_NOT_SPACE: ALL - SPACE,
}
EMPTY = ("empty",)
EPS = ("eps",)


# -- trees from Python's regex parser ------------------------------------------------


def from_sre(seq) -> tuple:
    items = [node(op, av) for op, av in seq]
    return cat(items)


def node(op, av) -> tuple:
    if op is C.LITERAL:
        return ("set", frozenset([av]))
    if op is C.NOT_LITERAL:
        return ("set", ALL - {av})
    if op is C.ANY:
        return ("set", ALL - {0x0A})
    if op is C.IN:
        s, neg = set(), False
        for o, a in av:
            if o is C.NEGATE:
                neg = True
            elif o is C.LITERAL:
                s.add(a)
            elif o is C.RANGE:
                s |= set(range(a[0], a[1] + 1))
            elif o is C.CATEGORY:
                s |= CATS[a]
            else:
                raise ValueError(f"unsupported class item {o}")
        return ("set", frozenset(ALL - s if neg else s))
    if op is C.BRANCH:
        return alt([from_sre(b) for b in av[1]])
    if op is C.SUBPATTERN:
        return from_sre(av[3])
    if op is C.MAX_REPEAT:
        lo, hi, sub = av
        return rep(from_sre(sub), lo, None if hi is C.MAXREPEAT else hi)
    raise ValueError(f"unsupported regex op {op}")


# -- smart constructors and derivatives ------------------------------------------------


def cat(items) -> tuple:
    out = []
    for x in items:
        if x == EMPTY:
            return EMPTY
        if x == EPS:
            continue
        out += list(x[1]) if x[0] == "cat" else [x]
    return EPS if not out else out[0] if len(out) == 1 else ("cat", tuple(out))


def alt(items) -> tuple:
    out, seen = [], set()
    for x in items:
        for y in x[1] if x[0] == "alt" else [x]:
            if y != EMPTY and y not in seen:
                seen.add(y)
                out.append(y)
    return EMPTY if not out else out[0] if len(out) == 1 else ("alt", tuple(sorted(out, key=repr)))


def rep(x, lo, hi) -> tuple:
    if hi == 0 or x == EPS:
        return EPS
    if x == EMPTY:
        return EPS if lo == 0 else EMPTY
    return ("rep", x, lo, hi)


def nullable(r) -> bool:
    k = r[0]
    if k == "eps":
        return True
    if k in ("empty", "set"):
        return False
    if k == "cat":
        return all(nullable(x) for x in r[1])
    if k == "alt":
        return any(nullable(x) for x in r[1])
    return r[2] == 0 or nullable(r[1])


def is_empty(r) -> bool:
    k = r[0]
    if k == "empty":
        return True
    if k == "eps":
        return False
    if k == "set":
        return not r[1]
    if k == "cat":
        return any(is_empty(x) for x in r[1])
    if k == "alt":
        return all(is_empty(x) for x in r[1])
    return r[2] > 0 and is_empty(r[1])


def deriv(r, b: int) -> tuple:
    k = r[0]
    if k in ("empty", "eps"):
        return EMPTY
    if k == "set":
        return EPS if b in r[1] else EMPTY
    if k == "alt":
        return alt([deriv(x, b) for x in r[1]])
    if k == "cat":
        head, rest = r[1][0], cat(r[1][1:])
        d = cat([deriv(head, b), rest])
        return alt([d, deriv(rest, b)]) if nullable(head) else d
    x, lo, hi = r[1], r[2], r[3]
    return cat([deriv(x, b), rep(x, max(lo - 1, 0), None if hi is None else hi - 1)])


def deriv_bytes(r, data: bytes) -> tuple:
    for b in data:
        r = deriv(r, b)
        if r == EMPTY:
            break
    return r


# -- the JSON-schema subset, transcribed from constrain.pyi ------------------------------


def lit(text: str) -> tuple:
    return cat([("set", frozenset([b])) for b in text.encode("utf-8")])


HEX = frozenset(b"0123456789abcdefABCDEF")


def R(lo: int, hi: int) -> tuple:
    return ("set", frozenset(range(lo, hi + 1)))


def lit_b(b: int) -> tuple:
    return ("set", frozenset([b]))


CONT = R(0x80, 0xBF)  # a UTF-8 continuation byte
STRING = cat(
    [
        lit('"'),
        rep(
            alt(
                [
                    ("set", frozenset(range(0x20, 0x80)) - frozenset(b'"\\')),
                    cat([R(0xC2, 0xDF), CONT]),
                    cat([lit_b(0xE0), R(0xA0, 0xBF), CONT]),
                    cat([("set", frozenset(range(0xE1, 0xED)) | {0xEE, 0xEF}), CONT, CONT]),
                    cat([lit_b(0xED), R(0x80, 0x9F), CONT]),
                    cat([lit_b(0xF0), R(0x90, 0xBF), CONT, CONT]),
                    cat([R(0xF1, 0xF3), CONT, CONT, CONT]),
                    cat([lit_b(0xF4), R(0x80, 0x8F), CONT, CONT]),
                    cat([lit("\\"), ("set", frozenset(b'"\\/bfnrt'))]),
                    cat([lit("\\u")] + [("set", HEX)] * 4),
                ]
            ),
            0,
            None,
        ),
        lit('"'),
    ]
)
NONZERO = ("set", frozenset(b"123456789"))
DIG = ("set", DIGIT)
INTEGER = cat([rep(lit("-"), 0, 1), alt([lit("0"), cat([NONZERO, rep(DIG, 0, None)])])])
NUMBER = cat(
    [
        INTEGER,
        rep(cat([lit("."), rep(DIG, 1, None)]), 0, 1),
        rep(cat([("set", frozenset(b"eE")), rep(("set", frozenset(b"+-")), 0, 1), rep(DIG, 1, None)]), 0, 1),
    ]
)


def schema_tree(s: dict) -> tuple:
    if "enum" in s or "const" in s:
        vals = s["enum"] if "enum" in s else [s["const"]]
        return alt([lit(json.dumps(v, separators=(",", ":"), ensure_ascii=False)) for v in vals])
    if "anyOf" in s:
        return alt([schema_tree(x) for x in s["anyOf"]])
    t = s["type"]
    if t == "string":
        return STRING
    if t == "integer":
        return INTEGER
    if t == "number":
        return NUMBER
    if t == "boolean":
        return alt([lit("true"), lit("false")])
    if t == "null":
        return lit("null")
    if t == "array":
        item = schema_tree(s.get("items", {}))
        lo, hi = s.get("minItems", 0), s.get("maxItems")
        if hi == 0:
            return lit("[]")
        body = cat([item, rep(cat([lit(","), item]), max(lo - 1, 0), None if hi is None else hi - 1)])
        return cat([lit("["), body if lo >= 1 else rep(body, 0, 1), lit("]")])
    if t == "object":
        req = set(s.get("required", []))
        # Every subset of the properties that holds the required ones, in
        # the given order, comma-separated: a tree per subset, all OR-ed.
        names = list(s.get("properties", {}))
        members = [cat([lit(json.dumps(k, ensure_ascii=False)), lit(":"), schema_tree(s["properties"][k])]) for k in names]
        alts = []
        for mask in range(1 << len(names)):
            chosen = [i for i in range(len(names)) if mask >> i & 1]
            if not req <= {names[i] for i in chosen}:
                continue
            parts = [lit("{")]
            for j, i in enumerate(chosen):
                if j:
                    parts.append(lit(","))
                parts.append(members[i])
            parts.append(lit("}"))
            alts.append(cat(parts))
        return alt(alts)
    raise ValueError(t)


# -- cases ------------------------------------------------------------------------------

LETTERS = [bytes([c]) for c in b"abcdgorstx"]
SMALL_VOCAB = [b"c", b"a", b"t", b"r", b"d", b"o", b"g", b"s", b"ca", b"cat", b"dog", b"ts", b"x", None, b""]
JSON_VOCAB = (
    [bytes([c]) for c in b'{}[]",:0123456789-.eE+abcdefilmnrstu\\/ ']
    + [b'{"', b'":', b'",', b'"}', b"true", b"false", b"null", b'"name"', b'"age"', b'"tags"', b"\xc3\xa9", b"\xc3", b"\xa9", b"12", b'\\"']
    + [None, b""]
)

CASES = [
    {"name": "hand_example", "pattern": r"(cat|car|dog)s?", "vocab": SMALL_VOCAB, "eos_id": 13},
    {"name": "classes_and_repeats", "pattern": r"[a-c]{2,3}(?:x|\d)*[^a-z]", "vocab": LETTERS + [b"0", b"5", b"!", b"ab", b"bc", b"x0", None], "eos_id": 16},
    {"name": "escapes", "pattern": r"\(\d+\.\d\)|\[\w\s?\]", "vocab": [bytes([c]) for c in b"()[].0123456789ab _"] + [b"(1", b".5)", None], "eos_id": 21},
    {
        "name": "json_object",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
            "required": ["name"],
        },
        "vocab": JSON_VOCAB,
        "eos_id": len(JSON_VOCAB) - 2,
    },
    {
        "name": "json_enum_number_bool",
        "schema": {"type": "array", "items": {"anyOf": [{"enum": ["on", "off", 3]}, {"type": "number"}, {"type": "boolean"}]}, "minItems": 1, "maxItems": 3},
        "vocab": JSON_VOCAB,
        "eos_id": len(JSON_VOCAB) - 2,
    },
]


def allowed(tree, prefix: bytes, vocab, eos_id) -> list[int]:
    out = []
    d = deriv_bytes(tree, prefix)
    for i, tb in enumerate(vocab):
        if i == eos_id:
            if nullable(d):
                out.append(i)
        elif tb and not is_empty(deriv_bytes(d, tb)):
            out.append(i)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    rng = random.Random(87)
    cases = []
    for case in CASES:
        if "pattern" in case:
            tree = from_sre(P.parse(case["pattern"], 0))
        else:
            tree = schema_tree(case["schema"])
        vocab, eos = case["vocab"], case["eos_id"]
        paths = []
        for _ in range(3):
            prefix, steps = b"", []
            for _ in range(14):
                ids = allowed(tree, prefix, vocab, eos)
                complete = nullable(deriv_bytes(tree, prefix))
                choice = [i for i in ids if i != eos]
                if not choice or (eos in ids and rng.random() < 0.25):
                    tok = eos if eos in ids else None
                else:
                    tok = rng.choice(choice)
                steps.append({"mask": ids, "complete": complete, "token": tok})
                if tok is None or tok == eos:
                    break
                prefix += vocab[tok]
            paths.append(steps)
        out = {k: v for k, v in case.items() if k != "vocab"}
        out["vocab"] = [None if v is None else v.hex() for v in vocab]
        out["paths"] = paths
        cases.append(out)
    doc = {
        "generator": "course/oracle/L8.7/masks_golden.py (Brzozowski derivatives over re._parser trees)",
        "cases": cases,
    }
    dst = root / "course" / "fixtures" / "L8.7" / "masks_golden.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1) + "\n"  # key order matters: properties are ordered
    dst.write_text(text)
    data = text.encode()
    print(
        "\t".join(
            [
                "course/fixtures/L8.7/masks_golden.json",
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/L8.7/masks_golden.py",
                "python stdlib",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
