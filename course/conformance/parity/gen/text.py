"""Fuzz inputs for tokenizers: gen/text.py <seed> <n> prints n JSON cases
{"text": ...} mixing ASCII words, whitespace runs, digits, punctuation,
accented Latin, CJK, emoji (astral plane), and combining marks."""

import json
import random
import sys

POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    " \t\n  ",
    "0123456789",
    ".,;:!?'\"()[]{}-_/\\@#$%^&*+=<>|~`",
    "éèêüñçßø",
    "中文字词日本語",
    "\U0001f600\U0001f680\U0001f9e0\U0001f308",
    "éäõ",
]

rng = random.Random(int(sys.argv[1]))
for _ in range(int(sys.argv[2])):
    parts = []
    for _ in range(rng.randint(0, 12)):
        pool = rng.choice(POOLS)
        parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
    print(json.dumps({"text": "".join(parts)}))
