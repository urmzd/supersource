"""The fixture system's byte bigram (shared by its engine and its CLI).

Byte-level (D32): token ids are UTF-8 bytes. Greedy decoding follows a fixed
table; sampling draws from a seeded stream, so the same seed repeats.
"""

import random


def greedy_next(prev: int) -> int:
    return 97 + (prev * 7 + 3) % 26


def generate(prompt: str, max_tokens: int, temperature: float = 0.0, seed: int = 0) -> list[int]:
    data = prompt.encode()
    prev = data[-1] if data else 0
    rng = random.Random(seed)
    out = []
    for _ in range(max_tokens):
        nxt = greedy_next(prev) if temperature == 0 else 97 + (prev + rng.randrange(26)) % 26
        out.append(nxt)
        prev = nxt
    return out


def logits(prompt: str, prefix: list[int]) -> list[float]:
    data = list(prompt.encode()) + prefix
    prev = data[-1] if data else 0
    row = [0.0] * 256
    row[greedy_next(prev)] = 5.0
    return row
