"""Maintainer generator for the Bloom filter goldens (ds.08, parity suite `bloom`).

    uv run --project course/harness python course/oracle/ds.08/bloom_golden.py [--check]

An independent transcription of contracts/formats/bloom.md in plain Python
integers (no Rust, no float tricks beyond the two sizing formulas), so a
Rust or Python filter that agrees with these bytes agrees with the page.
Writes course/fixtures/parity/bloom.json:

    {"cases": [{"name", "input": {"n", "p", "insert": [hex], "probe": [hex]},
                "output": {"m", "k", "bytes": hex of to_bytes(), "contains": [bool per probe]}}]}

Items are byte strings written as lowercase hex, so every byte value can
appear. Case 0 is the worked example of formats/bloom.md (cat, dog, bird).
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

M64 = (1 << 64) - 1
FNV_OFFSET, FNV_PRIME = 0xCBF29CE484222325, 0x100000001B3


def fnv1a64(b: bytes) -> int:
    h = FNV_OFFSET
    for x in b:
        h = ((h ^ x) * FNV_PRIME) & M64
    return h


def mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def sizing(n: int, p: float) -> tuple[int, int]:
    m = math.ceil(-n * math.log(p) / (math.log(2) ** 2))
    k = m / n * math.log(2)
    k = int(math.floor(k + 0.5))  # round half away from zero (k > 0)
    return m, max(1, k)


def positions(item: bytes, m: int, k: int) -> list[int]:
    h1 = fnv1a64(item)
    h2 = mix64(h1) | 1
    return [((h1 + i * h2) & M64) % m for i in range(k)]


def build(n: int, p: float, insert: list[bytes]) -> tuple[int, int, bytearray]:
    m, k = sizing(n, p)
    bits = bytearray((m + 7) // 8)
    for it in insert:
        for g in positions(it, m, k):
            bits[g // 8] |= 1 << (g % 8)
    return m, k, bits


def to_bytes(m: int, k: int, count: int, bits: bytes) -> bytes:
    return (
        b"TLBF"
        + (1).to_bytes(4, "little")
        + m.to_bytes(8, "little")
        + k.to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + count.to_bytes(8, "little")
        + bytes(bits)
    )


def case(name: str, n: int, p: float, insert: list[bytes], probe: list[bytes]) -> dict:
    m, k, bits = build(n, p, insert)
    hit = [all(bits[g // 8] >> (g % 8) & 1 for g in positions(x, m, k)) for x in probe]
    return {
        "name": name,
        "input": {
            "n": n,
            "p": p,
            "insert": [x.hex() for x in insert],
            "probe": [x.hex() for x in probe],
        },
        "output": {
            "m": m,
            "k": k,
            "bytes": to_bytes(m, k, len(insert), bits).hex(),
            "contains": hit,
        },
    }


def cases() -> list[dict]:
    rng = random.Random(20261009)
    out = [case("worked_example", 4, 0.1, [b"cat", b"dog"], [b"cat", b"dog", b"bird"])]
    out.append(case("empty_item", 3, 0.2, [b""], [b"", b"\x00"]))
    out.append(case("one_item_k1", 1, 0.5, [b"x"], [b"x", b"y", b"z"]))
    for i, (n, p) in enumerate(
        [(10, 0.01), (50, 0.05), (100, 0.001), (7, 0.3), (1000, 0.02), (33, 1e-6)]
    ):
        ins = [
            bytes(rng.randrange(256) for _ in range(rng.randrange(0, 24)))
            for _ in range(n)
        ]
        probe = ins[: min(10, n)] + [
            bytes(rng.randrange(256) for _ in range(rng.randrange(1, 24)))
            for _ in range(30)
        ]
        out.append(case(f"random_{i}_n{n}", n, p, ins, probe))
    # m not a multiple of 8: the last byte's unused bits stay 0
    out.append(case("odd_m", 5, 0.25, [b"a", b"b", b"c", b"d", b"e"], [b"a", b"f"]))
    return out


def main() -> int:
    root = Path.cwd()
    path = root / "course" / "fixtures" / "parity" / "bloom.json"
    text = json.dumps({"cases": cases()}, separators=(",", ":")) + "\n"
    w = cases()[0]
    assert (w["output"]["m"], w["output"]["k"]) == (20, 3)
    assert w["output"]["bytes"].endswith("096802"), w["output"]["bytes"]
    assert w["output"]["contains"] == [True, True, False]
    if "--check" in sys.argv[1:]:
        same = path.is_file() and path.read_text() == text
        print(("same  " if same else "DIFF  ") + str(path.relative_to(root)))
        return 0 if same else 1
    path.write_text(text)
    print(f"wrote {path.relative_to(root)} ({len(text)} bytes, {len(cases())} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
