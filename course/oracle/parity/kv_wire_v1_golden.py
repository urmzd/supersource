"""Maintainer generator for the KV export envelope goldens (rt.04, L10.6;
parity suite `kv.wire.v1`).

    uv run --project course/harness python course/oracle/parity/kv_wire_v1_golden.py [--check]

An independent transcription of contracts/formats/kv-block.md in plain
Python (struct, a bitwise CRC-32C, FNV-1a over bytes): a sequence's tokens
are cut into blocks of B; each full block is hashed with the chained FNV-1a
64 (0 replaced by 1), the partial tail gets hash 0; each block's payload is
f16 [layer][K, V][head][position][dim] with positions past its fill written
as zeros; the trailer is the CRC-32C of everything before it.

Writes course/fixtures/parity/kv_wire_v1.json:

    {"cases": [{"name", "input": {"B", "L", "H", "D", "tokens": [...],
                                  "values": [f16 bit patterns]}, "output": {"hex": envelope}}]}

`values` holds the sequence's K and V, row-major [position][layer][K, V][head][dim]
(n_tokens * L * 2 * H * D numbers). Case 0 is formats/kv-block.md's worked
example (60 bytes, CRC 0x4F0B541C).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "fixtures" / "parity" / "kv_wire_v1.json"
M64 = 2**64 - 1


def fnv1a64(data: bytes, h: int = 0xCBF29CE484222325) -> int:
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & M64
    return h


def crc32c(data: bytes) -> int:
    c = 0xFFFFFFFF
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
    return c ^ 0xFFFFFFFF


def envelope(B: int, L: int, H: int, D: int, tokens: list[int], values: list[int]) -> bytes:
    n = len(tokens)
    nb = (n + B - 1) // B
    out = bytearray(b"TLKV" + struct.pack("<HHIIIII", 1, 1, nb, B, L, H, D))
    parent = 0
    for i in range(nb):
        toks = tokens[i * B : (i + 1) * B]
        if len(toks) == B:
            h = fnv1a64(struct.pack("<Q", parent) + b"".join(struct.pack("<I", t) for t in toks)) or 1
            parent = h
        else:
            h = 0
        out += struct.pack("<QI", h, len(toks))
        for layer in range(L):
            for kv in range(2):
                for head in range(H):
                    for t in range(B):
                        p = i * B + t
                        for d in range(D):
                            v = values[(((p * L + layer) * 2 + kv) * H + head) * D + d] if t < len(toks) else 0
                            out += struct.pack("<H", v)
    out += struct.pack("<I", crc32c(bytes(out)))
    return bytes(out)


class Pcg32:
    def __init__(self, seed: int, seq: int = 54) -> None:
        self.state, self.inc = 0, ((seq << 1) | 1) & M64
        self.next()
        self.state = (self.state + seed) & M64
        self.next()

    def next(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & M64
        xs = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & 0xFFFFFFFF


def cases() -> list[dict]:
    out = []

    def add(name, B, L, H, D, tokens, values):
        out.append({"name": name, "input": {"B": B, "L": L, "H": H, "D": D, "tokens": tokens, "values": values},
                    "output": {"hex": envelope(B, L, H, D, tokens, values).hex()}})

    # formats/kv-block.md: K = [[1, 2], [3, 4]], V = [[0.5, -1], [0, 0.25]] as [position][dim]
    add("worked-example", 2, 1, 1, 2, [1, 2],
        [0x3C00, 0x4000, 0x3800, 0xBC00, 0x4200, 0x4400, 0x0000, 0x3400])
    r = Pcg32(4096, 6)
    for i, (B, L, H, D, n) in enumerate([(2, 1, 1, 2, 5), (4, 2, 2, 3, 9), (16, 2, 1, 4, 33), (3, 3, 2, 2, 6), (8, 1, 4, 8, 7)]):
        tokens = [r.next() % 50000 for _ in range(n)]
        # finite f16 patterns only (exponent field below 31)
        values = []
        for _ in range(n * L * 2 * H * D):
            v = r.next() & 0xFFFF
            values.append(v if (v >> 10) & 0x1F != 0x1F else v & 0x83FF)
        add(f"random-{i}", B, L, H, D, tokens, values)
    return out


def main() -> int:
    doc = {"generator": "course/oracle/parity/kv_wire_v1_golden.py", "cases": cases()}
    assert doc["cases"][0]["output"]["hex"].endswith("1c540b4f"), "the worked example's CRC"
    text = json.dumps(doc, separators=(",", ":")) + "\n"
    if "--check" in sys.argv:
        same = OUT.read_text() == text
        print("kv_wire_v1.json is current" if same else "kv_wire_v1.json differs from the oracle")
        return 0 if same else 1
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes, {len(doc['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
