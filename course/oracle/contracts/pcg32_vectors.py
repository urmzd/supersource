# /// script
# requires-python = ">=3.11"
# ///
"""Maintainer generator for course/contracts/spec/pcg32.vectors.json.

Implements spec/pcg32.md directly (stdlib only, independent of every
course and learner implementation) and writes the reference vectors the
parity/rng suite and the Python, C, Rust, and Go ports are checked against.

    uv run --script course/oracle/contracts/pcg32_vectors.py          # rewrite
    uv run --script course/oracle/contracts/pcg32_vectors.py --check  # compare

The harness test test_contracts_b2_b13.py cross-checks the file against an
independent C transcription of O'Neill's pcg32_srandom_r / pcg32_random_r.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1
MULT = 6364136223846793005
GOLDEN = 0x9E3779B97F4A7C15
PURPOSES = {"init": 1, "dropout": 2, "shuffle": 3, "sample": 4, "mutation": 5}
OUT = Path(__file__).resolve().parents[2] / "contracts" / "spec" / "pcg32.vectors.json"


def mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


class Pcg32:
    def __init__(self, seed: int, seq: int = 54) -> None:
        self.state = 0
        self.inc = ((seq << 1) | 1) & M64
        self.next_u32()
        self.state = (self.state + seed) & M64
        self.next_u32()
        self.spare: float | None = None

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * MULT + self.inc) & M64
        xs = (((old >> 18) ^ old) >> 27) & M32
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & M32

    def uniform(self) -> float:
        a, b = self.next_u32() >> 5, self.next_u32() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def normal(self) -> float:
        if self.spare is not None:
            z, self.spare = self.spare, None
            return z
        u1, u2 = self.uniform(), self.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        self.spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)

    def below(self, n: int) -> int:
        threshold = ((1 << 32) - n) % n
        while True:
            r = self.next_u32()
            if r >= threshold:
                return r % n

    def shuffle(self, xs: list) -> list:
        xs = list(xs)
        for i in range(len(xs) - 1, 0, -1):
            j = self.below(i + 1)
            xs[i], xs[j] = xs[j], xs[i]
        return xs


def child_seed(seed: int, purpose_id: int) -> int:
    return mix64((seed + purpose_id * GOLDEN) & M64)


def stream(seed: int, purpose: str) -> Pcg32:
    pid = PURPOSES[purpose]
    return Pcg32(child_seed(seed, pid), pid)


def build() -> dict:
    seeds = [0, 1, 1 << 63]
    out: dict = {
        "spec": "spec/pcg32.md",
        "seq": 54,
        "next_u32": {
            str(s): [g.next_u32() for _ in range(1024)]
            for s in seeds
            for g in [Pcg32(s)]
        },
    }
    uni, nor, bel = {}, {}, {}
    for s in seeds:
        g = Pcg32(s)
        uni[str(s)] = [g.uniform() for _ in range(8)]
        g = Pcg32(s)
        nor[str(s)] = [g.normal() for _ in range(8)]
        g = Pcg32(s)
        bel[str(s)] = [g.below(10) for _ in range(16)]
    out["uniform_f64"] = uni
    out["normal"] = nor
    out["below_10"] = bel
    out["shuffle_10"] = {str(s): Pcg32(s).shuffle(range(10)) for s in seeds}
    out["purposes"] = PURPOSES
    out["child_seed"] = {
        str(s): {p: child_seed(s, i) for p, i in PURPOSES.items()} for s in seeds
    }
    out["stream_next_u32"] = {
        str(s): {
            p: [g.next_u32() for _ in range(4)]
            for p in PURPOSES
            for g in [stream(s, p)]
        }
        for s in seeds
    }
    return out


def render(d: dict) -> str:
    return json.dumps(d, indent=1) + "\n"


def same(a, b) -> bool:
    """Integers and strings exactly; floats to 1e-12 relative.

    The normal() vectors go through libm log/sin/cos, which differ in the last
    bit between macOS and glibc, so a byte-for-byte text compare only passes on
    the machine that generated the file.
    """
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(same, a, b))
    return a == b


if __name__ == "__main__":
    text = render(build())
    if "--check" in sys.argv:
        sys.exit(0 if same(json.loads(OUT.read_text()), json.loads(text)) else 1)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
