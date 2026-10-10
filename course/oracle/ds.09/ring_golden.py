"""Golden key-to-node maps for parity/ring.hash (ds.09).

An independent Python reference of the consistent hash ring the ds.09 chapter
specifies: Mix64(FNV-1a 64) over the vnode label "<node>#<i>", points sorted by
(position, node, index), a key owned by the first point at or after
Mix64(FNV-1a(key)) (wrapping), and bounded loads that walk clockwise to the first
node whose load is below ceil(c * (total + 1) / n).

    python course/oracle/ds.09/ring_golden.py   # rewrites course/fixtures/parity/ring_hash.json

Each case's input is what drivers/ring_hash.go reads on one stdin line:
{"vnodes", "nodes", "remove", "keys", "bounded": {"c"} | null}. With
"bounded", keys are placed one after another and each placement adds 1 to
its node's load, so the output also pins the load-feedback order.
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

OFFSET = 14695981039346656037
PRIME = 1099511628211
MASK = (1 << 64) - 1


def fnv1a64(b: bytes) -> int:
    h = OFFSET
    for c in b:
        h ^= c
        h = (h * PRIME) & MASK
    return h


def mix64(z: int) -> int:
    """The SplitMix64 finalizer (spec/pcg32.md, formats/bloom.md)."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def hash64(b: bytes) -> int:
    return mix64(fnv1a64(b))


class Ring:
    def __init__(self, vnodes: int) -> None:
        self.vnodes = vnodes
        self.points: list[tuple[int, str, int]] = []
        self.nodes: set[str] = set()

    def add(self, node: str) -> None:
        if node in self.nodes:
            return
        self.nodes.add(node)
        for i in range(self.vnodes):
            self.points.append((hash64(f"{node}#{i}".encode()), node, i))
        self.points.sort()

    def remove(self, node: str) -> None:
        self.nodes.discard(node)
        self.points = [p for p in self.points if p[1] != node]

    def _first(self, x: int) -> int:
        i = bisect.bisect_left([p[0] for p in self.points], x)
        return 0 if i == len(self.points) else i

    def get(self, key: bytes) -> str:
        return self.points[self._first(hash64(key))][1]

    def get_bounded(self, key: bytes, load: dict[str, int], c: float) -> str:
        c = max(c, 1.0)
        total = sum(load.get(n, 0) for n in self.nodes)
        limit = math.ceil(c * (total + 1) / len(self.nodes))
        start = self._first(hash64(key))
        seen: set[str] = set()
        for k in range(len(self.points)):
            node = self.points[(start + k) % len(self.points)][1]
            if node in seen:
                continue
            seen.add(node)
            if load.get(node, 0) < limit:
                return node
        return self.points[start][1]


def run(inp: dict) -> dict:
    r = Ring(inp["vnodes"])
    for n in inp["nodes"]:
        r.add(n)
    for n in inp.get("remove", []):
        r.remove(n)
    owners = []
    if inp.get("bounded"):
        load: dict[str, int] = {}
        for k in inp["keys"]:
            n = r.get_bounded(k.encode(), load, inp["bounded"]["c"])
            load[n] = load.get(n, 0) + 1
            owners.append(n)
    else:
        owners = [r.get(k.encode()) for k in inp["keys"]]
    return {"owners": owners}


def cases() -> list[dict]:
    out = []

    def case(name, vnodes, nodes, keys, remove=(), bounded=None):
        inp = {
            "vnodes": vnodes,
            "nodes": list(nodes),
            "remove": list(remove),
            "keys": keys,
            "bounded": bounded,
        }
        out.append({"name": name, "input": inp, "output": run(inp)})

    eng = [f"engine-{c}" for c in "abcdefgh"]
    case("three-nodes-4-vnodes", 4, eng[:3], [f"key-{i}" for i in range(64)])
    case("eight-nodes-160-vnodes", 160, eng, [f"prefix-{i:04d}" for i in range(400)])
    case(
        "insertion-order-irrelevant",
        40,
        list(reversed(eng[:5])),
        [f"k{i}" for i in range(100)],
    )
    case(
        "remove-two",
        40,
        eng[:6],
        [f"k{i}" for i in range(100)],
        remove=["engine-b", "engine-e"],
    )
    case("single-node", 3, ["only"], ["", "a", "b", "zzz"])
    case(
        "bounded-c1.25",
        40,
        eng[:4],
        [f"hot-{i % 3}" for i in range(60)],
        bounded={"c": 1.25},
    )
    case(
        "bounded-c1",
        40,
        eng[:4],
        [f"hot-{i % 2}" for i in range(41)],
        bounded={"c": 1.0},
    )
    case(
        "bounded-c2-spread",
        80,
        eng[:5],
        [f"s{i}" for i in range(150)],
        bounded={"c": 2.0},
    )
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dst = root / "fixtures" / "parity" / "ring_hash.json"
    doc = {"generator": "course/oracle/ds.09/ring_golden.py", "cases": cases()}
    dst.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(dst, dst.stat().st_size)


if __name__ == "__main__":
    main()
