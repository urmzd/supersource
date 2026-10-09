"""Maintainer generator for the L8.4 reference trace (radix prefix cache).

    uv run --project course/harness python course/oracle/L8.4/radix_trace.py [--check]

An independent Python model of the prefix cache the L8.4 chapter specifies:
a radix tree over token ids at block granularity, stamped with a clock that
advances once per match or insert, matches that leave the last prompt token
uncovered, inserts that hand back the caller's blocks for an already-cached
prefix when they differ from the cached ones, locks that pin a node and its
ancestors, and eviction of the unlocked leaf with the oldest stamp (found
here by a linear scan, not a linked list), whole leaves at a time, until at
least n blocks are freed.

It plays a seeded workload (prompts drawn from four shared system prompts
with random tails, a running set of locked requests, evictions under
pressure) and writes course/fixtures/L8.4/radix_trace.txt, one operation
per line, with the results the model gives:

    B <block_size>
    M <tokens...> ; <matched> <blocks...>          match_prefix
    I <tokens...> / <blocks...> ; <duplicates...>  insert
    L <k>  /  U <k>                                lock / unlock the node of op line k
    E <n> ; <blocks...>                            evict(n), in eviction order
    C <cached_tokens>                              after every operation

Op lines are numbered from 0 in file order (C and B lines are not ops).
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "fixtures" / "L8.4" / "radix_trace.txt"
B = 4


class Pcg32:
    def __init__(self, seed: int, seq: int = 54) -> None:
        self.state, self.inc = 0, ((seq << 1) | 1) & (2**64 - 1)
        self.next()
        self.state = (self.state + seed) & (2**64 - 1)
        self.next()

    def next(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & (2**64 - 1)
        xs = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & 0xFFFFFFFF

    def below(self, n: int) -> int:
        return self.next() % n


class Node:
    def __init__(self, key, blocks, parent, stamp):
        self.key, self.blocks, self.parent, self.stamp = list(key), list(blocks), parent, stamp
        self.children: dict[tuple, Node] = {}
        self.lock = 0


class Cache:
    def __init__(self, b: int) -> None:
        self.b, self.root, self.clock = b, Node([], [], None, 0), 0

    def _walk(self, toks):
        b, cur, done, path = self.b, self.root, 0, []
        while len(toks) - done >= b:
            child = cur.children.get(tuple(toks[done : done + b]))
            if child is None:
                break
            m = 0
            while (m + 1) * b <= len(child.key) and (m + 1) * b <= len(toks) - done and \
                    child.key[m * b : (m + 1) * b] == toks[done + m * b : done + (m + 1) * b]:
                m += 1
            node = child
            if m * b < len(child.key):  # split
                mid = Node(child.key[: m * b], child.blocks[:m], cur, self.clock)
                mid.lock = child.lock
                child.key, child.blocks = child.key[m * b :], child.blocks[m:]
                child.parent = mid
                mid.children[tuple(child.key[:b])] = child
                cur.children[tuple(mid.key[:b])] = mid
                node = mid
            node.stamp = self.clock
            path.append(node)
            done += m * b
            cur = node
            if node is not child:
                break
        return done, path

    def match(self, toks):
        self.clock += 1
        usable = (len(toks) - 1) // self.b * self.b if toks else 0
        done, path = self._walk(toks[:usable])
        blocks = [x for n in path for x in n.blocks]
        return done, blocks, (path[-1] if path else self.root)

    def insert(self, toks, blocks):
        full = len(toks) // self.b
        assert len(blocks) == full
        self.clock += 1
        done, path = self._walk(toks[: full * self.b])
        last = path[-1] if path else self.root
        back, fresh = blocks[: done // self.b], blocks[done // self.b :]
        kept = [x for n in path for x in n.blocks]
        dups = [a for a, k in zip(back, kept) if a != k]
        if fresh:
            leaf = Node(toks[done : full * self.b], fresh, last, self.clock)
            last.children[tuple(leaf.key[: self.b])] = leaf
            return leaf, dups
        return last, dups

    def lock(self, n, d):
        while n is not self.root:
            n.lock += d
            assert n.lock >= 0
            n = n.parent

    def nodes(self):
        out, stack = [], list(self.root.children.values())
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def evict(self, n):
        out = []
        while len(out) < n:
            leaves = [x for x in self.nodes() if not x.children and x.lock == 0]
            if not leaves:
                break
            v = min(leaves, key=lambda x: x.stamp)
            del v.parent.children[tuple(v.key[: self.b])]
            out.extend(v.blocks)
        return out

    def cached_tokens(self):
        return sum(len(n.blocks) for n in self.nodes()) * self.b


def trace() -> str:
    r = Pcg32(2026, 84)
    c = Cache(B)
    systems = [[r.below(50) for _ in range(B * (1 + r.below(4)))] for _ in range(4)]
    lines = ["# L8.4 reference trace: course/oracle/L8.4/radix_trace.py (do not edit)", f"B {B}"]
    ops = 0
    nodes: dict[int, Node] = {}
    running: list[int] = []  # op lines whose node is locked
    next_block = 100
    for _ in range(320):
        k = r.below(10)
        if k < 5:
            prompt = list(systems[r.below(4)]) + [r.below(50) for _ in range(r.below(11))]
            matched, blocks, node = c.match(prompt)
            lines.append(f"M {' '.join(map(str, prompt))} ; {matched} {' '.join(map(str, blocks))}".rstrip())
            nodes[ops] = node
            this = ops
            ops += 1
            if r.below(3) and node is not c.root:
                c.lock(node, 1)
                lines.append(f"L {this}")
                running.append(this)
                ops += 1
            if r.below(2):  # the request finishes prefill: its full blocks go in
                full = len(prompt) // B
                new = []
                for _ in range(full - len(blocks)):
                    new.append(next_block)
                    next_block += 1
                own = list(blocks) if r.below(4) else [next_block + 1000 + i for i in range(len(blocks))]
                node2, dups = c.insert(prompt, own + new)
                lines.append(
                    f"I {' '.join(map(str, prompt))} / {' '.join(map(str, own + new))} ; {' '.join(map(str, dups))}".rstrip()
                )
                nodes[ops] = node2
                ops += 1
        elif k < 7 and running:
            line = running.pop(r.below(len(running)))
            c.lock(nodes[line], -1)
            lines.append(f"U {line}")
            ops += 1
        elif k < 9:
            n = 1 + r.below(6)
            out = c.evict(n)
            lines.append(f"E {n} ; {' '.join(map(str, out))}".rstrip())
            ops += 1
        else:
            continue
        lines.append(f"C {c.cached_tokens()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    text = trace()
    if "--check" in sys.argv:
        same = OUT.read_text() == text
        print("radix_trace.txt is current" if same else "radix_trace.txt differs from the oracle")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
