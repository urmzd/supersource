"""Recompute every auto-checked answer of S-M06a independently (see _confirm.py).

Graph answers by brute force over permutations and explicit path enumeration,
matrix powers with numpy, modular and bit arithmetic with Python integers,
expected collisions by exact enumeration of small cases fitted to a formula.
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

E1 = [(1, 3), (2, 3), (3, 5), (2, 4), (4, 5)]
NODES = [1, 2, 3, 4, 5]


def is_topo(order, edges) -> bool:
    pos = {v: i for i, v in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in edges)


def paths(u, v, edges) -> int:
    if u == v:
        return 1
    return sum(paths(w, v, edges) for a, w in edges if a == u)


def has_cycle(nodes, edges) -> bool:
    return not any(is_topo(p, edges) for p in itertools.permutations(nodes))


def fmt(m) -> str:
    return (
        "[" + ", ".join("[" + ", ".join(str(int(x)) for x in r) + "]" for r in m) + "]"
    )


def main() -> int:
    c: dict[str, str] = {}
    c["q1.a"] = str(sum(1 for _, v in E1 if v == 5))
    c["q1.b"] = str(sum(1 for u, _ in E1 if u == 2))
    c["q2.a"] = str(is_topo((1, 2, 3, 4, 5), E1)).lower()
    c["q2.b"] = str(is_topo((2, 4, 3, 1, 5), E1)).lower()
    c["q3"] = str(sum(is_topo(p, E1) for p in itertools.permutations(NODES)))
    c["q4"] = "[" + ", ".join(str(sum(1 for _, v in E1 if v == n)) for n in NODES) + "]"
    c["q5"] = str(paths(2, 5, E1))
    c["q6"] = str(not has_cycle([1, 2, 3, 4], [(1, 2), (2, 3), (3, 1), (3, 4)])).lower()
    A = np.zeros((4, 4), dtype=int)
    for i in range(3):
        A[i, i + 1] = 1
    c["q7.a"] = fmt(np.linalg.matrix_power(A, 2))
    c["q7.b"] = fmt(np.linalg.matrix_power(A, 3))

    # q8: backward needs every consumer of a node before the node itself
    consumers = {"a": {"t"}, "b": {"t", "y"}, "t": {"y"}, "y": set()}
    opts = {"a": "ytab", "b": "abty", "c": "tyab", "d": "byta"}
    ok = [
        k
        for k, o in opts.items()
        if all(o.index(cons) < o.index(n) for n in o for cons in consumers[n])
    ]
    assert len(ok) == 1, ok
    c["q8"] = ok[0]

    # q9: simulate the recursion depth on a chain without recursing
    n, depth, node = 10**5, 1, 10**5 - 1
    while node > 0:
        node, depth = node - 1, depth + 1
    c["q9"] = str(depth)
    assert depth == n

    c["q10.a"] = str(17 % 5)
    c["q10.b"] = str(-17 % 5)  # Python's % is the nonnegative remainder for m > 0
    c["q11"] = str(next(x for x in range(11) if 3 * x % 11 == 1))
    c["q12"] = str((4294967295 + 2) & 0xFFFFFFFF)
    x, r = 2147483649, 1
    bits = [(x >> i) & 1 for i in range(32)]
    rot = [0] * 32
    for i in range(32):
        rot[(i - r) % 32] = bits[i]
    c["q13"] = str(sum(b << i for i, b in enumerate(rot)))

    h = 14695981039346656037
    for byte in b"a":
        h = ((h ^ byte) * 1099511628211) % 2**64
    c["q14"] = str(h)
    h1 = 14695981039346656037
    for byte in b"a":
        h1 = ((h1 * 1099511628211) % 2**64) ^ byte
    print(f"(FNV-1 of 'a' for the q14 canary: {h1})")

    m_, n_ = 16, 12
    alpha = Fraction(n_, m_)
    c["q15.a"] = f"{alpha.numerator}/{alpha.denominator}"
    c["q15.b"] = f"{(1 + alpha).numerator}/{(1 + alpha).denominator}"
    c["q16"] = str(((3 * 10 + 5) % 17) % 8)

    # q17: exact expectation by enumeration for small (n, m), then fit
    def expected_pairs(nk, mb) -> Fraction:
        tot = Fraction(0)
        for assign in itertools.product(range(mb), repeat=nk):
            tot += sum(
                1
                for i, j in itertools.combinations(range(nk), 2)
                if assign[i] == assign[j]
            )
        return tot / mb**nk

    nn, mm = sp.symbols("n m", positive=True)
    cand = nn * (nn - 1) / (2 * mm)
    for nk in range(1, 5):
        for mb in range(1, 4):
            assert expected_pairs(nk, mb) == Fraction(str(cand.subs({nn: nk, mm: mb})))
    c["q17"] = str(cand)

    c["q23"] = str(len({6 * t % 16 for t in range(16)}) == 16).lower()
    return confirm("S-M06a", c)


if __name__ == "__main__":
    sys.exit(main())
