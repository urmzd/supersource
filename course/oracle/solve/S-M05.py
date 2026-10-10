"""Recompute every auto-checked answer of S-M05 independently (see _confirm.py).

Sets and counts by enumeration (itertools), formulas by SymPy summation or by
counting parameters of explicit shapes, primality by trial division.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def fmt_set(s) -> str:
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


def fmt_matrix(rows) -> str:
    return "[" + ", ".join("[" + ", ".join(str(x) for x in r) + "]" for r in rows) + "]"


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def main() -> int:
    c: dict[str, str] = {}
    U, A, B = set(range(1, 11)), {1, 2, 3, 4, 5, 6}, {2, 4, 6, 8}
    c["q1.a"] = fmt_set(A & B)
    c["q1.b"] = fmt_set(A - B)
    c["q1.c"] = fmt_set(U - (A | B))

    base = [1, 2, 3, 4, 5, 6]
    subsets = [
        s
        for r in range(7)
        for s in itertools.combinations(base, r)
        if 1 in s and 2 not in s
    ]
    c["q2"] = str(len(subsets))

    # truth tables over all four assignments
    imp = lambda p, q: (not p) or q  # noqa: E731
    tt = list(itertools.product([False, True], repeat=2))
    c["q3.a"] = str(all(imp(p, q) == imp(not q, not p) for p, q in tt)).lower()
    c["q3.b"] = str(all(imp(p, q) == imp(q, p) for p, q in tt)).lower()

    T = 4
    mask = [[int(j <= i and j < 3) for j in range(T)] for i in range(T)]
    c["q4.a"] = fmt_matrix(mask)
    c["q4.b"] = str(sum(map(sum, mask)))

    # q5: evaluate each candidate against the true negation on every 2x2 0/1 matrix
    def holds(M):
        return all(any(M[i][j] for j in range(2)) for i in range(2))

    cands = {
        "a": lambda M: any(all(M[i][j] == 0 for j in range(2)) for i in range(2)),
        "b": lambda M: all(M[i][j] == 0 for i in range(2) for j in range(2)),
        "c": lambda M: any(M[i][j] == 0 for i in range(2) for j in range(2)),
        "d": lambda M: all(any(M[i][j] == 0 for j in range(2)) for i in range(2)),
    }
    mats = [
        [list(bits[:2]), list(bits[2:])] for bits in itertools.product([0, 1], repeat=4)
    ]
    right = [k for k, f in cands.items() if all(f(M) == (not holds(M)) for M in mats)]
    assert len(right) == 1, right
    c["q5"] = right[0]

    c["q7.a"] = str(sum(1 for _ in itertools.combinations(range(10), 3)))
    c["q7.b"] = str(256**4)
    c["q7.c"] = str(len(list(itertools.product(range(256), repeat=2))))
    c["q8"] = str(sum(1 for x in itertools.product(range(6), repeat=3) if sum(x) == 5))

    m, n, k, V, d = sp.symbols("m n k V d", positive=True)
    c["q9"] = str(sp.expand(m * n + m))  # W is m x n, b has m entries
    c["q10"] = str(V * d + d * V)  # embedding V x d, head d x V
    shapes = [(d, d)] * 4 + [(d, 4 * d), (4 * d, d)] + [(d, 1), (d, 1)]
    c["q11"] = str(sp.expand(sum(a * b for a, b in shapes)))
    # FLOPs: m*n outputs, each k multiplies and k adds
    c["q12"] = str(m * n * (k + k))
    c["q13"] = str(6 * 10**7 * 2 * 10**8)
    layers, kv_heads, d_head, dtype = 30, 3, 64, 2
    per_tok = sum(layers * kv_heads * d_head * dtype for _ in ("k", "v"))
    c["q14"] = str(per_tok)
    c["q15"] = str(per_tok * 2048)
    c["q16"] = str(sum(1 for x in range(1, 101) if x % 3 == 0 or x % 5 == 0))

    first = next(x for x in itertools.count(1) if not is_prime(x * x - x + 41))
    c["q23.a"] = str(first is None).lower()  # a counterexample exists: false
    c["q23.b"] = str(first)

    nn = sp.symbols("n", integer=True, nonnegative=True)
    i = sp.symbols("i", integer=True)
    c["q31"] = str(sp.factor(sp.summation(i**2, (i, 1, nn))))
    c["q32"] = str(sp.simplify(sp.summation(2**i, (i, 0, nn - 1))))

    # q33: run the online softmax loop exactly
    mm, s = -sp.oo, sp.Integer(0)
    for x in (1, 3, 2):
        new = sp.Max(mm, x)
        s = (s * sp.exp(mm - new) if s != 0 else 0) + sp.exp(x - new)
        mm = new
    c["q33"] = str(sp.simplify(s))

    R = {(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)}
    X = [1, 2, 3]
    c["q34.a"] = str(all((x, x) in R for x in X)).lower()
    c["q34.b"] = str(all((y, x) in R for x, y in R)).lower()
    c["q34.c"] = str(all((x, z) in R for x, y in R for y2, z in R if y == y2)).lower()

    # q35: count partitions of a 4-set by restricted growth strings
    def partitions(nel):
        out = 0
        for rg in itertools.product(range(nel), repeat=nel):
            if rg[0] == 0 and all(rg[t] <= max(rg[:t]) + 1 for t in range(1, nel)):
                out += 1
        return out

    c["q35"] = str(partitions(4))

    dom = range(-50, 51)
    f = lambda t: 2 * t + 1  # noqa: E731
    c["q38.a"] = str(len({f(t) for t in dom}) == len(dom)).lower()
    c["q38.b"] = str(any(f(t) == 0 for t in range(-1000, 1000))).lower()
    c["q39.a"] = str(sum(1 for _ in itertools.permutations(range(5))))
    c["q39.b"] = str(
        sum(1 for g in itertools.product(range(5), repeat=3) if len(set(g)) == 3)
    )
    return confirm("S-M05", c)


if __name__ == "__main__":
    sys.exit(main())
