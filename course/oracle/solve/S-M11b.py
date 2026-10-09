"""Recompute every auto-checked answer of S-M11b independently (see _confirm.py).

Huffman lengths by a from-scratch heap construction over exact rationals
(depth of each leaf in the merge tree); expected lengths and Kraft sums as
direct sums; Shannon lengths by integer search for the least l with 2^l >= 1/p;
mutual information and PMI as direct sums over the joint in SymPy (base 2);
I(X; X mod 2) from the joint table; the softmax probabilities by evaluating
the formula and the T -> oo limit with sympy.limit; R(D) by evaluating the
given formulas and solving for the rate.

    uv run --project course/harness python course/oracle/solve/S-M11b.py
"""

from __future__ import annotations

import heapq
import itertools
import sys
from fractions import Fraction
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

R = sp.Rational


def huffman_lengths(ps: list[Fraction]) -> list[int]:
    tie = itertools.count()
    heap = [(p, next(tie), [i]) for i, p in enumerate(ps)]
    heapq.heapify(heap)
    depth = [0] * len(ps)
    while len(heap) > 1:
        p1, _, a = heapq.heappop(heap)
        p2, _, b = heapq.heappop(heap)
        for i in a + b:
            depth[i] += 1
        heapq.heappush(heap, (p1 + p2, next(tie), a + b))
    return depth


def shannon_length(p: Fraction) -> int:
    l = 0
    while Fraction(2) ** l < 1 / p:
        l += 1
    return l


def frac(q: Fraction) -> sp.Rational:
    return R(q.numerator, q.denominator)


def main() -> int:
    c: dict[str, str] = {}
    p1 = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)]
    l1 = huffman_lengths(p1)
    c["q1.a"] = "[" + ", ".join(map(str, l1)) + "]"
    L1 = sum(p * l for p, l in zip(p1, l1))
    c["q1.b"] = str(frac(L1))
    H1 = sum(-frac(p) * sp.log(frac(p), 2) for p in p1)
    c["q1.c"] = str(sp.simplify(H1 - frac(L1)) == 0).lower()

    p2 = [Fraction(4, 10), Fraction(3, 10), Fraction(2, 10), Fraction(1, 10)]
    l2 = huffman_lengths(p2)
    c["q2.a"] = str(frac(sum(p * l for p, l in zip(p2, l2))))
    c["q2.b"] = str(frac(sum(Fraction(1, 2**l) for l in l2)))
    ls = [shannon_length(p) for p in p2]
    c["q3.a"] = str(frac(sum(p * l for p, l in zip(p2, ls))))
    c["q3.b"] = str(sum(Fraction(1, 2**l) for l in (1, 2, 2, 3)) <= 1).lower()

    joint = {(0, 0): R(3, 8), (1, 1): R(3, 8), (0, 1): R(1, 8), (1, 0): R(1, 8)}
    px = {x: sum(v for (a, _), v in joint.items() if a == x) for x in (0, 1)}
    py = {y: sum(v for (_, b), v in joint.items() if b == y) for y in (0, 1)}
    mi = sum(
        v * sp.log(v / (px[x] * py[y]), 2) for (x, y), v in joint.items() if v != 0
    )
    c["q5.a"] = str(sp.nsimplify(sp.expand_log(sp.simplify(mi), force=True)))
    c["q5.b"] = str(sp.log(joint[(0, 0)] / (px[0] * py[0]), 2))
    c["q5.c"] = str(sp.simplify(sp.log(joint[(0, 1)] / (px[0] * py[1]), 2)))
    j6 = {}
    for x in range(4):
        j6[(x, x % 2)] = j6.get((x, x % 2), 0) + R(1, 4)
    qx = {x: sum(v for (a, _), v in j6.items() if a == x) for x in range(4)}
    qy = {y: sum(v for (_, b), v in j6.items() if b == y) for y in (0, 1)}
    c["q6.a"] = str(
        sp.simplify(sum(v * sp.log(v / (qx[x] * qy[y]), 2) for (x, y), v in j6.items()))
    )
    mi_t = sum(
        v * sp.log(v / (qy[y] * qx[x]), 2) for (x, y), v in j6.items()
    )  # roles swapped
    c["q6.b"] = str(sp.simplify(mi_t - sp.sympify(c["q6.a"])) == 0).lower()

    z = [sp.Integer(0), sp.log(2), sp.log(3)]
    T = sp.Symbol("T", positive=True)
    p3 = sp.exp(z[2] / T) / sum(sp.exp(zi / T) for zi in z)
    c["q8.a"] = str(sp.simplify(p3.subs(T, 1)))
    c["q8.b"] = str(sp.simplify(p3.subs(T, R(1, 2))))
    lim = [
        sp.limit(sp.exp(zi / T) / sum(sp.exp(zj / T) for zj in z), T, sp.oo) for zi in z
    ]
    c["q8.c"] = "b" if all(v == R(1, 3) for v in lim) else "a"

    def hb(d):
        return -d * sp.log(d, 2) - (1 - d) * sp.log(1 - d, 2)

    c["q10.a"] = str(
        sp.nsimplify(sp.expand_log(sp.simplify(1 - hb(R(1, 8))), force=True))
    )
    c["q10.b"] = str(sp.simplify(1 - hb(R(1, 2))))
    s2, D, rate = sp.symbols("s2 D rate", positive=True)
    RD = sp.log(s2 / D, 2) / 2
    c["q11.a"] = str(sp.simplify(RD.subs(D, s2 / 16)))
    Dof = sp.solve(sp.Eq(RD, rate), D)[0]
    c["q11.b"] = str(sp.simplify(Dof / Dof.subs(rate, rate + 1)))
    return confirm("S-M11b", c)


if __name__ == "__main__":
    sys.exit(main())
