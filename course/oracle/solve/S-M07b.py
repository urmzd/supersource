"""Recompute every auto-checked answer of S-M07b independently (see _confirm.py).

Joint-table questions by exact summation over the table with Fractions,
Var(aX + bY) by SymPy expansion of E[(aX + bY - mean)^2] over a symbolic
joint distribution, the MLE questions by solving the score equation (SymPy)
and by direct Fraction counting, and the EM step by enumerating the two
segmentations of the word.
"""

from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def fr(x: F) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def main() -> int:
    c: dict[str, str] = {}

    joint = {
        (0, 0): F(1, 8),
        (0, 1): F(1, 4),
        (0, 2): F(1, 8),
        (1, 0): F(1, 4),
        (1, 1): F(1, 8),
        (1, 2): F(1, 8),
    }
    assert sum(joint.values()) == 1
    px = {x: sum(p for (a, _), p in joint.items() if a == x) for x in (0, 1)}
    py = {y: sum(p for (_, b), p in joint.items() if b == y) for y in (0, 1, 2)}
    c["q1.a"] = fr(py[1])
    c["q1.b"] = fr(joint[(1, 0)] / py[0])
    c["q1.c"] = str(all(joint[(x, y)] == px[x] * py[y] for x, y in joint)).lower()
    exy = sum(x * y * p for (x, y), p in joint.items())
    ex = sum(x * p for x, p in px.items())
    ey = sum(y * p for y, p in py.items())
    c["q2.a"] = fr(exy)
    c["q2.b"] = fr(exy - ex * ey)

    # q3: write X = mx + U, Y = my + W with E[U] = E[W] = 0, E[U^2] = vx,
    # E[W^2] = vy, E[UW] = c; Var(aX + bY) = E[(aU + bW)^2], expanded and
    # each monomial replaced by its moment.
    a, b, vx, vy, cc, U, W = sp.symbols("a b vx vy c U W", real=True)
    var = sp.expand((a * U + b * W) ** 2).subs({U**2: vx, W**2: vy}).subs(U * W, cc)
    assert not var.has(U) and not var.has(W)
    c["q3"] = str(var)

    xs = [-1, 0, 1]
    exy4 = sum(F(1, 3) * x * x * x for x in xs)
    ex4 = sum(F(1, 3) * x for x in xs)
    ey4 = sum(F(1, 3) * x * x for x in xs)
    c["q4.a"] = fr(exy4 - ex4 * ey4)
    # independence fails if any joint cell differs from the product of marginals
    pj = {(x, x * x): F(1, 3) for x in xs}
    pX = {x: F(1, 3) for x in xs}
    pY = {0: F(1, 3), 1: F(2, 3)}
    c["q4.b"] = str(
        all(pj.get((x, y), F(0)) == pX[x] * pY[y] for x in xs for y in pY)
    ).lower()

    counts = {"the": 4, "cat": 3, "sat": 2, "mat": 1, "dog": 0}
    n_tok = sum(counts.values())
    c["q5.a"] = fr(F(counts["cat"], n_tok))
    c["q5.b"] = fr(F(counts["dog"] + 1, n_tok + len(counts)))

    k, n, p = sp.symbols("k n p", positive=True)
    lik = p**k * (1 - p) ** (n - k)
    ll = sp.expand_log(sp.log(lik), force=True)
    c["q6.a"] = str(ll)
    sol = sp.solve(sp.Eq(sp.diff(ll, p), 0), p)
    assert len(sol) == 1
    c["q6.b"] = str(sp.simplify(sol[0]))

    lam, s = sp.symbols("lam s", positive=True)
    ll7 = n * sp.log(lam) - lam * s  # sum of log(lam) - lam x_i, with sum x_i = s
    sol7 = sp.solve(sp.Eq(sp.diff(ll7, lam), 0), lam)
    assert len(sol7) == 1
    c["q7"] = str(sol7[0])

    P = {"a": F(1, 4), "b": F(1, 4), "ab": F(1, 2)}
    segs = [["a", "b"], ["ab"]]
    weights = []
    for seg in segs:
        w = F(1)
        for piece in seg:
            w *= P[piece]
        weights.append(w)
    z = sum(weights)
    exp_count = {piece: F(0) for piece in P}
    for seg, w in zip(segs, weights):
        for piece in seg:
            exp_count[piece] += w / z
    total = sum(exp_count.values())
    c["q8.a"] = fr(exp_count["a"])
    c["q8.b"] = fr(exp_count["ab"] / total)
    return confirm("S-M07b", c)


if __name__ == "__main__":
    sys.exit(main())
