"""Recompute every auto-checked answer of S-M07d independently (see _confirm.py).

q1 and q2 by enumerating every sign vector and every split, q3 by summing the
binomial tail with exact Fractions, q4 by running Holm's step-down loop, q5
by solving the normal equations with SymPy, q6 from the log-odds difference,
q7 by SymPy differentiation of the actual loss, q8 by counting pairs, q10 to
q13 from the M/M/1 formulas with Fractions (W from Little's law, cross-checked
against 1 / (mu - lambda)), q14 and q16 by linearity of expectation checked
against brute-force enumeration for small n and m, q15 as the SymPy limit,
and q17 from Azar, Broder, Karlin, and Upfal (1994).
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction as F
from math import comb
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def fr(x: F) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def main() -> int:
    c: dict[str, str] = {}

    d = [2, 1, 3]
    t = abs(sum(d))
    hits = sum(
        abs(sum(s * v for s, v in zip(signs, d))) >= t
        for signs in itertools.product((1, -1), repeat=3)
    )
    c["q1"] = fr(F(hits, 8))

    z = [3, 5, 1, 2]
    obs = abs(F(3 + 5, 2) - F(1 + 2, 2))
    splits = list(itertools.combinations(range(4), 2))
    hits = 0
    for sp_ in splits:
        a = [z[i] for i in sp_]
        b = [z[i] for i in range(4) if i not in sp_]
        hits += abs(F(sum(a), 2) - F(sum(b), 2)) >= obs
    c["q2"] = fr(F(hits, len(splits)))

    n, k = 12, 2
    c["q3"] = fr(min(F(1), 2 * F(sum(comb(n, i) for i in range(k + 1)), 2**n)))

    p = [F(3, 100), F(2, 1000), F(4, 100), F(12, 1000), F(3, 10)]
    order = sorted(range(5), key=lambda i: p[i])
    rej = []
    for r, i in enumerate(order):
        if p[i] > F(5, 100) / (5 - r):
            break
        rej.append(i + 1)
    c["q4"] = "{" + ", ".join(str(i) for i in sorted(rej)) + "}"

    b0, b1 = sp.symbols("b0 b1")
    pts = [(0, 1), (1, 3), (2, 4), (3, 8)]
    sse = sum((y - b0 - b1 * x) ** 2 for x, y in pts)
    sol = sp.solve([sp.diff(sse, b0), sp.diff(sse, b1)], [b0, b1])
    c["q5.a"] = str(sol[b1])
    c["q5.b"] = str(sol[b0])

    x = sp.symbols("x")
    logit = -2 + sp.Rational(1, 2) * x
    c["q6"] = str(sp.simplify(sp.exp(logit.subs(x, x + 2)) / sp.exp(logit)))

    w, xx, y = sp.symbols("w x y", real=True)
    pp = 1 / (1 + sp.exp(-w * xx))
    loss = -(y * sp.log(pp) + (1 - y) * sp.log(1 - pp))
    c["q7"] = str(sp.simplify(sp.diff(loss, w))).replace("**", "^")

    pos, neg = [F(8, 10), F(5, 10), F(35, 100)], [F(6, 10), F(35, 100), F(1, 10)]
    score = sum(
        F(1) if a > b else F(1, 2) if a == b else F(0) for a in pos for b in neg
    )
    c["q8"] = fr(score / 9)

    c["q10"] = fr(F(40) * F(1, 4))
    lam, mu = F(8), F(10)
    rho = lam / mu
    L = rho / (1 - rho)
    W = L / lam
    assert W == 1 / (mu - lam)
    c["q11.a"], c["q11.b"], c["q11.c"] = fr(rho), fr(L), fr(W)
    c["q12"] = "1/(mu - lam)"

    def w_at(r):
        return 1 / (1 - r)  # times 1/mu, which cancels

    c["q13"] = fr(w_at(F(9, 10)) / w_at(F(1, 2)))

    def empty_bins_brute(n):
        tot = F(0)
        for assign in itertools.product(range(n), repeat=n):
            tot += n - len(set(assign))
        return tot / n**n

    for nn in (2, 3, 4):
        assert empty_bins_brute(nn) == nn * (1 - F(1, nn)) ** nn
    c["q14"] = "n*(1 - 1/n)^n"
    nsym = sp.symbols("n", positive=True)
    c["q15"] = str(sp.limit(nsym * (1 - 1 / nsym) ** nsym / nsym, nsym, sp.oo))

    def pairs_brute(m, n):
        tot = F(0)
        for assign in itertools.product(range(n), repeat=m):
            tot += sum(
                assign[i] == assign[j] for i in range(m) for j in range(i + 1, m)
            )
        return tot / n**m

    for m, nn in ((2, 3), (3, 2), (4, 3)):
        assert pairs_brute(m, nn) == F(m * (m - 1), 2 * nn)
    c["q16"] = "m*(m - 1)/(2*n)"
    c["q17"] = "b"
    return confirm("S-M07d", c)


if __name__ == "__main__":
    sys.exit(main())
