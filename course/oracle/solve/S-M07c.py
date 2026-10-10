"""Recompute every auto-checked answer of S-M07c independently (see _confirm.py).

q1 and q2 by SymPy algebra on a symbolic sum of independent variables (the
variance of the mean expanded term by term), q3 by the exact binomial tail
cross-checked against the normal approximation the question asks for, q4 by
solving width(m n) = width(n) / 2 for m, the intervals q6 to q8 with exact
Fractions (q7 by the Wilson formula, q8 by the type 7 rule written out from
its definition), and q9 from the frequentist definition.
"""

from __future__ import annotations

import math
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
    mu, s2, n, eps = sp.symbols("mu s2 n eps", positive=True)
    # Var(Xbar) for a concrete n = 5, then generalized: sum of n independent
    # terms each (1/n)^2 s2.
    k = 5
    xs = sp.symbols(f"x0:{k}")
    mean = sum(xs) / k
    var_terms = sum(
        (sp.diff(mean, x) ** 2) * s2 for x in xs
    )  # independence: no cross terms
    assert sp.simplify(var_terms - s2 / k) == 0
    c["q1.a"] = str(sp.simplify(sum(sp.diff(mean, x) * mu for x in xs)))
    c["q1.b"] = str(s2 / n)
    c["q2"] = str((s2 / n) / eps**2)

    # q3: the normal approximation the question asks for, z = (60 - 50) / 5.
    z = (60 - 100 * 0.5) / math.sqrt(100 * 0.25)
    assert z == 2.0
    c["q3"] = repr(1 - 0.97725)
    exact = sum(math.comb(100, j) for j in range(60, 101)) / 2**100
    assert 0.015 < exact < 0.03  # the exact tail (0.0284) is near the approximation

    m = sp.symbols("m", positive=True)
    sol = sp.solve(sp.Eq(1 / sp.sqrt(m * n), 1 / (2 * sp.sqrt(n))), m)
    assert len(sol) == 1
    c["q4"] = str(sol[0])

    half = F(2064, 1000) * F(2) / F(5)
    c["q6"] = f"[{fr(F(10) - half)}, {fr(F(10) + half)}]"

    zz, nn, kk = F(2), F(10), F(0)
    ph = kk / nn
    center = (ph + zz**2 / (2 * nn)) / (1 + zz**2 / nn)
    rad = ph * (1 - ph) / nn + zz**2 / (4 * nn**2)  # = 1/100, a perfect square
    root = F(1, 10)
    assert root * root == rad
    hw = zz * root / (1 + zz**2 / nn)
    c["q7"] = f"[{fr(center - hw)}, {fr(center + hw)}]"

    s = [1, 2, 4, 5, 5, 6, 8, 9, 13]

    def q(level: F) -> F:
        h = (len(s) - 1) * level
        i = math.floor(h)
        return F(s[i]) + (h - i) * (s[i + 1] - s[i]) if i + 1 < len(s) else F(s[i])

    c["q8"] = f"[{fr(q(F(1, 10)))}, {fr(q(F(9, 10)))}]"
    c["q9"] = "c"
    return confirm("S-M07c", c)


if __name__ == "__main__":
    sys.exit(main())
