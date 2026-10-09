"""Recompute every auto-checked answer of S-M07a independently (see _confirm.py).

Finite probabilities by exact enumeration of sample spaces with Fractions,
moments of named distributions by SymPy sums and integrals over their
densities, the inverse CDF by the sampling.md walk in exact arithmetic, and
Box-Muller by direct evaluation of the spec formula.
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction as F
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def fr(x: F) -> str:
    return f"{x.numerator}/{x.denominator}"


def main() -> int:
    c: dict[str, str] = {}

    # q1: build a finite space realizing the given probabilities (20 equally likely points)
    omega = range(20)
    A = set(range(10))  # P(A) = 10/20
    B = set(range(6, 14))  # P(B) = 8/20, A n B = {6..9}: 4/20
    P = lambda E: F(len(E), 20)  # noqa: E731
    assert (P(A), P(B), P(A & B)) == (F(1, 2), F(2, 5), F(1, 5))
    c["q1.a"] = fr(P(A | B))
    c["q1.b"] = fr(P(A & B) / P(B))
    c["q1.c"] = str(P(A & B) == P(A) * P(B)).lower()

    dice = list(itertools.product(range(1, 7), repeat=2))
    seven = {d for d in dice if sum(d) == 7}
    six = {d for d in dice if 6 in d}
    c["q2.a"] = fr(F(len(seven), 36))
    c["q2.b"] = fr(F(len(six), 36))
    c["q2.c"] = fr(F(len(seven & six), len(six)))

    # q3: a population of 2000 requests in exact proportions
    bad, ok = 20, 1980
    flag_bad, flag_ok = F(9, 10) * bad, F(1, 20) * ok
    c["q3.a"] = fr((flag_bad + flag_ok) / 2000)
    c["q3.b"] = fr(flag_bad / (flag_bad + flag_ok))

    distinct = sum(
        1 for t in itertools.product(range(256), repeat=3) if len(set(t)) == 3
    )
    c["q4"] = fr(F(distinct, 256**3))

    coins = {"fair": F(1, 2), "biased": F(3, 4)}
    ph = sum(F(1, 2) * h for h in coins.values())
    c["q5.a"] = fr(ph)
    c["q5.b"] = fr(F(1, 2) * coins["biased"] / ph)

    die = [F(k) for k in range(1, 7)]
    ex = sum(die) / 6
    ex2 = sum(k * k for k in die) / 6
    c["q7.a"] = fr(ex)
    c["q7.b"] = fr(ex2)
    c["q7.c"] = fr(sum((k - ex) ** 2 for k in die) / 6)

    p, n, lam, u, x, a, b = sp.symbols("p n lam u x a b", positive=True)
    k = sp.symbols("k", integer=True, nonnegative=True)
    bern = {0: 1 - p, 1: p}
    mu = sum(v * q for v, q in bern.items())
    c["q8.a"] = str(sp.factor(sum((v - mu) ** 2 * q for v, q in bern.items())))
    # Binomial variance by summing over its pmf for symbolic n is slow; check n = 1..8
    var_bin = []
    for nn in range(1, 9):
        pmf = {
            j: sp.binomial(nn, j) * p**j * (1 - p) ** (nn - j) for j in range(nn + 1)
        }
        m1 = sum(j * q for j, q in pmf.items())
        var_bin.append(
            sp.factor(sp.expand(sum(j * j * q for j, q in pmf.items()) - m1**2))
        )
    assert all(
        sp.simplify(v - nn * p * (1 - p)) == 0 for nn, v in enumerate(var_bin, 1)
    )
    c["q8.b"] = "n*p*(1 - p)"
    kk = sp.symbols("kk", integer=True, positive=True)
    geo = sp.summation(kk * p * (1 - p) ** (kk - 1), (kk, 1, sp.oo))
    geo = geo.replace(lambda e: isinstance(e, sp.Piecewise), lambda e: e.args[0][0])
    c["q9"] = str(sp.simplify(geo))  # the convergent branch, |1 - p| < 1

    q = [F(1, 10), F(2, 10), F(3, 10), F(4, 10)]
    c["q10.a"] = fr(sum(i * qi for i, qi in enumerate(q)))

    def inv_cdf(uu: F) -> int:
        tot = F(0)
        for i, qi in enumerate(q):
            tot += qi
            if uu < tot:
                return i
        return len(q) - 1

    c["q10.b"] = str(inv_cdf(F(35, 100)))
    c["q10.c"] = str(inv_cdf(F(3, 10)))
    c["q11"] = str(sp.exp(-lam) * lam**0 / sp.factorial(0))

    uu = sp.symbols("uu", positive=True)
    c["q12.a"] = str(sp.integrate(uu, (uu, 0, 1)))
    c["q12.b"] = str(
        sp.integrate(uu**2, (uu, 0, 1)) - sp.integrate(uu, (uu, 0, 1)) ** 2
    )
    sol = sp.solve(sp.Eq(1 - sp.exp(-lam * x), u), x)
    assert len(sol) == 1
    c["q13.a"] = str(sol[0])
    c["q13.b"] = str(sp.integrate(x * lam * sp.exp(-lam * x), (x, 0, sp.oo)))
    z = sp.symbols("z", real=True)
    phi = sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi)
    ar, br = sp.symbols("a b", real=True)
    m_ = sp.integrate((ar * z + br) * phi, (z, -sp.oo, sp.oo))
    v_ = sp.integrate((ar * z + br - m_) ** 2 * phi, (z, -sp.oo, sp.oo))
    c["q14"] = str(sp.simplify(v_))

    one_minus_u1, u2 = sp.exp(-2), sp.Rational(1, 12)
    r = sp.sqrt(-2 * sp.log(one_minus_u1))
    c["q15.a"] = str(sp.nsimplify(r * sp.cos(2 * sp.pi * u2)))
    c["q15.b"] = str(sp.nsimplify(r * sp.sin(2 * sp.pi * u2)))
    c["q16"] = str(sp.integrate(sp.exp(-(z**2) / 2), (z, -sp.oo, sp.oo)))
    return confirm("S-M07a", c)


if __name__ == "__main__":
    sys.exit(main())
