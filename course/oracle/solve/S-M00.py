"""Recompute every auto-checked answer of S-M00 independently (see _confirm.py).

Inverses by SymPy's solve on y = f(x), identities and limits by SymPy,
roots by polynomial root finding, series by explicit summation in exact
rationals, rotations by matrix products, and inequalities by SymPy's
solveset over the reals (or the stated domain).

    uv run --project course/harness python course/oracle/solve/S-M00.py
"""

from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

x, y, z, t, a, b = sp.symbols("x y z t a b", real=True)
xp, yp, p, k = sp.symbols("x y p k", positive=True)
R = sp.S.Reals


def inverse(expr, var, domain=R, name=None) -> str:
    """The unique branch of var = f^{-1}(w) solving w = expr, as a string in
    `name` (default: var's name)."""
    w = sp.Symbol("w", positive=domain is not R, real=True)
    sols = [s for s in sp.solve(sp.Eq(w, expr), var) if s.is_real is not False]
    assert len(sols) == 1, sols
    return str(sols[0].subs(w, sp.Symbol(name or var.name)))


def interval(s) -> str:
    """A SymPy set as the answer syntax: (a, b] pieces joined by U."""
    parts = s.args if isinstance(s, sp.Union) else (s,)
    out = []
    for iv in parts:
        lo = "-oo" if iv.start == -sp.oo else str(iv.start)
        hi = "oo" if iv.end == sp.oo else str(iv.end)
        out.append(
            ("(" if iv.left_open else "[")
            + f"{lo}, {hi}"
            + (")" if iv.right_open else "]")
        )
    return " U ".join(out)


def fr(v: F) -> str:
    return f"{v.numerator}/{v.denominator}"


def main() -> int:
    c: dict[str, str] = {}

    # ---- functions and inverses
    c["q1"] = inverse(3 * x - 7, x)
    c["q2"] = inverse(sp.exp(2 * x + 1), x, domain="positive")
    c["q3"] = inverse((2 * x + 1) / (x - 3), x)
    c["q4"] = str((x**2).subs(x, x + 1))
    c["q5"] = inverse(1 / (1 + sp.exp(-x)), x, domain="positive", name="p")
    sig = 1 / (1 + sp.exp(-x))
    c["q6"] = interval(
        sp.Interval.open(sp.limit(sig, x, -sp.oo), sp.limit(sig, x, sp.oo))
    )
    c["q7"] = interval(sp.solveset(4 - x**2 > 0, x, R))
    deriv = sp.diff(x**3 + x, x)
    c["q8"] = str(bool(sp.solveset(deriv <= 0, x, R) == sp.S.EmptySet)).lower()
    f9 = lambda v: 2**v  # noqa: E731
    g9 = lambda v: sp.log(v, 2)  # noqa: E731
    c["q9"] = str(sp.simplify(f9(g9(16)) + g9(f9(3))))
    c["q10"] = inverse(sp.log(1 + sp.exp(x)), x, domain="positive", name="y")

    # ---- exp / log
    c["q11"] = str(sp.simplify(sp.log(8, 2) + sp.log(4, 2)))
    c["q12"] = str(sp.nsimplify(sp.simplify(sp.log(32) / sp.log(8))))
    c["q13"] = str(sp.simplify(sp.exp(3 * sp.log(xp))))
    c["q14"] = str(sp.solve(sp.Eq(2**x, 1000), x)[0])
    c["q15"] = str(1 / sp.log(2))
    nll_tok, n_tok, n_bytes = 2, 1000, 4000
    c["q16"] = str(sp.Integer(nll_tok * n_tok) / (n_bytes * sp.log(2)))
    c["q17"] = str(sp.simplify(sp.log(8) / sp.log(2)))
    # log of a product of prime powers is the sum of exponent * log prime: 72 = 2^3 3^2
    c["q18"] = str(sum(e * {2: a, 3: b}[q] for q, e in sp.factorint(72).items()))
    sols = [s for s in sp.solve(sp.Eq(x * (x - 8), 3**2), x) if s > 8]
    c["q19"] = str(sols[0])
    c["q20"] = str(sp.solve(sp.Eq(4**x, 8), x)[0])
    c["q21"] = str(sp.solve(sp.Eq(sp.exp(-k * t), sp.Rational(1, 2)), t)[0])
    c["q22"] = str(len(str(2**100)))

    # ---- trig, complex, Euler
    c["q23"] = str(sp.sin(sp.pi / 6))
    c["q24"] = str(sp.cos(2 * sp.pi / 3))
    c["q25"] = str(sp.trigsimp(sp.cos(a) * sp.cos(b) - sp.sin(a) * sp.sin(b)))
    rot = lambda th: sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])  # noqa: E731
    v = rot(sp.pi / 3) * sp.Matrix([1, 0])
    c["q26"] = f"[{v[0]}, {v[1]}]"
    m = rot(sp.pi / 2) * rot(sp.pi / 3)
    c["q27"] = str(sp.atan2(m[1, 0], m[0, 0]) % (2 * sp.pi))
    c["q28"] = str(sp.expand((1 + sp.I) ** 8))
    c["q29"] = str(sp.Abs(3 + 4 * sp.I))
    c["q30"] = str(sp.expand((3 + 4 * sp.I) * (sp.Rational(3, 5) + 4 * sp.I / 5)))
    c["q31"] = str(sp.exp(sp.I * sp.pi / 2))
    c["q32"] = str(sp.cos(t).rewrite(sp.exp))
    c["q33"] = str(
        sp.simplify(sp.re(sp.exp(sp.I * a) * sp.conjugate(sp.exp(sp.I * b))))
    )
    c["q34"] = str(2 * sp.pi / sp.Rational(1, 100))

    # ---- sequences and series (exact sums)
    c["q35"] = str(
        sp.summation(
            sp.Rational(1, 2) ** sp.Symbol("j", integer=True),
            (sp.Symbol("j", integer=True), 0, sp.oo),
        )
    )
    rr, nn, jj = sp.symbols("r n j")
    c["q36"] = str(sp.simplify(sp.summation(rr**jj, (jj, 0, nn - 1)).args[1][0]))
    c["q37"] = str(sum(3 * 2**i for i in range(10)))
    w = [sp.Integer(10000) ** sp.Rational(-2 * i, 128) for i in range(64)]
    c["q38"] = str(sp.nsimplify(w[1] / w[0]))
    c["q39"] = str(sum(range(1, 101)))
    beta, mt, weights = F(9, 10), F(0), []
    for _ in range(3):
        mt = beta * mt + (1 - beta) * 1
    c["q40"] = fr(mt)
    weights = [(1 - beta) * beta**j for j in range(3)]  # newest first
    c["q41"] = fr(weights[-1])
    n_ = sp.Symbol("n", positive=True)
    c["q42"] = str(sp.limit((2 * n_ + 1) / (n_ + 3), n_, sp.oo))
    c["q43"] = fr(sum(F(27, 100**i) for i in range(1, 40)).limit_denominator(1000))
    c["q44"] = fr(sum(F(1, 2 ** (2 * i)) for i in range(1, 5)))

    # ---- polynomials
    coeffs = [2, -3, 1, 4]
    acc = 0
    for cf in reversed(coeffs):  # Horner, independently of the key
        acc = acc * 2 + cf
    c["q45"] = str(acc)
    q, r = sp.div(x**3 - 6 * x**2 + 11 * x - 6, x - 1)
    assert r == 0
    c["q46"] = str(q)
    c["q47"] = (
        "{" + ", ".join(str(s) for s in sp.roots(x**3 - 6 * x**2 + 11 * x - 6)) + "}"
    )
    c["q48"] = "{" + ", ".join(str(s) for s in sp.roots(2 * x**2 + 3 * x - 2)) + "}"
    c["q49"] = str(min(sp.solve(x**2 - 10**6 * x + 1, x), key=lambda s: float(s)))
    r50 = sp.solve(3 * x**2 - 7 * x + 2, x)
    c["q50"] = str(r50[0] * r50[1])
    c["q51"] = str(len([s for s in sp.solve(x**2 + x + 1, x) if s.is_real]))
    taylor = sp.series(sp.exp(x), x, 0, 4).removeO()
    c["q52"] = str([taylor.coeff(x, i) for i in range(4)])

    # ---- inequalities
    c["q53"] = interval(sp.solveset(sp.Abs(x - 3) < 2, x, R))
    c["q54"] = interval(sp.solveset(x**2 - 5 * x + 6 > 0, x, R))
    c["q55"] = str(next(n for n in range(64) if 2**n > 1000))
    c["q56"] = interval(sp.solveset((x - 1) / (x + 2) >= 0, x, R))
    c["q57"] = interval(sp.solveset(-sp.log(x, 2) > 3, x, sp.Interval.open(0, 1)))
    c["q58"] = interval(sp.solveset(sp.exp(x) <= 2, x, R))
    c["q59"] = str(sp.minimum(x + 1 / x, x, sp.Interval.open(0, sp.oo)))
    # ln x < x - 1 fails only where g(x) = x - 1 - ln x reaches its minimum 0.
    g = x - 1 - sp.log(x)
    crit = sp.solve(sp.diff(g, x), x)
    assert crit == [1] and g.subs(x, 1) == 0 and sp.diff(g, x, 2).subs(x, 1) > 0
    c["q60"] = interval(sp.Interval.open(0, sp.oo) - sp.FiniteSet(*crit))
    return confirm("S-M00", c)


if __name__ == "__main__":
    sys.exit(main())
