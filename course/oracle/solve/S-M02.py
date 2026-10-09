"""Recompute every auto-checked answer of S-M02 independently (see _confirm.py).

Definite and improper integrals by SymPy's `integrate`, series sums by
`summation`, convergence by SymPy's `is_convergent` (and the ratio-test limit),
Taylor polynomials by `series`, arc lengths and polar areas from their integral
formulas, ODEs by `dsolve` with the stated initial value, and Euler's method by
stepping in exact rationals.

    uv run --project course/harness python course/oracle/solve/S-M02.py
"""

from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

x, t, th, z = sp.symbols("x t theta z", real=True)
n = sp.Symbol("n", integer=True, positive=True)


def interval(s) -> str:
    lo = "-oo" if s.start == -sp.oo else str(s.start)
    hi = "oo" if s.end == sp.oo else str(s.end)
    return ("(" if s.left_open else "[") + f"{lo}, {hi}" + (")" if s.right_open else "]")


def taylor(expr, deg) -> str:
    return str(sp.series(expr, x, 0, deg + 1).removeO())


def main() -> int:
    c: dict[str, str] = {}

    # ---- integration techniques
    c["q1"] = str(sp.integrate(x * sp.exp(x), (x, 0, 1)))
    c["q2"] = str(sp.integrate(sp.log(x), (x, 1, sp.E)))
    c["q3"] = str(sp.integrate(x * sp.cos(x), (x, 0, sp.pi / 2)))
    c["q4"] = str(sp.integrate(x / (1 + x**2), (x, 0, 1)))
    c["q5"] = str(sp.integrate(sp.sin(x) ** 2, (x, 0, sp.pi / 2)))
    c["q6"] = str(sp.simplify(sp.integrate(sp.apart(1 / (x**2 - 1)), (x, 2, 3))))
    c["q7"] = str(sp.integrate(t**2 * sp.exp(t), (t, 0, x)))
    c["q8"] = str(sp.integrate(x * sp.sqrt(1 - x**2), (x, 0, 1)))
    c["q9"] = str(sp.integrate(x * sp.sin(x), (x, 0, sp.pi)))
    c["q10"] = str(sp.integrate(x**3 * sp.exp(x**2), (x, 0, 1)))
    c["q11"] = str(sp.integrate(1 / sp.sqrt(4 - x**2), (x, 0, 1)))

    # ---- improper integrals and the Gaussian
    c["q13"] = str(sp.integrate(1 / x**2, (x, 1, sp.oo)))
    c["q14"] = str(sp.integrate(sp.exp(-2 * x), (x, 0, sp.oo)))
    c["q15"] = str(sp.integrate(1 / sp.sqrt(x), (x, 0, 1)))
    c["q16"] = str(sp.integrate(sp.exp(-(x**2)), (x, -sp.oo, sp.oo)))
    c["q17"] = str(sp.integrate(z**2 * sp.exp(-(z**2) / 2) / sp.sqrt(2 * sp.pi), (z, 0, sp.oo)))

    # ---- series and convergence
    c["q19"] = str(bool(sp.Sum(1 / n**2, (n, 1, sp.oo)).is_convergent())).lower()
    c["q20"] = str(bool(sp.Sum(1 / n, (n, 1, sp.oo)).is_convergent())).lower()
    m = sp.Symbol("m", integer=True, nonnegative=True)
    c["q21"] = str(sp.summation(sp.Rational(2, 3) ** m, (m, 0, sp.oo)))
    c["q22"] = str(sp.summation(1 / (n * (n + 1)), (n, 1, sp.oo)))
    c["q23"] = str(sp.summation(n / 2**n, (n, 1, sp.oo)))
    ratio = sp.limit(sp.combsimp((sp.factorial(n + 1) / (n + 1) ** (n + 1)) / (sp.factorial(n) / n**n)), n, sp.oo)
    c["q24"] = str(bool(ratio < 1)).lower()
    c["q25"] = str(bool(sp.Sum((-1) ** (n + 1) / n, (n, 1, sp.oo)).is_convergent())).lower()
    c["q26"] = str(sp.summation((-1) ** (n + 1) / n, (n, 1, sp.oo)))
    an = 1 / n
    radius = sp.limit(an / an.subs(n, n + 1), n, sp.oo)
    c["q27"] = str(radius)
    left = sp.Sum((-1) ** n / n, (n, 1, sp.oo)).is_convergent()
    right = sp.Sum(1 / n, (n, 1, sp.oo)).is_convergent()
    c["q28"] = interval(sp.Interval(-radius, radius, left_open=not left, right_open=not right))
    # integral test: sum 1/n^p converges exactly when the integral of x^-p over [1, oo) does
    p, xpos = sp.Symbol("p", real=True), sp.Symbol("xpos", positive=True)
    tail = sp.integrate(xpos ** (-p), (xpos, 1, sp.oo), conds="piecewise")
    c["q29"] = interval(tail.args[0].cond.as_set())
    conv = [q for q in (sp.Rational(1, 2), 1, sp.Rational(3, 2), 2, 5) if sp.Sum(1 / n**q, (n, 1, sp.oo)).is_convergent()]
    assert conv == [sp.Rational(3, 2), 2, 5]  # spot checks of the same boundary

    # ---- Taylor series and remainders
    c["q31"] = taylor(sp.exp(x), 3)
    c["q32"] = taylor(sp.log(1 + x), 3)
    c["q33"] = str(sp.series(sp.sin(x), x, 0, 6).removeO().coeff(x, 5))
    c["q34"] = taylor(sp.cos(x), 4)
    c["q35"] = str(sp.exp(sp.Rational(1, 2)) * sp.Rational(1, 2) ** 4 / sp.factorial(4))
    c["q36"] = str(next(k for k in range(1, 30) if F(1, int(sp.factorial(k + 1))) < F(1, 10**6)))
    c["q37"] = taylor(1 / (1 - x), 3)
    k = int(sp.Integer(round(5 / float(sp.log(2)))))
    c["q38"] = str(5 - k * sp.log(2))
    c["q39"] = taylor(sp.erf(x), 5)

    # ---- parametric and polar
    X, Y = t**2, t**3
    c["q41"] = str((sp.diff(Y, t) / sp.diff(X, t)).subs(t, 1))
    c["q42"] = str(sp.integrate(sp.sqrt(sp.diff(sp.cos(t), t) ** 2 + sp.diff(sp.sin(t), t) ** 2), (t, 0, sp.pi / 2)))
    tp = sp.Symbol("tp", nonnegative=True)
    c["q43"] = str(sp.simplify(sp.integrate(sp.sqrt(sp.diff(tp**2, tp) ** 2 + sp.diff(tp**3, tp) ** 2), (tp, 0, 1))))
    c["q44"] = str(sp.integrate((2 * sp.cos(th)) ** 2 / 2, (th, -sp.pi / 2, sp.pi / 2)))
    xs, ys, r = sp.symbols("x y r", real=True)
    # r = 2 sin(theta): multiply by r, then r^2 = x^2 + y^2 and r sin(theta) = y
    polar = sp.Eq(r**2, 2 * r * sp.sin(th)).subs({r**2: xs**2 + ys**2}).subs({r * sp.sin(th): ys})
    c["q45"] = f"{polar.lhs} = {polar.rhs}"
    c["q46"] = str(sp.integrate((1 + sp.cos(th)) ** 2 / 2, (th, 0, 2 * sp.pi)))

    # ---- ODEs
    yf = sp.Function("y")
    c["q47"] = str(sp.dsolve(sp.Eq(yf(t).diff(t), -2 * yf(t)), yf(t), ics={yf(0): 3}).rhs)
    c["q48"] = str(sp.dsolve(sp.Eq(yf(t).diff(t), yf(t) * (1 - yf(t))), yf(t), ics={yf(0): sp.Rational(1, 2)}).rhs)
    c["q49"] = str(sp.dsolve(sp.Eq(yf(t).diff(t) + yf(t), t), yf(t), ics={yf(0): 0}).rhs)
    kk = sp.Symbol("k", positive=True)
    c["q50"] = str(sp.solve(sp.Eq(sp.exp(-kk * 10), sp.Rational(1, 2)), kk)[0])
    yv, h = F(1), F(1, 2)
    for _ in range(2):
        yv = yv + h * yv
    c["q51"] = f"{yv.numerator}/{yv.denominator}"
    sol = sp.dsolve(sp.Eq(yf(t).diff(t), -yf(t)), yf(t), ics={yf(0): 4}).rhs
    c["q52"] = str(sp.solve(sp.Eq(sol, 1), t)[0])
    return confirm("S-M02", c)


if __name__ == "__main__":
    sys.exit(main())
