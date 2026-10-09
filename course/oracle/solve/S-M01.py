"""Recompute every auto-checked answer of S-M01 independently (see _confirm.py).

Limits by SymPy's `limit` on the stated function, derivatives by SymPy's
`diff` of the stated function (and implicit derivatives by `idiff`), rates
by differentiating the stated relation in t, optima by solving f' = 0 and
comparing candidates, integrals by SymPy's `integrate`, and the Riemann sum
by exact summation.

    uv run --project course/harness python course/oracle/solve/S-M01.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

x, y, z, h, s, c, t = sp.symbols("x y z h s c t", real=True)
xp = sp.Symbol("x", positive=True)


def lim(expr, var, at, dirn="+-"):
    return str(sp.limit(expr, var, at, dirn) if dirn != "+-" else sp.limit(expr, var, at))


def d(expr, var=x, n=1) -> str:
    return str(sp.diff(expr, var, n))


def main() -> int:
    c_: dict[str, str] = {}

    # ---- limits
    c_["q1"] = lim((x**2 - 4) / (x - 2), x, 2)
    c_["q2"] = lim(sp.sin(3 * x) / x, x, 0)
    c_["q3"] = lim((3 * x**2 + 2 * x) / (5 * x**2 - 1), x, sp.oo)
    c_["q4"] = lim(((1 + h) ** 3 - 1) / h, h, 0)
    c_["q5"] = lim((1 - sp.cos(x)) / x**2, x, 0)
    c_["q6"] = lim((1 + 2 / x) ** x, x, sp.oo)
    c_["q7"] = lim(x * sp.log(x), x, 0, "+")
    c_["q8"] = lim(sp.sqrt(x**2 + x) - x, x, sp.oo)
    c_["q9"] = lim(sp.Abs(x) / x, x, 0, "-")

    # ---- derivative rules
    c_["q11"] = d(x**3 * sp.sin(x))
    c_["q12"] = d(sp.exp(x) / (1 + x**2))
    c_["q13"] = d(1 / (1 + sp.exp(-z)), z)
    c_["q14"] = d(sp.log(1 + sp.exp(x)))
    c_["q15"] = d((sp.exp(x) - sp.exp(-x)) / (sp.exp(x) + sp.exp(-x)))
    c_["q16"] = d(sp.sin(x**2))
    c_["q17"] = d(sp.sqrt(1 + x**4))
    c_["q18"] = d(sp.log(sp.cos(x)))
    c_["q19"] = str(sp.diff(xp**xp, xp))
    c_["q20"] = d(sp.exp(-(x**2) / 2))
    c_["q21"] = d(x / (1 + sp.exp(-x)))
    c_["q22"] = d(x * sp.exp(-x), x, 2)
    c_["q23"] = str(sp.diff((x**2 + 1) ** 5, x).subs(x, 1))
    c_["q24"] = str(sp.diff(sp.log(xp, 2), xp))

    # ---- implicit differentiation and related rates
    c_["q26"] = str(sp.idiff(x**2 + y**2 - 25, y, x).subs({x: 3, y: 4}))
    c_["q27"] = str(sp.idiff(x * y + y**3 - 2, y, x))
    r = sp.Function("r")(t)
    dA = sp.diff(sp.pi * r**2, t).subs(sp.Derivative(r, t), 2).subs(r, 5)
    c_["q28"] = str(dA)
    X, Y = sp.Function("X")(t), sp.Function("Y")(t)
    rel = sp.diff(X**2 + Y**2 - 100, t)  # 2 X X' + 2 Y Y' = 0
    ydot = sp.solve(rel, sp.Derivative(Y, t))[0]
    c_["q29"] = str(ydot.subs(sp.Derivative(X, t), 1).subs({X: 6, Y: 8}))
    dV = sp.diff(sp.Rational(4, 3) * sp.pi * r**3, t)
    rdot = sp.solve(sp.Eq(dV, 100), sp.Derivative(r, t))[0]
    c_["q30"] = str(rdot.subs(r, 5))
    yy = sp.Function("yy")(xp)
    c_["q31"] = str(sp.solve(sp.diff(sp.exp(yy) - xp, xp), sp.Derivative(yy, xp))[0].subs(yy, sp.log(xp)))

    # ---- optimization
    f = x**2 - 6 * x + 11
    (xm,) = sp.solve(sp.diff(f, x), x)
    c_["q32"] = str(xm)
    c_["q33"] = str(f.subs(x, xm))
    w = sp.Symbol("w", positive=True)
    area = w * (10 - w)
    c_["q34"] = str(max(area.subs(w, v) for v in sp.solve(sp.diff(area, w), w)))
    c_["q35"] = "{" + ", ".join(str(v) for v in sp.solve(sp.diff(x**3 - 3 * x, x), x)) + "}"
    g = x * sp.exp(-x)
    cands = [g.subs(x, v) for v in sp.solve(sp.diff(g, x), x)] + [g.subs(x, 0), sp.limit(g, x, sp.oo)]
    c_["q36"] = str(max(cands, key=lambda e: float(e)))
    L = (c - 1) ** 2 + (c - 2) ** 2 + (c - 6) ** 2
    c_["q37"] = str(sp.solve(sp.diff(L, c), c)[0])
    V = s * (12 - 2 * s) ** 2
    c_["q38"] = str(max((v for v in sp.solve(sp.diff(V, s), s) if 0 < v < 6), key=lambda v: V.subs(s, v)))
    D2 = x**2 + (x**2 - 2) ** 2
    crit = sp.solve(sp.diff(D2, x), x)
    c_["q39"] = str(sp.sqrt(min((D2.subs(x, v) for v in crit), key=lambda e: float(e))))

    # ---- integration and the FTC
    c_["q40"] = str(sp.integrate(x**2, (x, 0, 1)))
    c_["q41"] = str(sp.integrate(sp.sin(x), (x, 0, sp.pi)))
    c_["q42"] = str(sp.integrate(1 / x, (x, 1, sp.E)))
    c_["q43"] = str(sp.integrate(3 * x**2 - 2 * x + 1, (x, 0, 2)))
    F = sp.integrate(2 * t * sp.cos(t**2), (t, 0, x))
    c_["q44"] = str(F)
    c_["q45"] = str(sp.integrate(sp.exp(2 * x), (x, 0, 1)))
    tt = sp.Symbol("tt", nonnegative=True)
    xn = sp.Symbol("x", nonnegative=True)
    c_["q46"] = str(sp.diff(sp.Integral(sp.sqrt(1 + tt**3), (tt, 0, xn)), xn).doit())
    c_["q47"] = str(sp.diff(sp.integrate(sp.cos(t), (t, 0, x**2)), x))
    c_["q48"] = str(sp.integrate(sp.sin(x), (x, 0, sp.pi)) / sp.pi)
    c_["q49"] = str(sp.integrate(x - x**2, (x, 0, 1)))
    c_["q50"] = str(sp.integrate(1 / (1 + x), (x, 0, 1)))
    c_["q51"] = str(sum(sp.Rational(i, 4) * sp.Rational(1, 4) for i in range(4)))

    # ---- L'Hopital
    c_["q52"] = lim((sp.exp(x) - 1 - x) / x**2, x, 0)
    c_["q53"] = lim(x**2 * sp.exp(-x), x, sp.oo)
    c_["q54"] = lim((sp.sin(x) - x) / x**3, x, 0)
    c_["q55"] = lim(x**x, x, 0, "+")
    c_["q56"] = lim(sp.log(x) / (x - 1), x, 1)
    return confirm("S-M01", c_)


if __name__ == "__main__":
    sys.exit(main())
