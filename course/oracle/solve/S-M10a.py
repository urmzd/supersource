"""Recompute every auto-checked answer of S-M10a independently (see _confirm.py).

Convexity from Hessians (SymPy) or a sampled midpoint test, constants from
eigenvalues, step-size sets and optima by scanning exact rationals, every
iteration (GD, Armijo, heavy ball, Adam, AdamW) run step by step in exact
arithmetic, and closed forms checked against those runs.
"""

from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def fr(x: F) -> str:
    return f"{x.numerator}/{x.denominator}"


def main() -> int:
    c: dict[str, str] = {}
    x, y = sp.symbols("x y", real=True)
    xp = sp.symbols("xp", positive=True)
    neg = sp.solve_univariate_inequality(sp.diff(x**4, x, 2) < 0, x, relational=False)
    c["q1.a"] = str(neg == sp.S.EmptySet).lower()
    c["q1.b"] = str(bool(sp.diff(sp.log(xp), xp, 2).subs(xp, 1) >= 0)).lower()
    lse = sp.log(sp.exp(x) + sp.exp(y))
    H = sp.hessian(lse, (x, y))
    rng = np.random.default_rng(0)
    psd = True
    for _ in range(200):
        a, b = rng.uniform(-5, 5, size=2)
        ev = np.linalg.eigvalsh(np.array(H.subs({x: a, y: b}).evalf(), dtype=float))
        psd &= bool(ev.min() >= -1e-12)
    c["q1.c"] = str(psd).lower()

    A = sp.diag(1, 10)
    ev = sorted(A.eigenvals())
    c["q2.a"], c["q2.b"] = str(max(ev)), str(min(ev))
    c["q2.c"] = str(sp.Rational(max(ev), min(ev)))
    f3 = x**3 - 3 * x
    region = sp.solve_univariate_inequality(sp.diff(f3, x, 2) >= 0, x, relational=False)
    c["q3"] = "[" + str(region.start) + ", oo)" if region.end == sp.oo else str(region)

    # q5(a): scan step sizes on a fine rational grid and find where GD on 5x^2 converges
    def converges(eta: F) -> bool:
        return abs(1 - 10 * eta) < 1

    grid = [F(k, 1000) for k in range(1, 400)]
    good = [e for e in grid if converges(e)]
    assert good[0] == F(1, 1000) and good[-1] == F(199, 1000) and not converges(F(1, 5))
    c["q5.a"] = (
        "(0, 1/5)"  # open at 0 (eta > 0) and at 1/5 (|1 - 10 eta| = 1 never contracts)
    )

    lams = [F(1), F(10)]
    worst = lambda eta: max(abs(1 - eta * l) for l in lams)  # noqa: E731
    # worst() is piecewise linear in eta, so its minimum sits at a kink: solve
    # (1 - eta a) = +-(1 - eta b) for every pair, then confirm on a fine grid
    e = sp.Symbol("e", positive=True)
    kinks = set()
    for la, lb in itertools.product(lams, repeat=2):
        for sgn in (1, -1):
            for sol in sp.solve(sp.Eq(1 - e * la, sgn * (1 - e * lb)), e):
                kinks.add(F(int(sp.fraction(sol)[0]), int(sp.fraction(sol)[1])))
    best = min(kinks, key=worst)
    assert all(worst(F(k, 10000)) >= worst(best) for k in range(1, 4000))
    c["q5.b"] = fr(best)
    c["q5.c"] = fr(worst(best))
    c["q5.d"] = fr(worst(F(1, 10)))

    t = next(t for t in itertools.count(1) if F(9, 10) ** t <= F(1, 10**6))
    c["q6"] = str(t)

    xt = F(1)
    for _ in range(3):
        xt = xt - F(1, 4) * 2 * xt
    c["q7"] = fr(xt)

    fq = lambda z: z * z  # noqa: E731
    x0, g0 = F(1), F(2)
    d = -g0
    alpha = F(1)
    while not fq(x0 + alpha * d) <= fq(x0) + F(1, 2) * alpha * g0 * d:
        alpha *= F(1, 2)
    c["q8"] = fr(alpha)
    c["q9"] = fr(F(4 * 9, 2 * 10))

    g, beta = sp.symbols("g beta", real=True)
    kk = sp.Symbol("k", integer=True, nonnegative=True)
    series = sp.summation(g * beta**kk, (kk, 0, sp.oo))  # v_t -> sum of beta^k g
    series = series.replace(
        lambda e: isinstance(e, sp.Piecewise), lambda e: e.args[0][0]
    )
    c["q10.a"] = str(sp.simplify(series))  # the convergent branch, |beta| < 1
    vf = 0.0
    for _ in range(200):
        vf = 0.5 * vf + 1.0
    assert (
        abs(
            float(
                sp.sympify(c["q10.a"], locals={"beta": beta, "g": g}).subs(
                    {beta: 0.5, g: 1}
                )
            )
            - vf
        )
        < 1e-12
    )
    vv = F(0)
    for _ in range(500):
        vv = F(9, 10) * vv + 1
    c["q10.b"] = str(round(float(vv)))

    r, eta, h = sp.symbols("r eta h", real=True)
    # substitute x_t = r^t into the recurrence and divide by r^(t-1)
    tt = sp.symbols("tt", integer=True)
    rp = sp.Symbol("r", positive=True)
    rec = rp ** (tt + 1) - (1 + beta - eta * h) * rp**tt + beta * rp ** (tt - 1)
    c["q11.a"] = str(sp.expand(sp.powsimp(sp.expand(rec / rp ** (tt - 1)), force=True)))
    rc = sp.Symbol("rc")  # complex: the roots of interest are not real
    roots = sp.solve(
        sp.Eq(rc**2 - sp.Rational(1, 2) * rc + sp.Rational(81, 100), 0), rc
    )
    assert len(roots) == 2 and all(not z.is_real for z in roots)
    mods = {sp.sqrt(sp.simplify(sp.expand(z * sp.conjugate(z)))) for z in roots}
    assert len(mods) == 1
    c["q11.b"] = str(mods.pop())
    k = 100
    c["q12"] = str(((sp.sqrt(k) - 1) / (sp.sqrt(k) + 1)) ** 2)
    vb = F(0)
    for _ in range(3):
        vb = F(1, 2) * vb + 1
    c["q13"] = fr(vb)

    b1, tsym = (
        sp.Symbol("beta1", positive=True),
        sp.Symbol("t", integer=True, positive=True),
    )
    for b1v, T in ((F(9, 10), 5), (F(1, 3), 7)):
        m = F(0)
        for _ in range(T):
            m = b1v * m + (1 - b1v) * 1
        assert m == 1 - b1v**T
    gg = sp.Symbol("g", real=True)
    unrolled = sp.summation((1 - b1) * b1 ** (tsym - kk - 1) * gg, (kk, 0, tsym - 1))
    c["q14"] = str(sp.simplify(unrolled))

    def adam_step(theta, grad, lr, wd=F(0)):
        b1v, b2v = F(9, 10), F(999, 1000)
        m = (1 - b1v) * grad
        v2 = (1 - b2v) * grad * grad
        mh, vh = m / (1 - b1v), v2 / (1 - b2v)
        root = F(math.isqrt(vh.numerator), math.isqrt(vh.denominator))
        assert root * root == vh
        return theta - lr * wd * theta - lr * mh / root

    c["q15"] = fr(adam_step(F(0), F(-3), F(1, 100)) - 0)
    c["q16"] = fr(adam_step(F(1), F(4), F(1, 10), F(1, 10)))
    return confirm("S-M10a", c)


if __name__ == "__main__":
    sys.exit(main())
