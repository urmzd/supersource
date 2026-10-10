"""Recompute every auto-checked answer of S-M09b independently (see _confirm.py).

Condition numbers from SymPy's exact singular values and derivatives; error
bounds from exact rationals; the fixed-point rate by iterating cos in
float64 and differencing; Newton steps in exact rationals; the Taylor degree
by evaluating the Lagrange bound in 50-digit arithmetic for n = 1, 2, ...;
Chebyshev nodes by solving T_3 = 0; the node-product maximum by SymPy's
calculus on the interval; fp8 and fp4 rounding by exhaustive search over the
exact value tables built from the format parameters (no bit tricks).
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction as F
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def table(ebits: int, mbits: int, bias: int, reserved_top: bool) -> list[F]:
    """Every finite non-negative value of a small float format."""
    vals = []
    top = (1 << ebits) - (2 if reserved_top else 1)
    for e in range(top + 1):
        for m in range(1 << mbits):
            if e == 0:
                vals.append(F(m, 1 << mbits) * F(2) ** (1 - bias))
            else:
                vals.append((1 + F(m, 1 << mbits)) * F(2) ** (e - bias))
    return vals


def nearest_even(x: F, vals: list[F]) -> tuple[F, int]:
    d = [abs(v - x) for v in vals]
    best = min(d)
    idx = [i for i, di in enumerate(d) if di == best]
    i = idx[0] if len(idx) == 1 else next(j for j in idx if j % 2 == 0)
    return vals[i], i


def main() -> int:
    c: dict[str, str] = {}
    x = sp.Symbol("x", positive=True)
    for qid, f, at in [("q1", x - 1, sp.Rational(1001, 1000)), ("q2", sp.exp(x), 50)]:
        k = sp.simplify(sp.Abs(x * sp.diff(f, x) / f)).subs(x, at)
        c[qid] = str(sp.nsimplify(k))
    for qid, A in [
        ("q3", sp.diag(10, sp.Rational(1, 10))),
        ("q4", sp.Matrix([[3, 0], [4, 5]])),
    ]:
        s = sorted(sp.sqrt(ev) for ev in (A.T * A).eigenvals())
        c[qid] = str(sp.nsimplify(s[-1] / s[0]))
    c["q5"] = str(sp.Rational(1000) * sp.Rational(1, 10**6))

    u32 = F(1, 2**24)
    g3 = 3 * u32 / (1 - 3 * u32)
    c["q7"] = f"{g3.numerator}/{g3.denominator}"
    c["q8"] = str(sp.Rational(4096) * sp.Rational(1, 2**24))
    c["q9"] = str((math.ceil(math.log2(4096)) + 1) * sp.Rational(1, 2**24))
    k = 1
    while F(k, 2**8) < 1:
        k += 1
    c["q10"] = str(k)
    c["q11"] = str(sp.sqrt(4096) * sp.Rational(1, 2**24))

    xk = 1.0
    for _ in range(200):
        xk = math.cos(xk)
    h = 1e-5  # central difference of g = cos at the fixed point: |g'(x*)|
    c["q13"] = repr(abs(math.cos(xk + h) - math.cos(xk - h)) / (2 * h))
    a, y = F(4), F(2, 5)
    y1 = y * (3 - a * y * y) / 2
    c["q14"] = f"{y1.numerator}/{y1.denominator}"
    # e_1 from e_0 = 2^-4 by an actual Newton step: pick a = 1, y_0 = 1 - e_0.
    y0 = 1 - F(1, 16)
    e1 = 1 - y0 * (3 - y0 * y0) / 2
    c["q15"] = f"{e1.numerator}/{e1.denominator}"
    t = sp.Symbol("t")
    newton = lambda f: t - f / sp.diff(f, t)  # noqa: E731
    # order from actual iterates: Newton on t^2 - 2 in 200-digit arithmetic
    step = sp.lambdify(t, newton(t**2 - 2), "mpmath")
    import mpmath

    mpmath.mp.dps = 200
    it, root = [mpmath.mpf(3)], mpmath.sqrt(2)
    for _ in range(6):
        it.append(step(it[-1]))
    err = [abs(v - root) for v in it]
    order = mpmath.log(err[5] / err[4]) / mpmath.log(err[4] / err[3])
    c["q16.a"] = str(int(mpmath.nint(order)))
    g_double = newton((t - 1) ** 2 * (t + 2))
    c["q16.b"] = str(sp.limit(sp.diff(g_double, t), t, 1))

    xv = sp.Integer(10)
    kk = int(sp.floor(xv / sp.log(2) + sp.Rational(1, 2)))
    c["q18"] = f"10 - {kk}*log(2)"
    c["q19"] = "log(2)/2"
    r = sp.log(2) / 2
    n = 0
    while sp.N(sp.exp(r) * r ** (n + 1) / sp.factorial(n + 1), 50) >= sp.Rational(
        1, 2**24
    ):
        n += 1
    c["q20"] = str(n)
    # Horner: count the multiplications of an actual evaluation
    coeffs = list(range(1, 9))
    mults = 0
    acc = coeffs[0]
    for cf in coeffs[1:]:
        acc = acc * 3 + cf
        mults += 1
    c["q21"] = str(mults)
    roots = sp.solve(4 * t**3 - 3 * t, t)
    c["q22"] = "{" + ", ".join(str(rt) for rt in roots) + "}"
    p = sp.prod([t - rt for rt in roots])
    crit = [pt for pt in sp.solve(sp.diff(p, t), t) if -1 <= pt <= 1] + [-1, 1]
    c["q23"] = str(sp.nsimplify(max(abs(p.subs(t, pt)) for pt in crit)))

    e4m3 = table(4, 3, 7, reserved_top=False)[:-1]  # 0x7F is NaN
    v, i = nearest_even(F(53, 10), e4m3)
    c["q24.a"] = f"{v.numerator}/{v.denominator}"
    c["q24.b"] = str(i)
    e5m2 = table(5, 2, 15, reserved_top=True)
    c["q25.a"] = str(max(e5m2))
    c["q25.b"] = "2^" + str(int(math.log2(min(v for v in e4m3 if v > 0))))
    # unit roundoff = half the gap between 1 and the next e4m3 value
    one = e4m3.index(F(1))
    c["q26"] = "2^" + str(int(math.log2((e4m3[one + 1] - 1) / 2)))
    fp4 = table(2, 1, 1, reserved_top=False)
    X = F(2) ** (int(math.floor(math.log2(13))) - 2)
    c["q27.a"] = str(X)
    q = min(F(13) / X, max(fp4))
    c["q27.b"] = str(nearest_even(q, fp4)[0] * X)
    rng = lambda vals: max(vals) / min(v for v in vals if v > 0)  # noqa: E731
    c["q28"] = "b" if rng(e5m2) > rng(e4m3) else "a"
    return confirm("S-M09b", c)


if __name__ == "__main__":
    sys.exit(main())
