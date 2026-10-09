"""Recompute every auto-checked answer of S-M04 independently (see _confirm.py).

Dot and cross products, norms, and angles with SymPy matrices; the plane from
a cross product of edge vectors; partials, gradients, Hessians, and Jacobians
by SymPy's `diff`; chain-rule results by substituting first and then
differentiating the composite; directional derivatives by differentiating
f(x + s u) in s; critical points by `solve` and classification by the
Hessian's eigenvalues; double integrals by iterated `integrate` (polar where
the region is a disk); and constrained optima by solving the Lagrange system.

    uv run --project course/harness python course/oracle/solve/S-M04.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

M = sp.Matrix
x, y, z, t, w, r, th, s, lam = sp.symbols("x y z t w r theta s lam", real=True)
a, b, c = sp.symbols("a b c", real=True)


def vec(v) -> str:
    return "[" + ", ".join(str(sp.simplify(e)) for e in list(v)) + "]"


def mat(m) -> str:
    return "[" + ", ".join("[" + ", ".join(str(v) for v in m.row(i)) + "]" for i in range(m.rows)) + "]"


def grad(f, vs):
    return M([sp.diff(f, v) for v in vs])


def classify(f, point) -> str:
    H = sp.hessian(f, (x, y)).subs(point)
    ev = list(H.eigenvals())
    if all(e > 0 for e in ev):
        return "minimum"
    if all(e < 0 for e in ev):
        return "maximum"
    return "saddle"


def lagrange(f, g, vs, positive=False):
    """Stationary points of f on g = 0; returns the list of f values."""
    eqs = [sp.diff(f, v) - lam * sp.diff(g, v) for v in vs] + [g]
    sols = sp.solve(eqs, list(vs) + [lam], dict=True)
    vals = []
    for so in sols:
        if positive and not all(so[v].is_positive for v in vs):
            continue
        vals.append(sp.simplify(f.subs(so)))
    return vals, sols


def main() -> int:
    k: dict[str, str] = {}

    # ---- vectors and geometry
    u1, v1 = M([1, 2, 3]), M([4, -5, 6])
    k["q1"] = str(u1.dot(v1))
    k["q2"] = vec(M([1, 2, 3]).cross(M([4, 5, 6])))
    p, q = M([1, 0]), M([1, 1])
    k["q3"] = str(sp.acos(p.dot(q) / (p.norm() * q.norm())))
    k["q4"] = str(M([2, 3, 6]).norm())
    P1, P2, P3 = M([1, 0, 0]), M([0, 1, 0]), M([0, 0, 1])
    nrm = (P2 - P1).cross(P3 - P1)
    k["q5"] = f"{nrm.dot(M([x, y, z]))} = {nrm.dot(P1)}"
    n6 = M([1, 1, 1])
    k["q6"] = str(sp.Abs(n6.dot(M([1, 2, 3]))) / n6.norm())
    k["q7"] = str(nrm.norm() / 2)
    k["q8"] = str(M([1, 2, -1]).dot(M([3, -1, 1])) == 0).lower()

    # ---- partials and the gradient
    f9 = x**2 * y + sp.sin(x * y)
    k["q9"], k["q10"] = str(sp.diff(f9, x)), str(sp.diff(f9, y))
    k["q11"] = vec(grad(x**2 + 3 * x * y + y**2, (x, y)).subs({x: 1, y: 2}))
    k["q12"] = vec(grad(a**2 + b**2 + c**2, (a, b, c)))
    k["q13"] = vec(grad(3 * x - 2 * y + 7, (x, y)))
    k["q14"] = str(sp.diff(x**3 * y**2, x, y))
    k["q15"] = vec(grad(sp.log(sp.exp(x) + sp.exp(y)), (x, y)))
    A, bb, X = M([[1, 0], [0, 2]]), M([1, 1]), M([x, y])
    f16 = ((A * X - bb).T * (A * X - bb))[0] / 2
    k["q16"] = vec(grad(f16, (x, y)).subs({x: 0, y: 0}))
    k["q17"] = str(sp.diff(x * sp.exp(y), y).subs({x: 2, y: 0}))
    k["q18"] = str(sp.diff((w * x - y) ** 2, w))

    # ---- chain rule (composite first, then differentiate)
    k["q19"] = str(sp.simplify(sp.diff(sp.cos(t) ** 2 + sp.sin(t) ** 2, t)))
    k["q20"] = str(sp.diff(t**2 * t**3, t))
    f21 = (x + y) * (x - y)
    k["q21"], k["q22"] = str(sp.diff(f21, x)), str(sp.diff(f21, y))
    sig = 1 / (1 + sp.exp(-w * x))
    k["q23"] = str(sp.simplify(sp.diff((sig - 1) ** 2, w).subs({w: 0, x: 2})))
    rp = sp.Symbol("r", positive=True)
    k["q24"] = str(sp.diff(sp.simplify((rp * sp.cos(th)) ** 2 + (rp * sp.sin(th)) ** 2), rp))
    k["q25"] = mat(M([x + y, x * y]).jacobian([x, y]).subs({x: 1, y: 2}))
    k["q26"] = str(sp.simplify(M([rp * sp.cos(th), rp * sp.sin(th)]).jacobian([rp, th]).det()))
    k["q27"] = str(sp.diff(((3 * x + 1) ** 2 - 4) ** 2, x).subs(x, 0))
    J = M([[1, 2], [3, 4]]) * X
    k["q28"] = vec((M([[1, 1]]) * J.jacobian([x, y])).T)

    # ---- directional derivatives: d/ds f(p + s u) at s = 0
    def dd(f, pt, u):
        return sp.diff(f.subs({x: pt[0] + s * u[0], y: pt[1] + s * u[1]}), s).subs(s, 0)

    k["q29"] = str(dd(x**2 + y**2, (1, 1), (sp.Rational(3, 5), sp.Rational(4, 5))))
    g30 = grad(x * y, (x, y)).subs({x: 2, y: 3})
    k["q30"] = str(sp.simplify(dd(x * y, (2, 3), list(g30 / g30.norm()))))
    g31 = grad(x**2 + 2 * y**2, (x, y)).subs({x: 1, y: 1})
    k["q31"] = vec(-g31 / g31.norm())
    k["q32"] = str(dd(sp.exp(x) * sp.sin(y), (0, sp.pi / 2), (1, 0)))
    g33 = grad(x**2 - y**2, (x, y)).subs({x: 1, y: 1})
    perp = M([-g33[1], g33[0]])
    assert dd(x**2 - y**2, (1, 1), list(perp)) == 0
    k["q33"] = vec(perp / perp.norm())

    # ---- Hessian and the second-derivative test
    k["q35"] = mat(sp.hessian(x**2 + 3 * x * y + 2 * y**2, (x, y)))
    k["q36"] = mat(sp.hessian(x**3 + y**3, (x, y)).subs({x: 1, y: 1}))
    k["q37"] = classify(x**2 - y**2, {x: 0, y: 0})
    k["q38"] = classify(x**2 + x * y + y**2, {x: 0, y: 0})
    (cp,) = sp.solve(list(grad(x**2 + y**2 - 2 * x + 4 * y, (x, y))), [x, y], dict=True)
    k["q39"] = vec(M([cp[x], cp[y]]))
    f40 = x**3 - 3 * x + y**2
    assert grad(f40, (x, y)).subs({x: 1, y: 0}) == M([0, 0])
    k["q40"] = classify(f40, {x: 1, y: 0})
    k["q41"] = str(sp.hessian(x**2 + 3 * x * y + 2 * y**2, (x, y)).is_positive_definite).lower()
    H42 = sp.hessian((x**2 + 100 * y**2) / 2, (x, y))
    ev42 = list(H42.eigenvals())
    k["q42"] = str(max(ev42) / min(ev42))

    # ---- multiple integrals
    k["q43"] = str(sp.integrate(x * y, (y, 0, 2), (x, 0, 1)))
    k["q44"] = str(sp.integrate(1, (y, 0, x), (x, 0, 1)))
    k["q45"] = str(sp.integrate(rp, (rp, 0, 1), (th, 0, 2 * sp.pi)))
    k["q46"] = str(sp.integrate(sp.exp(-(rp**2)) * rp, (rp, 0, sp.oo), (th, 0, 2 * sp.pi)))
    k["q47"] = str(sp.integrate(2 * y, (y, 0, x), (x, 0, 1)))
    k["q48"] = str(sp.integrate((1 - rp**2) * rp, (rp, 0, 1), (th, 0, 2 * sp.pi)))
    k["q49"] = str(sp.integrate(sp.exp(-(x**2 + y**2) / 2) / (2 * sp.pi), (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo)))
    jac = M([x + y, x - y]).jacobian([x, y]).det()
    k["q50"] = str(sp.integrate(1 / sp.Abs(jac), (s, -1, 1), (t, -1, 1)))

    # ---- Lagrange
    v51, _ = lagrange(x + y, x**2 + y**2 - 1, (x, y))
    k["q51"] = str(max(v51, key=float))
    v52, _ = lagrange(x**2 + y**2, x + y - 1, (x, y))
    k["q52"] = str(min(v52, key=float))
    v53, _ = lagrange(x * y, x + y - 10, (x, y))
    k["q53"] = str(max(v53, key=float))
    p1, p2, p3 = sp.symbols("p1 p2 p3", positive=True)
    H = -(p1 * sp.log(p1) + p2 * sp.log(p2) + p3 * sp.log(p3))
    _, s54 = lagrange(H, p1 + p2 + p3 - 1, (p1, p2, p3))
    k["q54"] = str(s54[0][p1])
    xp, yp, zp = sp.symbols("xp yp zp", positive=True)
    v55, _ = lagrange(xp * yp * zp, xp + yp + zp - 3, (xp, yp, zp), positive=True)
    k["q55"] = str(max(v55, key=float))
    v56, _ = lagrange(2 * xp + 3 * yp, xp * yp - 6, (xp, yp), positive=True)
    k["q56"] = str(min(v56, key=float))
    _, s57 = lagrange(x**2 + y**2 + z**2, x + 2 * y + 2 * z - 9, (x, y, z))
    k["q57"] = vec(M([s57[0][x], s57[0][y], s57[0][z]]))
    return confirm("S-M04", k)


if __name__ == "__main__":
    sys.exit(main())
