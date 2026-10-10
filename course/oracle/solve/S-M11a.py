"""Recompute every auto-checked answer of S-M11a independently (see _confirm.py).

Every entropy and divergence is a direct sum over the stated distribution in
SymPy (base 2 or e as asked), the conditional entropy is averaged over the
conditional distributions rather than taken from the chain rule, the
maximizer of the binary entropy is found by solving H_b'(p) = 0, and the k3
sign claim is checked by its minimum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

R = sp.Rational


def H(ps, base=2):
    return sp.nsimplify(sum(-p * sp.log(p, base) for p in ps if p != 0))


def main() -> int:
    c: dict[str, str] = {}
    c["q1.a"] = str(sp.simplify(H([R(1, 2), R(1, 2)])))
    c["q1.b"] = str(sp.simplify(H([R(1, 256)] * 256)))
    c["q1.c"] = str(sp.simplify(H([R(1, 2), R(1, 4), R(1, 4)])))
    V = sp.Symbol("V", positive=True, integer=True)
    i = sp.Symbol("i", integer=True)
    c["q2"] = str(sp.simplify(sp.summation(-(1 / V) * sp.log(1 / V), (i, 1, V))))

    p = sp.Symbol("p", positive=True)
    hb = -p * sp.log(p, 2) - (1 - p) * sp.log(1 - p, 2)
    crit = sp.solve(sp.diff(hb, p), p)
    assert len(crit) == 1 and sp.diff(hb, p, 2).subs(p, crit[0]) < 0
    c["q3.a"] = str(crit[0])
    c["q3.b"] = str(sp.simplify(hb.subs(p, R(1, 4))))

    joint = {(0, 0): R(1, 2), (0, 1): R(1, 4), (1, 0): R(0), (1, 1): R(1, 4)}
    c["q4.a"] = str(H(joint.values()))
    px = {x: sum(v for (a, _), v in joint.items() if a == x) for x in (0, 1)}
    c["q4.b"] = str(H(px.values()))
    hcond = sum(
        px[x] * H([joint[(x, y)] / px[x] for y in (0, 1)]) for x in (0, 1) if px[x] != 0
    )
    c["q4.c"] = str(sp.simplify(hcond))

    P, Q = [R(1, 2), R(1, 2)], [R(1, 4), R(3, 4)]
    ce = sum(-a * sp.log(b, 2) for a, b in zip(P, Q))
    kl_pq = sum(a * sp.log(a / b, 2) for a, b in zip(P, Q))
    kl_qp = sum(b * sp.log(b / a, 2) for a, b in zip(P, Q))
    c["q6.a"] = str(sp.simplify(ce))
    c["q6.b"] = str(sp.simplify(kl_pq))
    c["q6.c"] = str(sp.simplify(kl_qp))
    c["q6.d"] = str(sp.simplify(kl_pq - kl_qp) == 0).lower()
    c["q7"] = str(sp.simplify(H([R(1, 256)] * 256, base=sp.E)))

    r = sp.Symbol("r", real=True)
    k3 = sp.exp(r) - r - 1
    c["q8.a"] = str(sp.simplify(k3.subs(r, sp.log(2))))
    stationary = sp.solve(sp.diff(k3, r), r)
    convex = sp.simplify(sp.diff(k3, r, 2)) == sp.exp(
        r
    )  # > 0: the stationary point is the global min
    c["q8.b"] = str(convex and all(k3.subs(r, s) >= 0 for s in stationary)).lower()
    return confirm("S-M11a", c)


if __name__ == "__main__":
    sys.exit(main())
