"""Recompute every auto-checked answer of S-M03b independently (see _confirm.py).

Inner products, norms, and projections by SymPy matrix products; singular
values as square roots of the eigenvalues of A^T A (SymPy, exact), confirmed
against numpy's float64 SVD; Eckart-Young errors from the singular values by
their definitions (largest dropped value, root of the sum of squares) and
confirmed on an explicit diagonal matrix with numpy norms; the adapter count
by summing the factor shapes; FLOPs by counting the terms of the triple loop
symbolically; bytes by listing what is read and written; the smallest
compute-bound N by search.

    uv run --project course/harness python course/oracle/solve/S-M03b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

Mx = sp.Matrix


def vec(v) -> str:
    return "[" + ", ".join(str(e) for e in list(v)) + "]"


def singular_values(A: sp.Matrix) -> list:
    ev = (A.T * A).eigenvals()
    vals = []
    for lam, mult in ev.items():
        vals += [sp.sqrt(sp.nsimplify(lam))] * mult
    vals = sorted(vals, key=lambda s: float(s), reverse=True)
    return vals[: min(A.shape)]


def main() -> int:
    c: dict[str, str] = {}
    u, v = Mx([1, 2, 2]), Mx([2, 0, 1])
    c["q1.a"] = str(u.dot(v))
    c["q1.b"] = str(sp.nsimplify(u.dot(v) / (u.norm() * v.norm())))
    a, b = Mx([1, 1, 1]), Mx([1, 2, 3])
    p = (a.dot(b) / a.dot(a)) * a
    assert (b - p).dot(a) == 0
    c["q2"] = vec(p)

    A4 = Mx([[1, 1], [0, 1]])
    s4 = singular_values(A4)
    np4 = np.linalg.svd(np.array([[1.0, 1.0], [0.0, 1.0]]), compute_uv=False)
    assert all(abs(float(x) - y) < 1e-12 for x, y in zip(s4, np4))
    c["q4.a"] = str(sp.radsimp(sp.sqrtdenest(s4[0])))
    c["q4.b"] = str(sp.radsimp(sp.sqrtdenest(s4[1])))
    A5 = Mx([[0, 2], [3, 0], [0, 0]])
    s5 = singular_values(A5)
    np5 = np.linalg.svd(
        np.array([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), compute_uv=False
    )
    assert all(abs(float(x) - y) < 1e-12 for x, y in zip(s5, np5))
    c["q5"] = vec(s5)

    S = [5, 3, 1]
    D = np.diag([5.0, 3.0, 1.0])
    D1 = np.diag([5.0, 0.0, 0.0])
    assert abs(np.linalg.norm(D - D1, 2) - 3.0) < 1e-12
    c["q6.a"] = str(max(S[1:]))
    c["q6.b"] = str(sp.sqrt(sum(sp.Integer(x) ** 2 for x in S[1:])))
    c["q6.c"] = str(
        next(
            r
            for r in range(0, 4)
            if sp.sqrt(sum(sp.Integer(x) ** 2 for x in S[r:])) <= sp.Rational(3, 2)
        )
    )

    r = sp.Symbol("r", positive=True)
    c["q7"] = str(4096 * r + r * 4096)

    M, N, K = sp.symbols("M N K", positive=True)
    i, j, k = sp.symbols("i j k", integer=True)
    flops = sp.summation(sp.summation(sp.summation(2, (k, 1, K)), (j, 1, N)), (i, 1, M))
    c["q9.a"] = str(sp.simplify(flops))
    mv_flops = sp.summation(
        sp.summation(2, (k, 1, N)), (i, 1, N)
    )  # N rows, N terms each
    mv_bytes = 4 * (N * N) + 4 * N + 4 * N  # read W, read x, write y
    c["q9.b"] = str(sp.simplify(mv_flops / mv_bytes))
    mm_flops = flops.subs({M: N, K: N})
    mm_bytes = 4 * (N * N + N * N + N * N)  # read A, read B, write C
    c["q10.a"] = str(sp.simplify(mm_flops / mm_bytes))
    ridge = sp.Rational(10 * 10**12, 100 * 10**9)
    c["q10.b"] = str(next(n for n in range(1, 10**4) if sp.Rational(n, 6) >= ridge))
    return confirm("S-M03b", c)


if __name__ == "__main__":
    sys.exit(main())
