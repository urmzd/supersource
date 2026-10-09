"""Recompute every auto-checked answer of S-M03a independently (see _confirm.py).

Systems by SymPy's `linsolve`, reduced echelon forms by `rref`, LU by a
from-scratch Doolittle elimination (with and without partial pivoting) in
exact rationals, ranks, null spaces, column spaces, determinants, and
eigenvalues by SymPy's Matrix methods, Gram-Schmidt by SymPy's `QRdecomposition`
(signs normalized to a positive diagonal of R), and least squares by the normal
equations.

    uv run --project course/harness python course/oracle/solve/S-M03a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402

M = sp.Matrix
x, y, z, k, c, t = sp.symbols("x y z k c t")


def mat(m: sp.Matrix) -> str:
    return "[" + ", ".join("[" + ", ".join(str(v) for v in m.row(i)) + "]" for i in range(m.rows)) + "]"


def vec(v) -> str:
    return "[" + ", ".join(str(e) for e in list(v)) + "]"


def basis(vs) -> str:
    return "[" + ", ".join(vec(v) for v in vs) + "]"


def doolittle(A: sp.Matrix, pivot: bool):
    """PA = LU by elimination in exact arithmetic; returns L, U."""
    n = A.rows
    U, L, P = A.copy(), sp.eye(n), sp.eye(n)
    for j in range(n):
        if pivot:
            p = max(range(j, n), key=lambda i: abs(U[i, j]))
            if p != j:
                U.row_swap(p, j)
                P.row_swap(p, j)
                for q in range(j):
                    L[p, q], L[j, q] = L[j, q], L[p, q]
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, :] = U[i, :] - L[i, j] * U[j, :]
    assert P * A == L * U
    return L, U


def main() -> int:
    a: dict[str, str] = {}

    # ---- Gauss and LU
    (s1,) = sp.linsolve([x + 2 * y - 5, 3 * x + 4 * y - 6], [x, y])
    a["q1"] = vec(s1)
    (s2,) = sp.linsolve([x + y + z - 6, 2 * x - y + z - 3, x + 2 * y - z - 2], [x, y, z])
    a["q2"] = vec(s2)
    a["q3"] = mat(M([[1, 2, 3], [2, 4, 7]]).rref()[0])
    L, U = doolittle(M([[2, 1], [4, 5]]), pivot=False)
    a["q4"], a["q5"] = mat(L), mat(U)
    L, U = doolittle(M([[1, 2], [3, 4]]), pivot=True)
    a["q6"], a["q7"] = mat(L), mat(U)
    a["q8"] = str(len(sp.linsolve([x + y - 1, x + y - 2], [x, y])))
    ks = [kv for kv in sp.solve(sp.Matrix([[1, k], [k, 1]]).det(), k)
          if sp.linsolve([x + kv * y - 1, kv * x + y - 1], [x, y]).free_symbols]
    a["q9"] = "{" + ", ".join(str(v) for v in ks) + "}"

    # ---- span, basis, dimension
    span = M([[1, 0], [0, 1], [1, 1]])
    a["q11"] = str(span.rank() == span.row_join(M([1, 2, 3])).rank()).lower()
    a["q12"] = str(M([[1, 2], [2, 4]]).rank() == 2).lower()
    a["q13"] = str(M([[1, 0, 1], [0, 1, 1], [1, 1, 2]]).rank())
    a["q14"] = basis(M([[1, 2], [2, 4], [3, 6]]).columnspace())
    a["q15"] = basis(M([[1, 1, 1]]).nullspace())
    a["q16"] = vec(M([[1, 1], [1, -1]]).solve(M([3, 5])))
    sym_basis = [M([[1, 0], [0, 0]]), M([[0, 0], [0, 1]]), M([[0, 1], [1, 0]])]
    a["q17"] = str(M([list(b) for b in sym_basis]).rank())
    a["q18"] = "{" + ", ".join(str(v) for v in sp.solve(M([[1, c], [c, 4]]).det(), c)) + "}"
    a["q19"] = str(len(M([[1, 2, 3], [2, 4, 6]]).nullspace()))

    # ---- maps and rank-nullity
    th = sp.pi / 2
    a["q21"] = mat(M([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]]))
    e1, e2 = M([1, 0]), M([0, 1])
    T = lambda v: M([v[0] + v[1], 2 * v[0], v[1]])  # noqa: E731
    a["q22"] = mat(T(e1).row_join(T(e2)))
    A3 = M([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a["q23"] = str(A3.rank())
    a["q24"] = str(len(A3.nullspace()))
    A57 = sp.zeros(5, 7)
    for i in range(4):
        A57[i, i] = 1
        A57[i, 4 + (i % 3)] = i + 2
    A57[4, :] = A57[0, :] + 3 * A57[2, :]
    assert A57.rank() == 4  # a concrete 5 x 7 matrix of rank 4
    a["q25"] = str(len(A57.nullspace()))
    T1 = lambda v: v + 1  # noqa: E731
    a["q26"] = str(all(T1(v + w) == T1(v) + T1(w) and T1(3 * v) == 3 * T1(v) for v, w in ((1, 2), (0, 0)))).lower()
    Tm, Sm = M([[0, 1], [1, 0]]), M([[2, 0], [0, 1]])
    a["q27"] = mat(Sm * Tm)

    # ---- determinants
    a["q29"] = str(M([[2, 1], [7, 4]]).det())
    a["q30"] = str(M([[1, 2, 3], [0, 4, 5], [0, 0, 6]]).det())
    a["q31"] = str(M([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).det())
    D = sp.diag(5, 1, 1)  # any 3 x 3 matrix with det 5
    a["q32"] = str((2 * D).det())
    a["q33"] = str(sp.diag(4, 1).inv().det())
    a["q34"] = str(abs(M([[3, 1], [1, 2]]).det()))

    # ---- eigen
    def evs(m):
        return "{" + ", ".join(str(v) for v in m.eigenvals()) + "}"

    a["q35"] = evs(M([[2, 0], [0, 3]]))
    A = M([[2, 1], [1, 2]])
    a["q36"] = evs(A)
    lam_max = max(A.eigenvals())
    a["q37"] = vec(next(vs[0] for lam, _, vs in A.eigenvects() if lam == lam_max))
    a["q38"] = str(M([[1, 2], [3, 4]]).charpoly(t).as_expr())
    a["q39"] = evs(M([[0, -1], [1, 0]]))
    a["q40"] = str(sp.prod(lam**m for lam, m in M([[4, 1], [2, 3]]).eigenvals().items()))
    a["q41"] = vec(A**10 * M([1, 1]))
    R = sp.Rational
    a["q42"] = str(max(abs(v) for v in M([[R(1, 2), R(2, 5)], [R(2, 5), R(1, 2)]]).eigenvals()))
    ev = sorted(A.eigenvals(), key=abs)
    a["q43"] = str(abs(ev[0] / ev[-1]))

    # ---- orthogonality and QR
    u, b = M([1, 1, 1]), M([1, 2, 3])
    a["q45"] = vec((u.dot(b) / u.dot(u)) * u)
    Q, Rm = M([[1, 1], [1, 0], [0, 1]]).QRdecomposition()
    S = sp.diag(*[sp.sign(Rm[i, i]) for i in range(Rm.rows)])
    Q, Rm = Q * S, S * Rm
    a["q46"] = mat(sp.simplify(Q))
    a["q47"] = mat(sp.simplify(Rm))
    X, Y = M([1, 2, 3]), M([1, 2, 2])
    a["q48"] = str(((X.T * X).inv() * (X.T * Y))[0])
    return confirm("S-M03a", a)


if __name__ == "__main__":
    sys.exit(main())
