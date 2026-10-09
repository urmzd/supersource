"""Recompute every auto-checked answer of S-M08 independently (see _confirm.py).

Shapes and strides from numpy itself; every gradient by SymPy differentiation
of the scalar function it belongs to (never by the closed-form rule the
chapter teaches), so a wrong rule in the key cannot confirm itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def vec(xs) -> str:
    return "[" + ", ".join(str(sp.simplify(x)) for x in xs) + "]"


def mat(M) -> str:
    M = sp.Matrix(M)
    return (
        "["
        + ", ".join(
            "[" + ", ".join(str(sp.simplify(M[i, j])) for j in range(M.cols)) + "]"
            for i in range(M.rows)
        )
        + "]"
    )


def grad_matrix(f, X: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(X.rows, X.cols, lambda i, j: sp.diff(f, X[i, j]))


def softmax(xs):
    e = [sp.exp(x) for x in xs]
    s = sum(e)
    return [t / s for t in e]


def main() -> int:
    c: dict[str, str] = {}
    B, T, din, dout = 2, 3, 4, 5
    X = np.zeros((B, T, din), dtype=np.float32)
    W = np.zeros((dout, din), dtype=np.float32)
    b = np.zeros((dout,), dtype=np.float32)
    Y = X @ W.T + b
    c["q1.a"] = vec(Y.shape)
    c["q1.b"] = vec(W.shape)
    c["q1.c"] = vec(b.shape)
    flat = np.arange(X.size).reshape(X.shape)
    c["q2.a"] = str(int(flat[1, 2, 1]))
    c["q2.b"] = vec([s // X.itemsize for s in X.strides])
    view = X.transpose(0, 2, 1)
    assert np.shares_memory(view, X)
    c["q2.c"] = vec([s // X.itemsize for s in view.strides])
    c["q3"] = str(int(np.broadcast_to(b, Y.shape)[..., 0].size))
    c["q4"] = vec([Y.size, X.size])

    x1, x2 = sp.symbols("x1 x2", real=True)
    xv = sp.Matrix([x1, x2])
    A = sp.Matrix([[1, 2], [0, 3]])
    f = (xv.T * A * xv)[0]
    c["q5"] = vec([sp.diff(f, v) for v in (x1, x2)])

    Xs = sp.Matrix(2, 2, sp.symbols("a0:4"))
    A6 = sp.Matrix([[1, 2], [3, 4]])
    c["q6"] = mat(grad_matrix((A6 * Xs).trace(), Xs))

    A7, b7 = sp.Matrix([[1, 0], [1, 1]]), sp.Matrix([1, 0])
    r = A7 * xv - b7
    f7 = (r.T * r)[0]
    c["q7"] = vec([sp.diff(f7, v).subs({x1: 1, x2: 2}) for v in (x1, x2)])

    t = sp.symbols("t", real=True)
    Xt = sp.Matrix([[2, t], [1, 1]])
    c["q8"] = mat(sp.diff(Xt.inv(), t).subs(t, 0))

    X9 = sp.Matrix([[2, 1], [0, 3]])
    g9 = grad_matrix(sp.log(Xs.det()), Xs).subs(dict(zip(Xs, X9)))
    c["q9"] = mat(g9)

    xw = sp.Matrix([1, 2])
    Wy = Xs * xw
    g10 = grad_matrix((Wy.T * Wy)[0], Xs).subs(dict(zip(Xs, sp.eye(2))))
    c["q10"] = mat(g10)

    z = sp.symbols("z", real=True)
    c["q11"] = str(sp.diff(1 / (1 + sp.exp(-z)), z))
    c["q12"] = str(sp.diff(sp.log(1 + sp.exp(z)), z))
    lse = sp.log(sp.exp(x1) + sp.exp(x2))
    c["q13"] = vec([sp.diff(lse, v) for v in (x1, x2)])
    nrm = sp.sqrt(x1**2 + x2**2)
    c["q14"] = vec([sp.diff(nrm, v).subs({x1: 3, x2: 4}) for v in (x1, x2)])
    relu_jac = np.diag((np.array([-1, 2, 0.5]) > 0).astype(int))
    c["q15"] = mat(relu_jac.tolist())

    # trace identities: check on random integer matrices (a counterexample suffices for b)
    rng = np.random.default_rng(0)
    Am, Bm, Cm = (rng.integers(-3, 4, size=(3, 3)) for _ in range(3))
    c["q17.a"] = str(
        all(
            np.trace(P @ Q @ R) == np.trace(R @ P @ Q)
            for P, Q, R in [
                tuple(rng.integers(-3, 4, size=(3, 3)) for _ in range(3))
                for _ in range(50)
            ]
        )
    ).lower()
    c["q17.b"] = str(
        bool(np.trace(Am @ Bm @ Cm) == np.trace(Bm @ Am @ Cm))
        and all(
            np.trace(P @ Q @ R) == np.trace(Q @ P @ R)
            for P, Q, R in [
                tuple(rng.integers(-3, 4, size=(3, 3)) for _ in range(3))
                for _ in range(50)
            ]
        )
    ).lower()

    # q18: L = <G, X W> with G = I; differentiate in each matrix
    Ws = sp.Matrix(2, 2, sp.symbols("w0:4"))
    X18, W18, G = sp.Matrix([[1, 2], [3, 4]]), sp.Matrix([[1, 1], [0, 2]]), sp.eye(2)
    L_w = (G.T * (X18 * Ws)).trace()
    c["q18.a"] = mat(grad_matrix(L_w, Ws))
    L_x = (G.T * (Xs * W18)).trace()
    c["q18.b"] = mat(grad_matrix(L_x, Xs))
    c["q19"] = str(int(np.sum(np.array([[1, 2], [3, 4]]) * np.array([[0, 1], [1, 0]]))))

    # VJPs: pick logits that realize the stated softmax, then differentiate L = <g, y>
    xs = sp.symbols("s0:3", real=True)
    at = {xs[0]: sp.log(2), xs[1]: 0, xs[2]: 0}  # softmax = (1/2, 1/4, 1/4)
    y = softmax(xs)
    assert [sp.simplify(v.subs(at)) for v in y] == [
        sp.Rational(1, 2),
        sp.Rational(1, 4),
        sp.Rational(1, 4),
    ]
    L21 = y[0]
    c["q21"] = vec([sp.diff(L21, v).subs(at) for v in xs])
    u = sp.symbols("u0:2", real=True)
    y2 = softmax(u)
    at2 = {u[0]: 0, u[1]: sp.log(3)}  # softmax = (1/4, 3/4)
    c["q22"] = mat(
        [[sp.diff(y2[i], u[j]).subs(at2) for j in range(2)] for i in range(2)]
    )
    ls = [sp.log(v) for v in y]
    c["q23"] = vec([sp.diff(ls[1], v).subs(at) for v in xs])
    at24 = {xs[0]: 0, xs[1]: sp.log(2), xs[2]: sp.log(2)}
    L24 = -sp.log(y[1])
    c["q24.a"] = vec([sp.diff(L24, v).subs(at24) for v in xs])
    c["q24.b"] = str(sp.simplify(L24.subs(at24)))
    kept = [p for p in range(4) if p != 2]  # position 2 carries ignore_index
    c["q25"] = str(sp.Rational(1, len(kept)))

    r1, r2 = sp.symbols("r1 r2", positive=True)
    w1, w2, g1, g2 = sp.symbols("w1 w2 g1 g2", real=True)
    rms = sp.sqrt((r1**2 + r2**2) / 2)
    yr = [w1 * r1 / rms, w2 * r2 / rms]
    Lr = g1 * yr[0] + g2 * yr[1]
    dx = [sp.diff(Lr, v) for v in (r1, r2)]
    at26 = {r1: 3, r2: 4, w1: 1, w2: 1, g1: 1, g2: 0}
    c["q26.a"] = vec([d.subs(at26) for d in dx])
    c["q26.b"] = str(sp.simplify(r1 * dx[0] + r2 * dx[1]) == 0).lower()

    l0, l1, l2 = sp.symbols("l0 l1 l2", real=True)
    lx = [l0, l1, l2]
    mu = sum(lx) / 3
    sig = sp.sqrt(sum((v - mu) ** 2 for v in lx) / 3)
    yl = [(v - mu) / sig for v in lx]
    at27 = {l0: 0, l1: 1, l2: 2}
    c["q27"] = vec([sp.diff(yl[0], v).subs(at27) for v in lx])

    c["q29.a"] = str(10**6)  # one JVP per input direction
    c["q29.b"] = "1"  # one VJP per output
    n_in, n_out = 3, 1000
    c["q30"] = "a" if n_in < n_out else "b"
    c["q31"] = str(16 * 32 * 1024 * np.dtype(np.float32).itemsize)
    m, k, n = sp.symbols("m k n", positive=True)
    fwd = 2 * m * k * n  # Y = X W
    bwd = (
        2 * m * n * k + 2 * k * m * n
    )  # dX = G W^T (m x n by n x k), dW = X^T G (k x m by m x n)
    c["q32.a"] = str(bwd)
    c["q32.b"] = str(sp.simplify((fwd + bwd) / fwd))

    # SDPA with symbols, then differentiate L = <dO, O>
    q0, q1_, k0, k1, v0, v1 = sp.symbols("q0 q1 k0 k1 v0 v1", real=True)
    Qm, Km, Vm = sp.Matrix([q0, q1_]), sp.Matrix([k0, k1]), sp.Matrix([v0, v1])
    S = Qm * Km.T / sp.sqrt(1)
    P = sp.Matrix(
        2, 2, lambda i, j: sp.exp(S[i, j]) / sum(sp.exp(S[i, kk]) for kk in range(2))
    )
    O = P * Vm
    at33 = {q0: 1, q1_: 0, k0: 0, k1: 1, v0: 1, v1: 3}
    c["q33.a"] = mat(O.subs(at33))
    Lo = O[0, 0]  # dO = (1, 0)
    c["q33.b"] = mat(sp.Matrix([sp.diff(Lo, v).subs(at33) for v in (v0, v1)]))
    c["q33.c"] = mat(sp.Matrix([sp.diff(Lo, v).subs(at33) for v in (q0, q1_)]))
    return confirm("S-M08", c)


if __name__ == "__main__":
    sys.exit(main())
