"""The frozen gradient check (DESIGN 4.0 `G`, D35).

For a scalar function f of arrays x_1..x_n, the central difference

    d f / d x[i]  ~=  (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)

has truncation error O(eps^2) and rounding error O(u / eps); in float64
(u ~ 1.1e-16) eps = 1e-6 balances the two near 1e-10. The analytic gradient
passes when every element satisfies

    |analytic - numeric| <= atol + rtol * |numeric|,   rtol 1e-5, atol 1e-7.

Course tests call this, never the learner's own gradcheck (M04.1).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

EPS, RTOL, ATOL = 1e-6, 1e-5, 1e-7


def numeric_grad(
    f: Callable[..., float], inputs: Sequence[np.ndarray], eps: float = EPS
) -> list[np.ndarray]:
    xs = [np.array(x, dtype=np.float64, copy=True) for x in inputs]
    grads = []
    for k, x in enumerate(xs):
        g = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            orig = x[i]
            x[i] = orig + eps
            hi = float(f(*xs))
            x[i] = orig - eps
            lo = float(f(*xs))
            x[i] = orig
            g[i] = (hi - lo) / (2 * eps)
        grads.append(g)
    return grads


def gradcheck(
    f: Callable[..., float],
    inputs: Sequence[np.ndarray],
    analytic: Sequence[np.ndarray],
    *,
    eps: float = EPS,
    rtol: float = RTOL,
    atol: float = ATOL,
    names: Sequence[str] | None = None,
) -> None:
    """Raise AssertionError naming the first input and element that disagree."""
    if len(inputs) != len(analytic):
        raise AssertionError(
            f"{len(analytic)} analytic gradients for {len(inputs)} inputs"
        )
    numeric = numeric_grad(f, inputs, eps)
    for k, (a, n) in enumerate(zip(analytic, numeric)):
        a = np.asarray(a, dtype=np.float64)
        label = names[k] if names else f"input {k}"
        if a.shape != n.shape:
            raise AssertionError(
                f"{label}: analytic gradient shape {a.shape} != input shape {n.shape}"
            )
        bad = np.abs(a - n) > atol + rtol * np.abs(n)
        if bad.any():
            i = tuple(int(j) for j in np.argwhere(bad)[0])
            raise AssertionError(
                f"{label}: gradient differs at {i}: analytic {a[i]:.10g}, central difference {n[i]:.10g} "
                f"(eps {eps:g}, rtol {rtol:g}, atol {atol:g}; {int(bad.sum())} of {a.size} elements bad)"
            )
