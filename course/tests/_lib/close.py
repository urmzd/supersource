"""Closeness assertions with the course tolerances (DESIGN 5.11).

    abs(actual - expected) <= atol + rtol * abs(expected)      elementwise

| dtype   | rtol    | atol  |
|---------|---------|-------|
| float64 | 1e-10   | 1e-12 |
| float32 | 1e-5    | 1e-6  |
| float16 | 1e-3    | 1e-4  |
| bfloat16| 1.6e-2  | 1e-3  |

A reduction over K terms accumulates up to about sqrt(K) rounding errors of
size eps, so `assert_close_bounded(..., k=K)` scales both tolerances by
sqrt(K). These are the formulas M09.3 teaches; the learner's own copy never
decides another module's verdict.
"""

from __future__ import annotations

import math

import numpy as np

TOL = {
    "float64": (1e-10, 1e-12),
    "float32": (1e-5, 1e-6),
    "float16": (1e-3, 1e-4),
    "bfloat16": (1.6e-2, 1e-3),
}


def tolerances(dtype) -> tuple[float, float]:
    name = np.dtype(dtype).name if not isinstance(dtype, str) else dtype
    if name not in TOL:
        raise ValueError(
            f"no course tolerance for dtype {name!r}; pass rtol and atol explicitly"
        )
    return TOL[name]


def _first_bad(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> str:
    diff = np.abs(a - b)
    allowed = atol + rtol * np.abs(b)
    both_nan = np.isnan(a) & np.isnan(b)
    same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    bad = ~((diff <= allowed) | both_nan | same_inf)
    if not bad.any():
        return ""
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    n = int(bad.sum())
    return (
        f"{n} of {a.size} elements differ; first at index {idx}: "
        f"actual {a[idx]!r}, expected {b[idx]!r}, |diff| {diff[idx]:.3e} > allowed {allowed[idx]:.3e} "
        f"(rtol {rtol:g}, atol {atol:g})"
    )


def assert_close(
    actual,
    expected,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    dtype=None,
    msg: str = "",
) -> None:
    """Assert elementwise closeness. Tolerances default to the dtype's row of
    the table (the expected value's dtype unless `dtype` is given)."""
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}shape {a.shape} != expected {b.shape}"
        )
    if rtol is None or atol is None:
        src = dtype if dtype is not None else np.asarray(expected).dtype
        if not isinstance(src, str):
            dt = np.dtype(src)
            src = dt.name if np.issubdtype(dt, np.floating) else "float64"
        r, t = tolerances(src)
        rtol = r if rtol is None else rtol
        atol = t if atol is None else atol
    bad = _first_bad(a, b, rtol, atol)
    if bad:
        raise AssertionError(f"{msg + ': ' if msg else ''}{bad}")


def assert_close_bounded(
    actual, expected, *, k: int, dtype="float32", msg: str = ""
) -> None:
    """Closeness for a reduction over k terms: tolerances scale by sqrt(k)."""
    rtol, atol = tolerances(dtype)
    s = math.sqrt(max(int(k), 1))
    assert_close(actual, expected, rtol=rtol * s, atol=atol * s, msg=msg)
