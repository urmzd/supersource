"""Batch and chunk invariance assertions (DESIGN 2.4, 5.11; L9.1, L10.3).

A batch-invariant kernel gives every row the same bits whether it is computed
alone or inside a batch of any size, at any position: its reduction order
depends only on the row, never on M or on the row's neighbours. A
chunk-invariant computation (chunked prefill, online softmax over tiles)
gives the same bits for every chunk size. Both are exact properties, so the
comparison is bitwise (NaN payloads and signed zeros included), and a
failure names the first differing element with its ULP distance.

    from _lib.invariance import assert_batch_invariant, assert_chunk_invariant

    assert_batch_invariant(lambda xb: matmul(xb, w), x)            # x: [B, K]
    assert_chunk_invariant(lambda c: prefill(tokens, chunk=c), [1, 3, 16, None])
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIZES = (1, 2, 3, 4, 5, 7, 8, 16)


def _bits(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        return a.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])
    return a


def ulps(a: float, b: float, dtype=np.float32) -> int:
    ia = int(
        np.array(a, dtype=dtype).view(
            {4: np.int32, 8: np.int64, 2: np.int16}[np.dtype(dtype).itemsize]
        )
    )
    ib = int(
        np.array(b, dtype=dtype).view(
            {4: np.int32, 8: np.int64, 2: np.int16}[np.dtype(dtype).itemsize]
        )
    )
    return abs(ia - ib)


def assert_bits_equal(got, want, what: str = "") -> None:
    g, w = np.asarray(got), np.asarray(want)
    assert g.shape == w.shape, f"{what}: shape {g.shape} != {w.shape}"
    assert g.dtype == w.dtype, f"{what}: dtype {g.dtype} != {w.dtype}"
    bg, bw = _bits(g), _bits(w)
    diff = np.flatnonzero(bg.reshape(-1) != bw.reshape(-1))
    if diff.size:
        i = int(diff[0])
        idx = np.unravel_index(i, g.shape) if g.shape else ()
        x, y = g.reshape(-1)[i], w.reshape(-1)[i]
        extra = f", {ulps(x, y, g.dtype)} ulp" if g.dtype.kind == "f" else ""
        raise AssertionError(
            f"{what}: {diff.size} element(s) differ in their bits; first at {tuple(int(t) for t in idx)}: {x!r} vs {y!r}{extra}"
        )


def assert_batch_invariant(fn, x, sizes=DEFAULT_SIZES, positions: bool = True) -> None:
    """fn maps a batch (rows on axis 0) to outputs (rows on axis 0). Every row
    computed alone must equal, bit for bit, that row inside every batch of
    each size in `sizes` (that fits), at every offset when `positions`."""
    x = np.asarray(x)
    n = x.shape[0]
    alone = [np.asarray(fn(x[i : i + 1]))[0] for i in range(n)]
    for b in sizes:
        if b > n:
            continue
        offsets = range(0, n - b + 1) if positions else [0]
        for off in offsets:
            out = np.asarray(fn(x[off : off + b]))
            assert out.shape[0] == b, (
                f"fn returned {out.shape[0]} rows for a batch of {b}"
            )
            for j in range(b):
                assert_bits_equal(
                    out[j],
                    alone[off + j],
                    f"row {off + j} inside a batch of {b} at position {j}",
                )


def assert_chunk_invariant(fn, chunks, reference=None) -> None:
    """fn(chunk) computes the same thing processed in pieces of `chunk` (None:
    all at once). Every chunk size must give the bits of `reference`, or of
    the first entry of `chunks` when no reference is given."""
    chunks = list(chunks)
    want = np.asarray(reference) if reference is not None else np.asarray(fn(chunks[0]))
    for c in chunks if reference is not None else chunks[1:]:
        assert_bits_equal(np.asarray(fn(c)), want, f"chunk size {c}")
