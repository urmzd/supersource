"""Course tests for ds.04 (Python side): your tl_topk_f32, called through your
rt.01 loader, against numpy and against your own L8.1 sampler's top-k.

The C side under sanitizers is test_topk.c. Inputs come from the frozen
PCG32 (course/tests/_lib) and the L8.1 golden logits
(course/fixtures/L8.1/sampler_golden.json), the same rows the Rust sampler
(L10.1) is held to.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import STATUS, TL_EINVAL, TlError, f32_ptr, load
from tinyllm.infer.sample import SamplingParams, process_logits

SEED = int(os.environ.get("SS_SEED", "0"))
FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L8.1" / "sampler_golden.json"


def lib():
    lb = load()
    lb.declare(
        "tl_topk_f32",
        STATUS,
        [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ],
    )
    return lb


def topk(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=np.float32)
    idx = np.full(max(k, 1), -1, dtype=np.int32)
    val = np.zeros(max(k, 1), dtype=np.float32)
    lib().tl_topk_f32(
        f32_ptr(x),
        x.size,
        k,
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        f32_ptr(val),
    )
    return idx[:k], val[:k]


def by_rank(x: np.ndarray) -> np.ndarray:
    """Every index ordered by (value desc, index asc): the spec's order."""
    return np.lexsort((np.arange(x.size), -x.astype(np.float64)))


def test_hand_example_through_ctypes():
    # WHY: the chapter's worked example across the boundary, exactly as the
    #      Rust sampler will call it: x = [1, 3, 2, 3, -1], k = 3 keeps ids
    #      1, 3, 2 in that order (the tie between the two 3s goes to id 1).
    # KIND: unit, smoke
    # CATCHES: s05, s06
    # CHAPTER: ds.04 section 3
    idx, val = topk(np.array([1, 3, 2, 3, -1], dtype=np.float32), 3)
    assert idx.tolist() == [1, 3, 2]
    assert val.tolist() == [3.0, 3.0, 2.0]


def test_matches_numpy_full_sort():
    # WHY: a vocabulary-sized row (V = 49152 like SmolLM2) with values on a
    #      coarse grid, so ties are frequent, against numpy's stable sort by
    #      (value desc, index asc), for k from 1 to past the heap's first
    #      few levels. A mistake in sift-down shows up only on large heaps.
    # KIND: differential
    # CATCHES: s01, s02, s06, m01
    # CHAPTER: ds.04 section 4
    rng = PCG32(SEED, seq=41)
    x = (np.floor(rng.uniform_array((49152,), -8.0, 8.0) * 64) / 64).astype(np.float32)
    order = by_rank(x)
    for k in (1, 2, 40, 1000):
        idx, val = topk(x, k)
        assert idx.tolist() == order[:k].tolist(), f"k={k}"
        assert np.array_equal(val, x[order[:k]])


@pytest.mark.parametrize("k", [1, 5, 12, 40])
def test_matches_your_l8_1_top_k_on_fixture_logits(k):
    # WHY: P6, Python is the specification. The kept set of your L8.1
    #      process_logits(top_k=k) (spec step 6) must equal what the C
    #      kernel keeps, on the golden logits the Rust sampler is held to,
    #      including the `masked` row with -inf entries (which the sampler
    #      never keeps, and the kernel ranks last).
    # KIND: differential
    # CATCHES: s01, s06, s09
    # CHAPTER: ds.04 section 6
    doc = json.loads(FIX.read_text())
    for case in doc["cases"]:
        x = np.array([float(v) for v in case["logits"]], dtype=np.float32)
        kept = process_logits(x, SamplingParams(temperature=1.0, top_k=k))
        want = set(np.flatnonzero(np.isfinite(kept)).tolist())
        idx, val = topk(x, k)
        got = {int(i) for i, v in zip(idx, val) if np.isfinite(v)}
        assert got == want, f"{case['name']} k={k}"


def test_nan_raises_einval_through_the_loader():
    # WHY: a NaN logit means a broken forward pass upstream; the kernel
    #      refuses it, and the refusal must reach Python as TlError with
    #      TL_EINVAL and the C message, with the output buffers untouched.
    # KIND: boundary
    # CATCHES: s03, s08
    # CHAPTER: ds.04 section 4
    x = np.array([0.5, np.nan, 2.0], dtype=np.float32)
    idx = np.full(2, 7, dtype=np.int32)
    val = np.full(2, 7.0, dtype=np.float32)
    with pytest.raises(TlError) as e:
        lib().tl_topk_f32(
            f32_ptr(x),
            3,
            2,
            idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            f32_ptr(val),
        )
    assert e.value.status == TL_EINVAL
    assert "tl_topk_f32" in e.value.message
    assert idx.tolist() == [7, 7] and val.tolist() == [7.0, 7.0]
