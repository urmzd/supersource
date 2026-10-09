"""Course tests for the optional ALiBi part of L7.4 (DESIGN D31).

alibi_bias is in the contract, but ALiBi has no call site in the final
system, so it is optional: if your alibi_bias still raises
NotImplementedError, these tests are skipped and do not decide the verdict.
Implement it and they run.

The worked example (chapter section 3): 2 heads, slopes 1/16 and 1/256,
two queries that are the last 2 of 3 keys. bias[0] = -(1/16) * distance:
query 1 (key index 1) sees distances (1, 0, -1), query 2 sees (2, 1, 0).

The golden fixture (course/fixtures/L7.4/rope_scaling_hf.json, key
alibi_bloom) holds BLOOM's build_alibi_tensor, slope_h * j, for several head
counts.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from _lib.close import assert_close
from tinyllm.modern.ctxext import alibi_bias

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.4", "rope_scaling_hf.json")


def bias(n: int, tq: int, tk: int) -> np.ndarray:
    try:
        return alibi_bias(n, tq, tk)
    except NotImplementedError:
        pytest.skip("alibi_bias is the optional part of L7.4 (D31): not implemented")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def test_hand_example_alibi():
    # WHY: the chapter's worked example: the bias falls by the head's slope
    #      for every position of distance, 0 on the query's own position.
    # KIND: unit
    # CATCHES: s08, s09
    # CHAPTER: L7.4 section 3, Worked example by hand
    b = bias(2, 2, 3)
    assert b.dtype == np.float32 and b.shape == (2, 2, 3)
    d = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
    assert_close(b, np.stack([-d / 16, -d / 256]), dtype="float32")


def test_golden_bloom_same_softmax():
    # WHY: BLOOM adds slope_h * j (key position) instead of -slope_h *
    #      distance; per row the two differ by a constant, so after the
    #      causal mask the softmax weights are identical.
    # KIND: golden
    # CATCHES: s08
    # CHAPTER: L7.4 section 2.6, ALiBi
    with open(FIX) as f:
        rows = json.load(f)["alibi_bloom"]
    for r in rows:
        n = r["n_heads"]
        ours = bias(n, 7, 7).astype(np.float64)
        bloom = np.asarray(r["bias"], dtype=np.float64)  # [n, 7]
        causal = np.tril(np.ones((7, 7), dtype=bool))
        a = softmax(np.where(causal, ours, -np.inf))
        b = softmax(np.where(causal, np.broadcast_to(bloom[:, None, :], (n, 7, 7)), -np.inf))
        assert_close(a, b, rtol=1e-5, atol=1e-6, msg=f"{n} heads")


def test_decode_row_equals_last_row_of_prefill():
    # WHY: in decoding one query is the last of Tk keys: its bias row must
    #      equal the last row of the full Tk x Tk prefill bias.
    # KIND: property
    # CATCHES: s09
    # CHAPTER: L7.4 section 2.6, ALiBi
    full = bias(6, 9, 9)
    assert np.array_equal(bias(6, 1, 9)[:, 0], full[:, -1])
    assert np.array_equal(bias(6, 3, 9), full[:, -3:])


def test_validation_alibi():
    # WHY: more queries than keys, or zero sizes, are wiring bugs.
    # KIND: boundary
    # CATCHES: m04
    # CHAPTER: L7.4 section 4, The interface
    bias(1, 1, 1)
    for args in ((2, 3, 2), (0, 1, 1), (2, 0, 1)):
        with pytest.raises(ValueError):
            alibi_bias(*args)
