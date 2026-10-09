"""lang.01 course tests: the broadcasting worksheet (primers/lang.01/worksheet.py).

Annotated exemplars (DESIGN 5.12): each test says WHY it exists and what
KIND of test it is. The first test is the chapter's worked example.
"""

import itertools

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32

from worksheet import as_c_float32, broadcast_shape, unbroadcast

# Every shape with up to 3 axes whose sizes come from {1, 2, 3}: 40 shapes,
# 1600 ordered pairs. Small enough to enumerate, big enough to hold every
# case of the rule (equal, one side 1, padding, mismatch).
SMALL_SHAPES = [s for n in range(4) for s in itertools.product((1, 2, 3), repeat=n)]


def _numpy_broadcast(a, b):
    try:
        return np.broadcast_shapes(a, b)
    except ValueError:
        return None


def test_broadcast_shape_hand_examples():
    # WHY: the chapter's section 3 table, worked by hand; you and the test
    #      agree on the rule before any code runs.
    # KIND: unit
    # CHAPTER: lang.01 section 3
    assert broadcast_shape((3, 1), (1, 4)) == (3, 4)
    assert broadcast_shape((2, 3), (3,)) == (2, 3)
    assert broadcast_shape((8, 1, 6, 1), (7, 1, 5)) == (8, 7, 6, 5)
    assert broadcast_shape((), (5,)) == (5,)
    assert broadcast_shape((4, 1), (0,)) == (4, 0)


def test_broadcast_shape_aligns_on_the_right():
    # WHY: shapes line up from their LAST axis. Padding on the right instead
    #      turns (3,) + (3, 1) into (3, 1) when numpy says (3, 3), which is
    #      the bias-add bug in every layer that adds a vector to a matrix.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 1
    assert broadcast_shape((3,), (3, 1)) == (3, 3)
    assert broadcast_shape((3, 1), (3,)) == (3, 3)


def test_broadcast_shape_rejects_incompatible_shapes():
    # WHY: two sizes that differ and are both not 1 cannot broadcast. Taking
    #      the max instead hides a shape bug until it corrupts a result.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 2
    with pytest.raises(ValueError):
        broadcast_shape((3,), (4,))
    with pytest.raises(ValueError):
        broadcast_shape((2, 1), (8, 4, 3))
    with pytest.raises(ValueError):
        broadcast_shape((0,), (3,))


def test_broadcast_shape_matches_numpy_on_every_small_pair():
    # WHY: numpy is the oracle; enumerating all 1600 small pairs covers every
    #      branch of the rule, so "you can predict a broadcast's shape" is
    #      proven, not sampled.
    # KIND: differential
    for a, b in itertools.product(SMALL_SHAPES, repeat=2):
        want = _numpy_broadcast(a, b)
        if want is None:
            with pytest.raises(ValueError):
                broadcast_shape(a, b)
        else:
            assert broadcast_shape(a, b) == want, f"{a} with {b}"


def test_unbroadcast_hand_example():
    # WHY: section 3: a (2, 3) gradient of ones summed back to the shapes
    #      that broadcast to it. Each result counts how many copies an
    #      element had.
    # KIND: unit
    # CHAPTER: lang.01 section 3
    g = np.ones((2, 3))
    assert_close(unbroadcast(g, (3,)), np.array([2.0, 2.0, 2.0]))
    assert_close(unbroadcast(g, (2, 1)), np.array([[3.0], [3.0]]))
    assert_close(unbroadcast(g, (1, 3)), np.array([[2.0, 2.0, 2.0]]))
    assert_close(unbroadcast(g, ()), np.array(6.0))
    assert_close(unbroadcast(g, (2, 3)), g)


def test_unbroadcast_keeps_size_one_axes():
    # WHY: a bias of shape (2, 1) needs a gradient of shape (2, 1). Summing
    #      without keepdims returns (2,), and the optimizer update `b -= lr * g`
    #      then broadcasts b up to (2, 2): the parameter silently changes shape.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 3
    out = unbroadcast(np.arange(6.0).reshape(2, 3), (2, 1))
    assert out.shape == (2, 1)
    assert_close(out, np.array([[3.0], [12.0]]))


def test_unbroadcast_sums_leading_axes():
    # WHY: broadcasting (3,) against (4, 2, 3) adds two leading axes; their
    #      copies must be summed too, or the gradient keeps a batch axis.
    # KIND: boundary
    g = np.ones((4, 2, 3))
    out = unbroadcast(g, (3,))
    assert out.shape == (3,)
    assert_close(out, np.full(3, 8.0))
    out = unbroadcast(g, (1, 3))
    assert out.shape == (1, 3)
    assert_close(out, np.full((1, 3), 8.0))


def test_unbroadcast_rejects_a_shape_that_did_not_broadcast():
    # WHY: (2,) never broadcasts to (2, 3) (axes align on the right), so a
    #      gradient for it is a caller bug and must not be invented.
    # KIND: boundary
    with pytest.raises(ValueError):
        unbroadcast(np.ones((2, 3)), (2,))
    with pytest.raises(ValueError):
        unbroadcast(np.ones((3,)), (2, 3))


def test_unbroadcast_is_the_adjoint_of_broadcasting():
    # WHY: the defining law. For every x of shape s and g of the broadcast
    #      shape T: sum(broadcast_to(x, T) * g) == sum(x * unbroadcast(g, s)).
    #      This is exactly what L0.1's backward pass relies on.
    # KIND: property
    rng = PCG32(seed=1)
    for s, t in itertools.product(SMALL_SHAPES, repeat=2):
        big = _numpy_broadcast(s, t)
        if big is None:
            continue
        x = rng.normal_array(s)
        g = rng.normal_array(big)
        lhs = np.sum(np.broadcast_to(x, big) * g)
        rhs = np.sum(x * unbroadcast(g, s))
        assert_close(rhs, lhs, msg=f"x{s} into {big}")


def test_as_c_float32_hand_example():
    # WHY: section 3: a (2, 3) float64 array becomes float32, values rounded
    #      to the nearest float32, laid out row by row.
    # KIND: unit
    # CHAPTER: lang.01 section 3
    x = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
    y = as_c_float32(x)
    assert y.dtype == np.float32
    assert y.flags.c_contiguous
    assert (
        np.frombuffer(y.tobytes(order="A"), dtype=np.float32).tolist()
        == np.float32([0.1, 0.2, 0.3, 1.0, 2.0, 3.0]).tolist()
    )


def test_as_c_float32_does_not_copy_when_it_need_not():
    # WHY: the ctypes loader (rt.01) calls this on every tensor it hands to C.
    #      A copy of an array that is already right doubles memory traffic
    #      for nothing; `np.array(x, dtype=...)` and `x.astype(...)` copy always.
    # KIND: unit
    # CHAPTER: lang.01 section 5, pitfall 4
    x = np.zeros((3, 4), dtype=np.float32)
    y = as_c_float32(x)
    assert np.shares_memory(x, y)


def test_as_c_float32_copies_a_transposed_view():
    # WHY: x.T is a view with swapped strides; its buffer is still x's
    #      row-major bytes. C reading `const float *` with dims (4, 3) would
    #      get the wrong elements. The result must be a fresh row-major copy.
    #      `np.asarray(x, dtype=np.float32)` keeps the view and fails here.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 5
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = as_c_float32(x.T)
    assert y.shape == (4, 3)
    assert y.flags.c_contiguous
    assert not np.shares_memory(x, y)
    assert (
        np.frombuffer(y.tobytes(order="A"), dtype=np.float32).tolist()
        == x.T.ravel().tolist()
    )


def test_as_c_float32_copies_a_strided_slice():
    # WHY: x[:, ::2] skips every other element, so its memory is not one
    #      dense block; C needs the dense copy.
    # KIND: boundary
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = as_c_float32(x[:, ::2])
    assert y.flags.c_contiguous and y.shape == (3, 2)
    assert_close(y, np.array([[0, 2], [4, 6], [8, 10]], dtype=np.float32))
