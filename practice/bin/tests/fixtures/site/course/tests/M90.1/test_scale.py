from _lib.close import assert_close
from tinyllm.demo.scale import scale


def test_hand_example():
    # WHY: the chapter's worked example: 2.5 * [1, -2, 0.5] = [2.5, -5, 1.25].
    # KIND: unit
    assert_close(scale([1.0, -2.0, 0.5], 2.5), [2.5, -5.0, 1.25])


def test_input_not_modified():
    # WHY: callers reuse their vector after scaling it; a function that
    #      scales in place silently corrupts the caller's data.
    # KIND: property
    xs = [1.0, 2.0, 3.0]
    scale(xs, 3.0)
    assert xs == [1.0, 2.0, 3.0]
