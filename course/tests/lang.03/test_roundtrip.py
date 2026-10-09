"""Course tests for the lang.03 ctypes round trip: YOUR primers/lang.03/
roundtrip.c (built by the check into a shared library at $LANG03_LIB) called
through YOUR roundtrip.py.
"""

from __future__ import annotations

import ctypes
import os

import pytest
import roundtrip  # primers/lang.03/roundtrip.py, put on sys.path by the check


@pytest.fixture(scope="module")
def lib():
    return roundtrip.load(os.environ["LANG03_LIB"])


def test_sum_by_hand(lib):
    # WHY: the chapter's first round trip: a Python list becomes a C int32
    #      array, C adds it up, the int64 result comes back as a Python int.
    # KIND: unit
    assert roundtrip.sum_i32(lib, [1, 2, 3]) == 6
    assert roundtrip.sum_i32(lib, []) == 0


def test_sum_needs_the_declared_restype(lib):
    # WHY: ctypes assumes every undeclared function returns a C int (32 bits).
    #      4e9 does not fit, so a missing `restype = c_int64` returns a wrapped
    #      negative number with no error; so does adding in int32 inside C.
    # KIND: boundary
    assert roundtrip.sum_i32(lib, [2_000_000_000, 2_000_000_000]) == 4_000_000_000
    assert roundtrip.sum_i32(lib, [-2_147_483_648, -1]) == -2_147_483_649


def test_scale_writes_through_the_pointer(lib):
    # WHY: C receives the address of Python's buffer, not a copy, so its
    #      writes are visible after the call. Powers of two keep it exact.
    # KIND: unit
    assert roundtrip.scale_f32(lib, [1.0, -2.5, 0.0], 2.0) == [2.0, -5.0, 0.0]
    assert roundtrip.scale_f32(lib, [], 3.0) == []


def test_struct_layout_matches_c(lib):
    # WHY: the chapter's layout worked by hand: id at offset 0, 4 bytes of
    #      padding, score at offset 8, 16 bytes in all. Python and C must
    #      agree byte for byte or every field read after the first is garbage.
    # KIND: conformance
    assert lib.rt_pair_size() == 16 == ctypes.sizeof(roundtrip.Pair)
    assert lib.rt_pair_score_offset() == 8 == roundtrip.Pair.score.offset


def test_best_pair_fills_an_out_parameter(lib):
    # WHY: the C convention of returning a status and writing the result
    #      through a pointer, which every tl_*_create in the course uses.
    #      Ties keep the lower index; an empty array is an error, not a read
    #      of ps[0].
    # KIND: boundary
    assert roundtrip.best_pair(lib, [(1, 0.5), (2, 0.9), (3, 0.9), (4, -1.0)]) == (
        2,
        0.9,
    )
    assert roundtrip.best_pair(lib, [(7, -3.0)]) == (7, -3.0)
    assert roundtrip.best_pair(lib, []) is None
