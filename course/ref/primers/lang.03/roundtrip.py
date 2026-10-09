"""primers/lang.03/roundtrip.py: the Python half of the ctypes round trip.

ctypes knows nothing about roundtrip.h. Every function's argument and
return types must be declared here by hand; anything left undeclared is
passed and returned as a C int, which silently truncates 64-bit values.
"""

from __future__ import annotations

import ctypes


class Pair(ctypes.Structure):
    """Mirror of rt_pair in roundtrip.h. ctypes inserts the same padding the
    C compiler does, as long as the fields are listed in the same order with
    the same types."""

    _fields_ = [("id", ctypes.c_int32), ("score", ctypes.c_double)]


def load(path: str) -> ctypes.CDLL:
    """Open the shared library at `path` and declare every function's
    argtypes and restype from roundtrip.h."""
    # SOLUTION-BEGIN lang.03
    lib = ctypes.CDLL(path)
    lib.rt_sum_i32.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    lib.rt_sum_i32.restype = ctypes.c_int64
    lib.rt_scale_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float]
    lib.rt_scale_f32.restype = None
    lib.rt_best_pair.argtypes = [ctypes.POINTER(Pair), ctypes.c_size_t, ctypes.POINTER(Pair)]
    lib.rt_best_pair.restype = ctypes.c_int
    lib.rt_pair_size.argtypes = []
    lib.rt_pair_size.restype = ctypes.c_size_t
    lib.rt_pair_score_offset.argtypes = []
    lib.rt_pair_score_offset.restype = ctypes.c_size_t
    return lib
    # SOLUTION-END


def sum_i32(lib: ctypes.CDLL, xs: list[int]) -> int:
    """Sum xs in C: copy them into a C int32 array and call rt_sum_i32."""
    # SOLUTION-BEGIN lang.03
    arr = (ctypes.c_int32 * len(xs))(*xs)
    return int(lib.rt_sum_i32(arr, len(xs)))
    # SOLUTION-END


def scale_f32(lib: ctypes.CDLL, xs: list[float], a: float) -> list[float]:
    """Scale xs by a in C and return the values C wrote back."""
    # SOLUTION-BEGIN lang.03
    arr = (ctypes.c_float * len(xs))(*xs)
    lib.rt_scale_f32(arr, len(xs), a)
    return list(arr)
    # SOLUTION-END


def best_pair(lib: ctypes.CDLL, pairs: list[tuple[int, float]]) -> tuple[int, float] | None:
    """The (id, score) with the highest score, ties to the first; None when
    pairs is empty. C fills a Pair that Python passes by pointer."""
    # SOLUTION-BEGIN lang.03
    arr = (Pair * len(pairs))(*[Pair(i, s) for i, s in pairs])
    out = Pair()
    if lib.rt_best_pair(arr, len(pairs), ctypes.byref(out)) != 0:
        return None
    return (out.id, out.score)
    # SOLUTION-END
