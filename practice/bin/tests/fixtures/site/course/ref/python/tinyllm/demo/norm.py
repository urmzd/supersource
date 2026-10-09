"""L1 normalization through the C sum kernel (fixture module M90.2)."""

import ctypes
import os

from tinyllm.demo.scale import scale


def _lib() -> ctypes.CDLL:
    """Given: the ctypes binding (the real course's rt.01 loader plays this role)."""
    lib = ctypes.CDLL(os.environ["TINYLLM_LIB"])
    f = lib.tl_demo_sum_f32
    f.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.POINTER(ctypes.c_float)]
    f.restype = ctypes.c_int32
    return lib


def l1_normalize(xs: list[float]) -> list[float]:
    """Divide xs by sum(|x|) so the absolute values sum to 1."""
    # SOLUTION-BEGIN M90.2
    n = len(xs)
    arr = (ctypes.c_float * n)(*[abs(x) for x in xs])
    total = ctypes.c_float()
    status = _lib().tl_demo_sum_f32(arr, n, ctypes.byref(total))
    if status != 0:
        raise ValueError(f"tl_demo_sum_f32 failed with status {status}")
    if total.value == 0.0:
        raise ValueError("l1_normalize: all-zero vector has no direction")
    return scale(xs, 1.0 / total.value)
    # SOLUTION-END
