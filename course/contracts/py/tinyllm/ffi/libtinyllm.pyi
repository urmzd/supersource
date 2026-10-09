# contracts/py/tinyllm/ffi/libtinyllm.pyi (rt.01): the Python side of the C ABI
# chapter: ml/08-tinyllm/p09-kernels/01-the-c-abi.md
#
# Loads libtinyllm with ctypes. The rules it enforces are in c/ABI.md:
# refuse a library whose tl_abi_version() differs from ABI_VERSION, bind each
# symbol lazily (on first attribute access, so a partial library loads), and
# turn a tl_status other than TL_OK into TlError carrying tl_last_error().
#
#     lib = load()                               # $TINYLLM_LIB, else <repo>/c/build/libtinyllm.*
#     lib.tl_matmul_f32(f32_ptr(A), f32_ptr(B), f32_ptr(C), M, N, K, K, N, N, 1.0, 0.0, 0, None)
from typing import Any

ABI_VERSION: (
    int  # 1, the TL_ABI_VERSION of tinyllm/abi.h this binding was written against
)

TL_OK: int  # 0; the other codes as in tinyllm/abi.h
TL_EINVAL: int
TL_ENOMEM: int
TL_ESHAPE: int
TL_EDTYPE: int
TL_EFULL: int
TL_ENOTFOUND: int
TL_EFORMAT: int
TL_EBUSY: int
TL_EUNSUPPORTED: int
TL_EIO: int
STATUS_NAMES: dict[
    int, str
]  # code -> "TL_OK", ..., equal to tl_status_str for every known code

STATUS: Any  # restype marker for a function that returns tl_status (see Lib.declare)

class TlError(RuntimeError):
    """A tl_ function returned a status other than TL_OK."""

    fn: str  # the C function's name
    status: int  # the raw tl_status value
    name: str  # STATUS_NAMES[status], or "TL_UNKNOWN" for a code this binding does not know
    message: str  # tl_last_error() read right after the failing call

    def __init__(self, fn: str, status: int, message: str) -> None: ...

class AbiMismatch(RuntimeError):
    """The library reports a tl_abi_version() other than ABI_VERSION."""

    path: str
    found: int
    expected: int

    def __init__(self, path: str, found: int, expected: int) -> None: ...

def signatures() -> dict[str, tuple[Any, list[Any]]]:
    """(restype, argtypes) of every Python-callable function in tinyllm.h v0:
    tl_abi_version, tl_status_str, tl_last_error, tl_matmul_f32. A tl_status
    return is written as STATUS. The allocator functions are not listed: they
    take C function pointers, and no callback crosses the boundary."""

def library_path(path: str | None = None) -> str:
    """`path` when given, else $TINYLLM_LIB, else <repo>/c/build/libtinyllm.dylib
    (.so on Linux) when that file exists, where <repo> holds python/tinyllm/.
    FileNotFoundError otherwise."""

class Lib:
    path: str  # the file that was loaded

    def __init__(self, path: str | None = None) -> None:
        """Load the library at library_path(path) and check its ABI version:
        AbiMismatch when tl_abi_version() != ABI_VERSION. Binds nothing else."""

    def declare(self, name: str, restype: Any, argtypes: list[Any]) -> None:
        """Add or replace the signature of `name` (for headers newer than v0).
        The symbol is looked up on the next attribute access, not now."""

    def abi_version(self) -> int: ...
    def status_str(self, status: int) -> str:
        """tl_status_str(status) decoded: "TL_OK" ... "TL_EIO", or "TL_UNKNOWN"."""

    def last_error(self) -> str:
        """tl_last_error() decoded; "" before any error on this thread."""

    def __getattr__(self, name: str) -> Any:
        """The ctypes function for a declared `name`, bound with its argtypes
        and restype on first access and cached. A STATUS function returns 0
        or raises TlError. AttributeError when `name` is not declared, or the
        library does not export it."""

def load(path: str | None = None) -> Lib:
    """The Lib for library_path(path), loaded once per resolved file path."""

def f32_ptr(a: Any) -> Any:
    """ctypes POINTER(c_float) to the first element of float32 numpy array `a`
    (None gives NULL). TypeError for another dtype; ValueError when the last
    axis does not have unit stride (pass row strides as a leading dimension)."""
