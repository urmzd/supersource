"""tinyllm.ffi.libtinyllm (rt.01): load libtinyllm through ctypes.

Contract: contracts/py/tinyllm/ffi/libtinyllm.pyi. Rules: contracts/c/ABI.md.

    lib = load()                       # $TINYLLM_LIB, else c/build/libtinyllm.*
    lib.abi_version()                  # 1
    lib.tl_matmul_f32(f32_ptr(A), ...) # bound on first use; returns 0 or raises TlError

Symbols are bound lazily, on first attribute access, so a library that lacks
a symbol still loads and only a call to the missing one fails.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Callable

ABI_VERSION = 1  # the TL_ABI_VERSION of contracts/c/include/tinyllm/abi.h

TL_OK = 0
TL_EINVAL = 1
TL_ENOMEM = 2
TL_ESHAPE = 3
TL_EDTYPE = 4
TL_EFULL = 5
TL_ENOTFOUND = 6
TL_EFORMAT = 7
TL_EBUSY = 8
TL_EUNSUPPORTED = 9
TL_EIO = 10

STATUS_NAMES = {
    TL_OK: "TL_OK", TL_EINVAL: "TL_EINVAL", TL_ENOMEM: "TL_ENOMEM", TL_ESHAPE: "TL_ESHAPE",
    TL_EDTYPE: "TL_EDTYPE", TL_EFULL: "TL_EFULL", TL_ENOTFOUND: "TL_ENOTFOUND",
    TL_EFORMAT: "TL_EFORMAT", TL_EBUSY: "TL_EBUSY", TL_EUNSUPPORTED: "TL_EUNSUPPORTED",
    TL_EIO: "TL_EIO",
}

# The restype of a function that returns tl_status. A function declared with
# it is a ctypes function whose errcheck returns TL_OK (0) and raises TlError
# for any other value, unknown codes included.
STATUS = "tl_status"


class TlError(RuntimeError):
    """A tl_ function returned a status other than TL_OK."""

    def __init__(self, fn: str, status: int, message: str) -> None:
        # SOLUTION-BEGIN rt.01
        self.fn = fn
        self.status = status
        self.name = STATUS_NAMES.get(status, "TL_UNKNOWN")
        self.message = message
        super().__init__(f"{fn}: {self.name} ({status}): {message}")
        # SOLUTION-END


class AbiMismatch(RuntimeError):
    """The library's tl_abi_version() differs from ABI_VERSION."""

    def __init__(self, path: str, found: int, expected: int) -> None:
        # SOLUTION-BEGIN rt.01
        self.path = path
        self.found = found
        self.expected = expected
        hint = " (a stubbed tl_abi_version returns 0: finish rt.01)" if found == 0 else ""
        super().__init__(f"{path}: ABI version {found}, this binding needs {expected}{hint}")
        # SOLUTION-END


def signatures() -> dict[str, tuple[Any, list[Any]]]:
    """(restype, argtypes) of every function in tinyllm.h v0 that Python may
    call. The allocator functions are left out on purpose: they take C
    function pointers, and no callback crosses the boundary (c/ABI.md)."""
    # SOLUTION-BEGIN rt.01
    f32p = ctypes.POINTER(ctypes.c_float)
    i64 = ctypes.c_int64
    return {
        "tl_abi_version": (ctypes.c_uint32, []),
        "tl_status_str": (ctypes.c_char_p, [ctypes.c_int32]),
        "tl_last_error": (ctypes.c_char_p, []),
        "tl_matmul_f32": (STATUS, [f32p, f32p, f32p, i64, i64, i64, i64, i64, i64,
                                   ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]),
    }
    # SOLUTION-END


def library_path(path: str | None = None) -> str:
    """The library to load: `path`, else $TINYLLM_LIB, else
    <repo>/c/build/libtinyllm.{dylib,so} next to this package. Raises
    FileNotFoundError naming all three when none is given or found."""
    # SOLUTION-BEGIN rt.01
    if path:
        return str(path)
    env = os.environ.get("TINYLLM_LIB")
    if env:
        return env
    ext = "dylib" if sys.platform == "darwin" else "so"
    built = Path(__file__).resolve().parents[3] / "c" / "build" / f"libtinyllm.{ext}"
    if built.is_file():
        return str(built)
    raise FileNotFoundError(f"no libtinyllm: pass a path, set TINYLLM_LIB, or build {built}")
    # SOLUTION-END


class Lib:
    """One loaded libtinyllm. Attribute access binds a declared tl_ symbol."""

    def __init__(self, path: str | None = None) -> None:
        # SOLUTION-BEGIN rt.01
        self.path = library_path(path)
        self._dll = ctypes.CDLL(self.path)
        self._sigs = dict(signatures())
        found = self.abi_version()
        if found != ABI_VERSION:
            raise AbiMismatch(self.path, found, ABI_VERSION)
        # SOLUTION-END

    def declare(self, name: str, restype: Any, argtypes: list[Any]) -> None:
        """Add or replace the signature of `name`; it binds on next access."""
        # SOLUTION-BEGIN rt.01
        self._sigs[name] = (restype, list(argtypes))
        self.__dict__.pop(name, None)
        # SOLUTION-END

    def abi_version(self) -> int:
        # SOLUTION-BEGIN rt.01
        return int(self.tl_abi_version())
        # SOLUTION-END

    def status_str(self, status: int) -> str:
        # SOLUTION-BEGIN rt.01
        return self.tl_status_str(status).decode()
        # SOLUTION-END

    def last_error(self) -> str:
        # SOLUTION-BEGIN rt.01
        return (self.tl_last_error() or b"").decode("utf-8", "replace")
        # SOLUTION-END

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # SOLUTION-BEGIN rt.01
        # Only reached when normal lookup fails: bind once, then cache in
        # __dict__ so the next access never comes back here.
        if name.startswith("_") or name not in self.__dict__.get("_sigs", {}):
            raise AttributeError(f"{name} is not a declared libtinyllm function (see Lib.declare)")
        restype, argtypes = self._sigs[name]
        cfn = self._dll[name]  # dlsym happens here (a fresh object); AttributeError if missing
        cfn.argtypes = argtypes
        if restype == STATUS:
            cfn.restype = ctypes.c_int32

            def check(status: int, func: Any, args: tuple) -> int:
                # ctypes calls this with the raw return value. Read the error
                # slot now: the next tl_ call on this thread may overwrite it.
                if status != TL_OK:
                    raise TlError(name, status, self.last_error())
                return status

            cfn.errcheck = check
        else:
            cfn.restype = restype
        self.__dict__[name] = cfn
        return cfn
        # SOLUTION-END


_LOADED: dict[str, Lib] = {}


def load(path: str | None = None) -> Lib:
    """The Lib for `path` (see library_path), loaded once per resolved path."""
    # SOLUTION-BEGIN rt.01
    key = os.path.realpath(library_path(path))
    if key not in _LOADED:
        _LOADED[key] = Lib(key)
    return _LOADED[key]
    # SOLUTION-END


def f32_ptr(a: Any) -> Any:
    """A float* to the first element of a float32 numpy array (None gives
    NULL). The last axis must have unit stride; rows may be strided, which
    the caller passes as a leading dimension."""
    # SOLUTION-BEGIN rt.01
    if a is None:
        return None
    if str(a.dtype) != "float32":
        raise TypeError(f"f32_ptr needs a float32 array, got {a.dtype}")
    if a.ndim and a.strides[-1] != 4 and a.shape[-1] > 1:
        raise ValueError(f"f32_ptr needs unit stride in the last axis, got strides {a.strides}")
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # SOLUTION-END
