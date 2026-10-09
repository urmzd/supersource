"""Course tests for rt.01 (Python side): python/tinyllm/ffi/libtinyllm.py.

`ss check rt.01` builds libtinyllm from your abi.c plus a stub of every unit
you have not started and points TINYLLM_LIB at it. Some tests also compile a
tiny throwaway library of their own (a few lines of C, below) to put the
loader in situations the real library never produces: a wrong ABI version, a
missing symbol, a status code from the future.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tinyllm.ffi.libtinyllm import (
    ABI_VERSION,
    STATUS,
    STATUS_NAMES,
    TL_EUNSUPPORTED,
    AbiMismatch,
    Lib,
    TlError,
    f32_ptr,
    library_path,
    load,
    signatures,
)

# A stand-in library: tl_abi_version returns VERSION, tl_matmul_f32 behaves
# like the stub `ss start` writes (sets the error slot, returns
# TL_EUNSUPPORTED), tl_demo_status returns any code it is given, and
# tl_demo_add is a plain function a test declares by hand.
FAKE = r"""
#include <stdint.h>
#include <string.h>
static char err[256];
static void set_err(const char *m) { strncpy(err, m, sizeof err - 1); }
uint32_t tl_abi_version(void) { return VERSION; }
const char *tl_last_error(void) { return err; }
const char *tl_status_str(int32_t s) { return s == 0 ? "TL_OK" : "TL_UNKNOWN"; }
#ifdef WITH_MATMUL
int32_t tl_matmul_f32(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K,
                      int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta, int trans_b, void *tp) {
    set_err("unimplemented: M03.1");
    return 9;
}
#endif
int32_t tl_demo_status(int32_t s) { set_err("tl_demo_status: as asked"); return s; }
int32_t tl_demo_add(int32_t a, int32_t b) { return a + b; }
"""


def build_fake(tmp: Path, version: int = 1, with_matmul: bool = True) -> str:
    """Compile FAKE into a shared library under tmp and return its path."""
    if shutil.which("cc") is None:
        pytest.fail("cc is not on PATH (ss doctor)")
    src = tmp / "fake.c"
    src.write_text(FAKE)
    ext = "dylib" if platform.system() == "Darwin" else "so"
    out = tmp / f"libfake_v{version}_{int(with_matmul)}.{ext}"
    flags = ["-dynamiclib"] if ext == "dylib" else ["-shared", "-fPIC"]
    defs = [f"-DVERSION={version}"] + (["-DWITH_MATMUL"] if with_matmul else [])
    subprocess.run(
        ["cc", "-std=c11", "-O0", *flags, *defs, str(src), "-o", str(out)], check=True
    )
    return str(out)


def test_load_checks_abi_version():
    # WHY: the tracer's very first cross-language call. Python loads the
    #      library the harness built from your abi.c and asks its version;
    #      everything in Pass 1 that touches C goes through this door.
    # KIND: unit, smoke
    # CATCHES: s01
    # CHAPTER: rt.01 section 3
    lib = load()
    assert lib.abi_version() == ABI_VERSION == 1
    assert lib.path == library_path()


def test_status_names_match_the_library():
    # WHY: Python's table of names and C's tl_status_str must agree for every
    #      code, or an exception and a log line name different errors.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: rt.01 section 2
    lib = load()
    assert sorted(STATUS_NAMES) == list(range(11))
    for code, name in STATUS_NAMES.items():
        assert lib.status_str(code) == name
    assert lib.status_str(-7) == "TL_UNKNOWN"


def test_refuses_a_mismatched_abi_version(tmp_path):
    # WHY: a binding written for ABI 1 that loads an ABI 2 library calls
    #      functions whose signatures may have changed, which corrupts memory
    #      silently. Refusing at load time turns that into one clear error.
    # KIND: conformance
    # CATCHES: s10
    # CHAPTER: rt.01 section 2
    path = build_fake(tmp_path, version=2)
    with pytest.raises(AbiMismatch) as e:
        Lib(path)
    assert (e.value.found, e.value.expected) == (2, 1)


def test_stubbed_call_raises_with_the_message(tmp_path):
    # WHY: a library built with stub units must load, and calling a stub must
    #      raise TlError carrying TL_EUNSUPPORTED and tl_last_error's text, so
    #      "unimplemented: M03.1" reaches the person reading the traceback.
    # KIND: conformance
    # CATCHES: s11, s12
    # CHAPTER: rt.01 section 4
    lib = Lib(build_fake(tmp_path))
    a = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(TlError) as e:
        lib.tl_matmul_f32(
            f32_ptr(a), f32_ptr(a), f32_ptr(a), 1, 1, 1, 1, 1, 1, 1.0, 0.0, 0, None
        )
    assert e.value.status == TL_EUNSUPPORTED
    assert e.value.name == "TL_EUNSUPPORTED"
    assert e.value.fn == "tl_matmul_f32"
    assert "unimplemented: M03.1" in e.value.message
    assert isinstance(e.value, RuntimeError)


def test_binds_symbols_lazily(tmp_path):
    # WHY: a library without a symbol (a unit built by an older ABI, or one
    #      you have not written) must still load; only touching the missing
    #      name fails. Binding the whole table at load breaks every partial
    #      build.
    # KIND: conformance
    # CATCHES: s13
    # CHAPTER: rt.01 section 2
    lib = Lib(build_fake(tmp_path, with_matmul=False))
    assert lib.abi_version() == 1
    with pytest.raises(AttributeError):
        lib.tl_matmul_f32


def test_unknown_status_is_an_error(tmp_path):
    # WHY: c/ABI.md rule 5: a binding maps a status it does not know to an
    #      error. Treating only the codes it knows as failures would let a
    #      newer library's failure pass as success.
    # KIND: boundary
    # CATCHES: s14
    # CHAPTER: rt.01 section 5, Pitfalls
    lib = Lib(build_fake(tmp_path))
    lib.declare("tl_demo_status", STATUS, [ctypes.c_int32])
    assert lib.tl_demo_status(0) == 0
    with pytest.raises(TlError) as e:
        lib.tl_demo_status(42)
    assert (e.value.status, e.value.name) == (42, "TL_UNKNOWN")
    with pytest.raises(TlError) as e:
        lib.tl_demo_status(-3)
    assert e.value.name == "TL_UNKNOWN"


def test_declare_binds_a_new_symbol(tmp_path):
    # WHY: later headers (softmax, attention, ...) add functions the v0 table
    #      does not list; declare is how their Python callers bind them. An
    #      undeclared name raises instead of guessing ctypes' default int
    #      signature, which truncates 64-bit pointers.
    # KIND: unit
    # CATCHES: s15
    # CHAPTER: rt.01 section 5, Pitfalls
    lib = Lib(build_fake(tmp_path))
    with pytest.raises(AttributeError):
        lib.tl_demo_add
    lib.declare("tl_demo_add", ctypes.c_int32, [ctypes.c_int32, ctypes.c_int32])
    assert lib.tl_demo_add(2, 3) == 5
    assert lib.tl_demo_add(-2_000_000_000, -1) == -2_000_000_001


def test_signatures_cover_the_v0_header():
    # WHY: the table is the header transcribed into ctypes types; one wrong
    #      width (an int where the header says int64_t) shifts every argument
    #      after it on some platforms.
    # KIND: unit
    # CATCHES: s16
    # CHAPTER: rt.01 section 4
    sigs = signatures()
    assert {"tl_abi_version", "tl_status_str", "tl_last_error", "tl_matmul_f32"} <= set(
        sigs
    )
    restype, argtypes = sigs["tl_matmul_f32"]
    assert restype == STATUS
    f32p = ctypes.POINTER(ctypes.c_float)
    assert argtypes[:3] == [f32p, f32p, f32p]
    assert argtypes[3:9] == [ctypes.c_int64] * 6
    assert argtypes[9:12] == [ctypes.c_float, ctypes.c_float, ctypes.c_int]
    assert len(argtypes) == 13
    assert sigs["tl_abi_version"][0] == ctypes.c_uint32
    assert sigs["tl_status_str"] == (ctypes.c_char_p, [ctypes.c_int32])


def test_load_caches_one_lib_per_path():
    # WHY: L0.0 calls load() once per matmul; reloading would re-run the ABI
    #      check (and dlopen) on every call. One Lib per resolved path.
    # KIND: unit
    # CATCHES: s17
    # CHAPTER: rt.01 section 4
    assert load() is load()
    assert load(os.environ["TINYLLM_LIB"]) is load()


def test_library_path_prefers_the_argument(monkeypatch):
    # WHY: the harness points every check at its own build with TINYLLM_LIB,
    #      and an explicit path must still win, so a test or a tool can load a
    #      specific file.
    # KIND: unit
    # CATCHES: s18
    # CHAPTER: rt.01 section 4
    monkeypatch.setenv("TINYLLM_LIB", "/from/env/libtinyllm.so")
    assert library_path() == "/from/env/libtinyllm.so"
    assert library_path("/explicit/libtinyllm.so") == "/explicit/libtinyllm.so"


def _loader_copy(root: Path):
    """Import a copy of YOUR libtinyllm.py placed at root/python/tinyllm/ffi/,
    so its default path (<repo>/c/build/...) points under root."""
    import tinyllm.ffi.libtinyllm as mine

    dest = root / "python" / "tinyllm" / "ffi" / "libtinyllm.py"
    dest.parent.mkdir(parents=True)
    dest.write_text(Path(mine.__file__).read_text())
    spec = importlib.util.spec_from_file_location("ss_loader_copy", dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_library_path_falls_back_then_raises(tmp_path, monkeypatch):
    # WHY: with no argument and no TINYLLM_LIB the loader looks for the
    #      library your c/Makefile builds, and when that is missing too it
    #      must raise FileNotFoundError, not hand ctypes a path that fails
    #      later with an unhelpful OSError.
    # KIND: boundary
    # CHAPTER: rt.01 section 4
    monkeypatch.delenv("TINYLLM_LIB", raising=False)
    mod = _loader_copy(tmp_path)
    with pytest.raises(FileNotFoundError):
        mod.library_path()
    ext = "dylib" if platform.system() == "Darwin" else "so"
    built = tmp_path / "c" / "build" / f"libtinyllm.{ext}"
    built.parent.mkdir(parents=True)
    built.write_bytes(b"")
    assert Path(mod.library_path()).resolve() == built.resolve()


def test_f32_ptr_checks_dtype_and_stride():
    # WHY: ctypes passes whatever address it is given. A float64 array or a
    #      column view handed to a float* kernel reads the wrong bytes with no
    #      error; f32_ptr is the one place that refuses them.
    # KIND: boundary
    # CATCHES: s19
    # CHAPTER: rt.01 section 5, Pitfalls
    with pytest.raises(TypeError):
        f32_ptr(np.zeros(3, dtype=np.float64))
    with pytest.raises(ValueError):
        f32_ptr(np.zeros((4, 4), dtype=np.float32)[:, ::2])
    assert f32_ptr(None) is None
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    p = f32_ptr(x[1:])  # a row slice keeps unit stride in the last axis
    assert [p[i] for i in range(3)] == [3.0, 4.0, 5.0]
    # The pointer is the array's own memory, not a temporary copy: C writes
    # through it (an output matrix) and the caller must see them.
    p[0] = -1.0
    assert x[1, 0] == -1.0
    assert ctypes.cast(f32_ptr(x), ctypes.c_void_p).value == x.ctypes.data


def test_exports_only_tl_symbols():
    # WHY: c/ABI.md rule 7. Every exported name shares one process-wide
    #      namespace with Python, Rust, and every other library loaded; a
    #      helper exported as `scale` can be silently replaced by another
    #      library's `scale`. Helpers are static, so only tl_ names remain.
    # KIND: conformance
    # CATCHES: s20
    # CHAPTER: rt.01 section 2
    if shutil.which("nm") is None:
        pytest.fail("nm is not on PATH (it ships with the C toolchain)")
    lib = os.environ["TINYLLM_LIB"]
    args = (
        ["nm", "-g", "-U", lib]
        if platform.system() == "Darwin"
        else ["nm", "-D", "--defined-only", lib]
    )
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-2] not in ("U", "w", "v") and parts[-2].isupper():
            name = parts[-1]
            if platform.system() == "Darwin" and name.startswith("_"):
                name = name[1:]  # Mach-O prefixes every C symbol with an underscore
            names.append(name)
    assert names, f"nm found no exported symbols in {lib}"
    bad = sorted(n for n in names if not n.startswith("tl_"))
    assert not bad, f"exported without the tl_ prefix: {bad} (make helpers static)"
