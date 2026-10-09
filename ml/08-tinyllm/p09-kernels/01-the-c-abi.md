<!-- ss:module rt.01 -->
# The C ABI: status codes, the error slot, the allocator hook, and a lazy ctypes loader

## Overview

| | |
|---|---|
| **Module** | `rt.01` · build · C and Python · Pass 1 · 3 to 4 h |
| **You build** | `c/src/runtime/abi.c`: `tl_abi_version`, `tl_status_str`, `tl_set_allocator`, `tl_alloc`, `tl_free` and the default allocator (the error slot is given); `python/tinyllm/ffi/libtinyllm.py`: `load`, `Lib`, `TlError`, `AbiMismatch`, `signatures`, `library_path`, `f32_ptr` |
| **Contract** | [`course/contracts/c/include/tinyllm/abi.h`](../../../course/contracts/c/include/tinyllm/abi.h) · [`course/contracts/py/tinyllm/ffi/libtinyllm.pyi`](../../../course/contracts/py/tinyllm/ffi/libtinyllm.pyi) · rules: [`c/ABI.md`](../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/rt.01/`: `test_abi.c` (C, under ASan and UBSan) and `test_loader.py` (Python) (what they check: section 4) |
| **Needs** | nothing in code. Reading: `lang.03` (pointers, linkage, ctypes) and `lang.01` (numpy buffers) |
| **Used by** | `M03.1` reports errors through the slot and is called through the loader · `L0.0` computes its logits through `load().tl_matmul_f32` · `L10.0` links `libtinyllm` from Rust and checks `tl_abi_version` · later every `tl_*` unit, `L9.7`, `L10.1` |
| **Milestone** | `MS-P1` (the tracer: every layer is yours and runs end to end) |
| **Optional depth** | Ulrich Drepper, *How To Write Shared Libraries* (free, sections 1 and 2); the System V AMD64 and AAPCS64 calling conventions; the Python `ctypes` documentation, "Return types" and "errcheck" |

## Key Takeaways

- An **ABI** is the binary agreement between separately compiled code: symbol names, calling convention, type sizes, struct layouts. Python and Rust never read your header; they depend on this agreement holding byte for byte (`test_signatures_cover_the_v0_header`).
- **A failing C function returns a positive `tl_status`** (an `int32_t`, never an enum type) **and first writes a message into a thread-local error slot**, so the caller learns both what kind of failure and which argument (`test_stubbed_call_raises_with_the_message`).
- **Every allocation goes through one hook** you can replace, which is how the course counts leaks on macOS and fails the n-th allocation on purpose (`alloc_goes_through_the_hook`).
- **The loader refuses a library whose ABI version differs** and **binds each symbol on first use**, so a partial library still loads and only the missing call fails (`test_refuses_a_mismatched_abi_version`, `test_binds_symbols_lazily`).
- **Only `tl_` names are exported**; every helper is `static` (`test_exports_only_tl_symbols`).

## How to work this chapter

```bash
ss start rt.01              # stubs abi.c and libtinyllm.py into your repo
ss tests rt.01              # read the test catalog first: rung R0, you write no tests here
ss check rt.01              # exit code is the verdict
ss diff  rt.01              # after passing: your code against the reference
```

`ss check` compiles your `abi.c` twice: into an ASan and UBSan test binary with `test_abi.c`, and into an `-O2` shared library (together with a stub of every C unit you have not started) whose path it puts in `TINYLLM_LIB` for `test_loader.py`.

---

## 1. Why now

Your system now has two languages and no way for them to talk. Python (`L0.0`) is about to compute a model's logits with a matrix multiply written in C (`M03.1`), and your Rust engine (`L10.0`) will call the same C function. Today, if that C function is handed a bad dimension it can only crash the Python process or return garbage, because C has no exceptions. If the library on disk was built from an older header, ctypes will happily call it with the wrong argument layout. If the function leaks a buffer, nothing on macOS will tell you. And until you write `M03.1`, `ss` links a stub in its place, so Python must be able to load a library where half the functions say "not written yet" and report that clearly instead of segfaulting. This module fixes the rules every C unit in the course follows (`c/ABI.md`), implements the few functions that carry them, and writes the one Python module through which all Python code reaches C.

## 2. Principles

Notation:

| Symbol | Meaning | Type / shape |
|---|---|---|
| `tl_status` | the result of a fallible function: `TL_OK` (0) or a positive error code | `int32_t` |
| `TL_ABI_VERSION` | the version of the binary agreement; 1 in this course | integer constant |
| $n$ | bytes requested from `tl_alloc` | `size_t`, $n > 0$ |
| $a$ | the requested alignment: the returned address is a multiple of $a$ | `size_t`, a power of two |
| $\mathrm{roundup}(n, a)$ | the smallest multiple of $a$ that is $\ge n$; for a power of two, `(n + a - 1) & ~(a - 1)` | `size_t` |
| `&`, `~` | bitwise AND, bitwise NOT | |

### 2.1 API and ABI

An **API** is what a programmer reads: the names and types in a header. An **ABI** (application binary interface) is what the machine code relies on once compiled: each function's **symbol name**, the **calling convention** (which registers carry which argument; on arm64 the first eight integer or pointer arguments go in `x0` to `x7` and floats in `v0` to `v7`), the **size and layout** of every type that crosses, and how return values come back. Python's ctypes and Rust's `extern "C"` blocks never see `tinyllm.h`. They are a second, hand-written statement of the same ABI. Change `int64_t M` to `int M` in the header and recompile only the C side, and the API looks fine while every call from Python passes the wrong bits. So the ABI is a contract with three copies (the header, the ctypes table, the Rust declarations), and the version number in 2.5 is how a copy notices it is out of date.

### 2.2 Status codes, not enums, not `-1`

Every fallible function returns a `tl_status`: 0 for success, a small positive code for a kind of failure (`TL_EINVAL` bad argument, `TL_ENOMEM` allocation failed, `TL_EUNSUPPORTED` not built, the full list is in `abi.h`). Results go out through pointer parameters, so the return value is always free for the status: `tl_status tl_x_create(..., tl_x **out)`.

The type is `typedef int32_t tl_status` with named constants, **never a C `enum` type**, for two reasons. A C enum's size is chosen by the compiler (the standard only says "an integer type that can hold the values"), so it is not a stable binary type. And Rust, which sees the code as a `#[repr(C)]` enum if you declare it that way, has undefined behavior when the integer holds a value with no variant, which is exactly what happens when an older binding meets a newer library. As a plain `int32_t`, an unknown value is just a number, and every binding maps unknown numbers to an error (c/ABI.md rule 5). `tl_status_str` turns a code into its constant's name for messages, and must cope with any `int32_t` a caller hands it.

### 2.3 The error slot

A status says **what kind** of failure; a person debugging needs **which**: "`tl_matmul_f32`: lda < K". So before returning a failure, a function writes a message into the **error slot** with `tl_set_last_error`, and the caller reads it with `tl_last_error()` before making another `tl_` call. This is the `errno` pattern of the C library, with text.

The slot is declared `_Thread_local`: every thread gets its own copy of the variable. Your Rust engine runs kernels on several threads; with one shared slot, thread A's failure message could be overwritten by thread B before A reads it. The setter **copies** the message into the slot (the caller's string may live in a stack buffer that is gone after it returns), treats `NULL` as the empty string, and truncates to `TL_LAST_ERROR_CAP - 1` bytes plus the terminating zero. This block of `abi.c` is **given** and sits outside the solution markers, for a practical reason: when `ss start` stubs a unit, each stubbed function calls `tl_set_last_error("unimplemented: <id>")`, so the setter must work before you have written anything.

### 2.4 The allocator hook

The library never calls `malloc` directly. Every allocation goes through `tl_alloc(n, a)`, which calls whatever **hook** is installed: a struct of two function pointers and a `user` pointer passed back to them unchanged.

```c
typedef struct { void *(*alloc)(void *user, size_t n, size_t align);
                 void  (*free)(void *user, void *p); void *user; } tl_allocator;
```

One seam gives three abilities. **Counting**: the test harness installs an allocator that counts live blocks and fails any test that ends with more than it started with; that is the leak check on macOS, where clang has no LeakSanitizer. **Fault injection**: the same allocator returns NULL on the n-th call, so a test can walk every failure path of a constructor (later modules check that each one returns `TL_ENOMEM` and leaks nothing). **Replacement**: `rt.02`'s arena and an engine's memory pool can sit behind the same calls. `tl_set_allocator(&a)` installs a **copy** of `*a` (callers build the struct on the stack), refuses a hook missing either function (`TL_EINVAL`), and `tl_set_allocator(NULL)` restores the default.

`tl_alloc` checks its arguments before the hook sees them: $n = 0$ is an error (C's `malloc(0)` may return NULL or a unique pointer, and that ambiguity is not allowed across the boundary), and $a$ must be a power of two, which has exactly one bit set, so $a \mathbin{\&} (a - 1) = 0$. The **default hook** uses C11 `aligned_alloc(a, size)`, which requires `size` to be a multiple of `a` (and, in some C libraries, `a` to be at least `sizeof(void *)`), so it rounds: $\mathrm{size} = \mathrm{roundup}(n, a)$. `tl_free(NULL)` does nothing, like `free(NULL)`; any other pointer goes back to the hook it came from (change the hook only while nothing allocated by the old one is alive).

### 2.5 Versioning

`tl_abi_version()` returns `TL_ABI_VERSION`. A binding written against version 1 calls it first and **refuses** any other answer: loading a version 2 library whose functions may take different arguments would corrupt memory silently, while refusing turns it into one clear error at load time. Changing the version is a migration with its own module (`craft.13` shows the pattern for the KV format). A stubbed `tl_abi_version` returns 0, so a library built before you finish this module is refused with a message that says so.

### 2.6 What a library exports

Every non-`static` function and file-scope variable in a C file becomes an exported **symbol** of the shared library, and all libraries loaded in one process share one namespace of symbol names. A helper you call `scale` could be bound to someone else's `scale`. So: every public name starts with `tl_`, and everything else is `static` (internal linkage, `lang.03` 2.6). `nm -g` lists the exported symbols; on macOS each C name carries a leading `_` (Mach-O convention), which the check removes before comparing.

### 2.7 Loading from Python: lazy, checked, and declared

ctypes gives you a library handle (`dlopen`) and functions looked up by name (`dlsym`), but it does not know any signature (`lang.03` 2.8). The loader wraps that in four rules:

1. **Find the library**: an explicit path, else the `TINYLLM_LIB` environment variable (which `ss` sets to the library it built for the check), else `<repo>/c/build/libtinyllm.dylib` (`.so` on Linux), which your own `c/Makefile` produces. Nothing found is a `FileNotFoundError` that names all three.
2. **Check the version** at load (2.5): `AbiMismatch`, carrying the found and expected numbers.
3. **Declare every signature**, once, in a table: `signatures()` maps each Python-callable function of `tinyllm.h` v0 to its `(restype, argtypes)`, written from the header (`int64_t` is `c_int64`, `float *` is `POINTER(c_float)`, a `tl_pool *` is `c_void_p`). Functions returning `tl_status` use the marker `STATUS`. Headers that arrive later are added with `Lib.declare`. An undeclared name raises `AttributeError` instead of falling back to ctypes' guess (`int` everywhere), which truncates 64-bit pointers.
4. **Bind lazily and check statuses**: a function is looked up only the first time its attribute is touched, then cached. A library that lacks a symbol (a unit from an older build) still loads; only the missing call fails. For a `STATUS` function the loader installs a ctypes `errcheck` hook, which ctypes calls with the raw return value: `TL_OK` passes through, anything else, including a code this binding has never heard of, raises `TlError(fn, status, last_error())`. The message must be read inside that hook, before any other `tl_` call can overwrite the slot.

`f32_ptr(a)` is the one place a numpy array becomes a `float *`. ctypes will pass any address it is given, so this function refuses a dtype other than float32 (`TypeError`) and a last axis without unit stride (`ValueError`); a row-strided view is fine, because its row stride becomes the kernel's leading dimension (`M03.1`).

## 3. Worked example by hand

**An aligned allocation.** `tl_alloc(10, 64)` with the default hook:

| Step | Computation | Result |
|---|---|---|
| size check | $n = 10 > 0$ | ok |
| power of two | $64 \mathbin{\&} 63$ = `0b1000000 & 0b0111111` = 0 | ok |
| round the size | $(10 + 63) \mathbin{\&} \lnot 63$ = `0b1001001 & ...11000000` = `0b1000000` | 64 |
| allocate | `aligned_alloc(64, 64)` | an address such as `0x600000c04040`; its low 6 bits are 0, so it is divisible by 64 |

With $a = 3$: $3 \mathbin{\&} 2 = 2 \ne 0$, so `tl_alloc` sets the slot to "tl_alloc: align must be a power of two" and returns NULL without calling the hook. The test `alloc_alignment_by_hand` repeats the first row for every power of two from 1 to 4096.

**Status names.** `tl_status_str(9)` is `"TL_EUNSUPPORTED"`. `tl_status_str(11)` and `tl_status_str(-1)` are `"TL_UNKNOWN"`: 11 is past the last code, and -1 would be index $2^{64} - 1$ if you looked names up in an array by `s` (pitfall 1).

**A stubbed call, end to end.** Before you write `M03.1`, `ss` links this stub for it:

```c
tl_status tl_matmul_f32(/* ... */) {
    tl_set_last_error("unimplemented: M03.1");
    return TL_EUNSUPPORTED;
}
```

From Python, `load().tl_matmul_f32(...)` goes through these steps: ctypes converts the 13 arguments by the declared argtypes, calls the C function, gets 9 back, and calls the errcheck hook with 9. The hook sees 9 ≠ 0, reads `tl_last_error()` = `"unimplemented: M03.1"`, and raises `TlError` whose text is `tl_matmul_f32: TL_EUNSUPPORTED (9): unimplemented: M03.1`. Exactly this is `test_stubbed_call_raises_with_the_message`, against a small stand-in library the test compiles itself.

## 4. The interface

C (`tinyllm/abi.h`; the error slot functions are given in `abi.c`):

```c
#define TL_ABI_VERSION 1
uint32_t    tl_abi_version(void);
typedef int32_t tl_status;            /* TL_OK = 0, TL_EINVAL = 1, ..., TL_EIO = 10 */
const char *tl_status_str(tl_status s);        /* "TL_OK" ... "TL_EIO", else "TL_UNKNOWN"; never NULL */
const char *tl_last_error(void);               /* given; "" before any error */
void        tl_set_last_error(const char *msg);/* given; copies, NULL is "", truncates */
tl_status   tl_set_allocator(const tl_allocator *a);  /* copy; NULL restores the default; TL_EINVAL for a half hook */
void       *tl_alloc(size_t n, size_t align);  /* NULL + error slot: n == 0, align not a power of two, hook failed */
void        tl_free(void *p);                  /* NULL is a no-op */
```

Python (`tinyllm/ffi/libtinyllm.pyi`):

```python
ABI_VERSION = 1; TL_OK = 0; ...; STATUS_NAMES: dict[int, str]; STATUS  # restype marker
class TlError(RuntimeError):      fn, status, name, message;  __init__(fn, status, message)
class AbiMismatch(RuntimeError):  path, found, expected;      __init__(path, found, expected)
def signatures() -> dict[str, tuple[restype, list[argtype]]]
def library_path(path=None) -> str          # path, else $TINYLLM_LIB, else <repo>/c/build/libtinyllm.*
class Lib:
    def __init__(self, path=None)           # loads, checks the version, binds nothing else
    def declare(self, name, restype, argtypes)
    def abi_version(self) -> int; def status_str(self, status) -> str; def last_error(self) -> str
    def __getattr__(self, name)             # lazily bound ctypes function; STATUS functions raise TlError
def load(path=None) -> Lib                  # one Lib per resolved path
def f32_ptr(a) -> POINTER(c_float) | None
```

Your own `c/Makefile` (entry-point territory: the course ships none) builds `c/build/libtinyllm.dylib` (or `.so`) from one object per unit, which is where the loader looks when `TINYLLM_LIB` is unset, and `SANITIZE=1` adds `-fsanitize=address,undefined` for your own test runs (`lang.02`). Add `-std=c11 -Wall -Wextra -Werror -pedantic`: the contract headers compile clean under them, and so should your units.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `alloc_alignment_by_hand` | unit | section 3: aligned for 1 to 4096 with the default hook | vector loads in `L9.*` kernels |
| `abi_version_matches_header` | unit, smoke | `tl_abi_version() == TL_ABI_VERSION == 1` | every binding's first call |
| `status_str_names_every_code` | unit | each code 0 to 10 gives its constant's name | messages name the error |
| `status_str_unknown_codes` | boundary | 11, -1, `INT32_MAX`, `INT32_MIN` give `"TL_UNKNOWN"` without reading out of bounds | codes from a newer ABI |
| `error_slot_copies_and_truncates` | boundary, regression | the given slot copies, maps NULL to "", truncates | every unit's error path |
| `alloc_goes_through_the_hook` | fault | `tl_alloc` and `tl_free` reach the installed hook (counted) | the leak check of every later C module |
| `alloc_rejects_zero_and_bad_alignment` | boundary | NULL, message set, hook never called | defined behavior at the boundary |
| `alloc_failure_sets_the_error` | fault | a failing hook gives NULL and a message | `TL_ENOMEM` paths in constructors |
| `set_allocator_rejects_half_a_hook` | boundary | `TL_EINVAL`; the old hook stays | no hook that can allocate but not free |
| `set_allocator_copies_the_struct` | fault | the hook keeps working after the caller's struct changes | stack-built hooks |
| `set_allocator_null_restores_the_default` | unit | after `tl_set_allocator(NULL)` the old hook is never called again | a test's or an arena's hook outliving its owner |
| `free_null_is_a_no_op` | boundary | `tl_free(NULL)` never reaches the hook | cleanup paths |
| `test_load_checks_abi_version` | unit, smoke | `load()` opens `TINYLLM_LIB` and reports version 1 | `L0.0`'s first call into C |
| `test_status_names_match_the_library` | unit | Python's `STATUS_NAMES` equals C's `tl_status_str` | one name per error everywhere |
| `test_refuses_a_mismatched_abi_version` | conformance | an ABI 2 library raises `AbiMismatch(found=2, expected=1)` | rule 12 of c/ABI.md |
| `test_stubbed_call_raises_with_the_message` | conformance | section 3's stubbed call raises `TlError` with the slot text | partial libraries in Pass 1 |
| `test_binds_symbols_lazily` | conformance | a library without `tl_matmul_f32` loads; touching it raises `AttributeError` | rule 11 of c/ABI.md |
| `test_unknown_status_is_an_error` | boundary | status 42 or -3 raises with name `TL_UNKNOWN` | rule 5 of c/ABI.md |
| `test_declare_binds_a_new_symbol` | unit | `declare` binds a new function; undeclared names raise | headers after v0 (`L9.7`) |
| `test_signatures_cover_the_v0_header` | unit | `tl_matmul_f32` has 13 argtypes with the header's widths | `M03.1` and `L0.0` |
| `test_load_caches_one_lib_per_path` | unit | `load()` returns the same `Lib` | no reload per call |
| `test_library_path_prefers_the_argument` | unit | explicit path, then `TINYLLM_LIB` | tools that load a specific file |
| `test_library_path_falls_back_then_raises` | boundary | with neither, `<repo>/c/build/libtinyllm.*`, else `FileNotFoundError` | a clear error when nothing is built |
| `test_f32_ptr_checks_dtype_and_stride` | boundary | float64 and column views are refused; the pointer is the array's own memory | `M03.1`'s Python tests and `L0.0` |
| `test_exports_only_tl_symbols` | conformance | `nm` on the built library shows only `tl_` names | rule 7 of c/ABI.md |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `tl_status_str` as `names[s]` | a negative or future code reads outside the array (ASan stops it) | `status_str_unknown_codes` (mutant `m001`) |
| 2. keeping a pointer to the caller's `tl_allocator` | the next `tl_alloc` calls through a dead stack frame | `set_allocator_copies_the_struct` (mutant `s07`) |
| 3. `tl_free` calling `free()` instead of the hook | the counting allocator never sees the free: every test reports a leak, and a non-malloc hook is corrupted | `alloc_goes_through_the_hook` (mutant `s04`) |
| 4. the default hook calling `malloc(n)` | 16-byte alignment only; a 64-byte request is misaligned | `alloc_alignment_by_hand` (mutant `s05`) |
| 5. a helper without `static` | an exported name outside `tl_` can collide with another library's | `test_exports_only_tl_symbols` (mutant `s20`) |
| 6. checking only the codes the binding knows | a newer library's failure passes as success | `test_unknown_status_is_an_error` (mutant `s14`) |
| 7. falling back to ctypes' default signature for an undeclared name | 64-bit pointers and sizes truncated to 32 bits, with no error | `test_declare_binds_a_new_symbol` (mutant `s15`) |
| 8. binding the whole table when the library loads | any library missing one symbol fails to load at all | `test_binds_symbols_lazily` (mutant `s13`) |
| 9. reading `tl_last_error` after another `tl_` call | the message belongs to the wrong call, or is empty | `test_stubbed_call_raises_with_the_message` (mutant `s12`) |
| 10. declaring `int64_t` dimensions as `c_int` | works for small matrices on some machines, garbage on others | `test_signatures_cover_the_v0_header` (mutant `s16`) |
| 11. `f32_ptr` taking the pointer of `np.array(a)` (a copy) | the pointer is to a temporary copy: C's writes to an output matrix vanish, and the copy may be freed before C reads it | `test_f32_ptr_checks_dtype_and_stride` |
| 12. `tl_set_allocator(NULL)` that keeps the old hook | memory keeps coming from a hook whose owner is gone | `set_allocator_null_restores_the_default` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.03` | pointers, structs, linkage, `nm`, sanitizers, and ctypes `argtypes` (reading) |
| Back | `lang.01` | numpy arrays as row-major float32 buffers that `f32_ptr` hands to C (reading) |
| Forward | `M03.1` | `tl_matmul_f32` returns `TL_EINVAL` with a message through the slot; its Python tests call it through `load()` |
| Forward | `L0.0` | `BigramLM.logits` calls `load().tl_matmul_f32(f32_ptr(...), ...)`; a failure arrives as `TlError` |
| Forward | `L10.0` | the Rust engine links `libtinyllm`, declares the same functions in an `extern "C"` block, and refuses a library whose `tl_abi_version` is not 1 |
| Forward | `L9.7` | the Python C backend `declare`s every kernel header that arrives after v0 |

If you skip this module, `ss check M03.1` and `ss check L0.0` stop with `BLOCKED ... needs rt.01`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_status` + error slot | SQLite result codes and `sqlite3_errmsg`; CUDA `cudaGetLastError` | extended codes; per-connection rather than per-thread messages | the SQLite C API, "Result and Error Codes" |
| allocator hook | `ggml` backend buffers, jemalloc arenas, CPython's `PyMem_SetAllocator` | per-domain allocators and debug hooks that detect buffer overruns | CPython `Objects/obmalloc.c`; ggml `ggml-alloc.c` |
| `TL_ABI_VERSION` | GNU symbol versioning, `SONAME` major versions | several versions of one symbol in one library, chosen at link time | Drepper, *How To Write Shared Libraries*, section 3 |
| ctypes loader | cffi (API mode), pybind11, nanobind, PyO3 | signatures checked at build time from the header or the types, no hand-written table | the cffi docs; `L1.5` uses PyO3 in this course |
| `nm` export check | `-fvisibility=hidden` with explicit export attributes, linker version scripts | exports chosen per symbol without making functions `static` | GCC "Visibility" wiki page |
