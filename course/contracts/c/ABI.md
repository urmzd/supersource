# The C ABI of `libtinyllm`

`include/tinyllm.h` is an umbrella header over one header per unit in `include/tinyllm/`. One module owns one header and one source file, which is what lets the course swap a single unit for a stub or a reference. These rules hold for every header, now and after later units land.

## Contract v0

Contract v0 holds the Pass 1 units only:

| Header | Unit | Owner |
|---|---|---|
| [`tinyllm/abi.h`](include/tinyllm/abi.h) | `c/src/runtime/abi.c` | `rt.01` |
| [`tinyllm/matmul.h`](include/tinyllm/matmul.h) | `c/src/kernels/matmul.c` | `M03.1` (v0), taken over by `L9.1` |

Every other header (arena, pool, KV pool, data structures, top-k, numerics, softmax, attention, quantized matmul, elementwise) arrives with its unit's batch, as a minor version bump of `contracts/VERSION`.

## Rules

1. **The caller owns every buffer.** No function allocates except a `*_create` constructor, and constructors allocate only through the allocator hook (`tl_alloc`).
2. **Constructors** return a `tl_status` and write the object through an out parameter: `tl_status tl_x_create(..., tl_x **out)`. Each has a matching `tl_x_destroy`, which accepts `NULL`.
3. **No callbacks cross the boundary** except `tl_parallel_for`'s range function (rt.03), which stays inside C.
4. **Shapes and strides are `int64_t`.** Matrices are row-major with an explicit leading dimension (the distance in elements between two rows).
5. **Enums never cross the boundary as C enum types.** `tl_status` and `tl_dtype` are `int32_t` with named constants, and struct fields that carry them are `int32_t`. A binding maps an unknown value to an error.
6. **Errors.** A function that fails returns a positive `tl_status` (or `NULL`) and first sets the thread's error slot with `tl_set_last_error`. Read it with `tl_last_error()` before the next `tl_` call on that thread.
7. **Exports.** `nm` on the library shows only `tl_` symbols, after removing the leading `_` that Mach-O adds. Helpers are `static`.
8. **Layout.** Every public struct's size and field offsets are asserted from Python with `ctypes.sizeof` and `ctypes.offsetof`.
9. **Thread safety.** Kernels are reentrant. Stateful objects (arena, KV pool, map, LRU) are not thread-safe: the caller serializes every call on one object. The error slot is `_Thread_local`. Threads use pthreads only (`<threads.h>` is missing on macOS).
10. **Batch invariance** (from L9.1, L9.3, L9.4). Each output element's reduction uses one fixed order, independent of the batch size, of the row's position in the batch, and of the tile. Row `i` of a product computed with `M = 1` equals row `i` computed with `M = 37` bit for bit.
11. **Lazy binding.** Bindings resolve each symbol on first use. The harness links a stub object for every unit not yet started: each function sets the error slot and returns `TL_EUNSUPPORTED` (or the type's zero value), so a partial library always loads and only the stubbed call fails.
12. **Versioning.** `tl_abi_version()` returns `TL_ABI_VERSION`. Python and Rust bindings refuse a library whose version differs from the one they were written for. Changing it is a migration.
13. **Standard.** Every header compiles on its own under `cc -std=c11 -pedantic -Wall -Werror`.

## Building

The learner's `c/Makefile` produces `c/build/libtinyllm.{a,dylib,so}`; `SANITIZE=1` adds `-fsanitize=address,undefined`. The harness compiles its own objects for tests: one sanitized build for the C tests and one `-O2` shared library for ctypes and Rust. It points the learner's loader at the latter with `TINYLLM_LIB`.
