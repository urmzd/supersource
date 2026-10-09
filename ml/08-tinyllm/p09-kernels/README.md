# Part 9: Kernels in C

`libtinyllm`: the runtime (ABI, arena, thread pool) and the kernels (matmul, softmax, FlashAttention, paged attention, quantized matmul, elementwise) that the Python backend and the Rust engine call through one C ABI. Pass 1 has one chapter here: the ABI itself, the status codes, the error slot, the allocator hook, and the lazy ctypes loader every later kernel is reached through. The kernels arrive in Pass 6.

**Course passes**: 1 (rt.01, gate [MS-P1](../../../paths/course-p01-tracer/milestone.md)), 6 (rt.02, rt.03, L9.1 to L9.7, gate MS-P6).

**Build contract**: your `c/Makefile` writes `c/build/libtinyllm.{a,dylib,so}`; `SANITIZE=1` adds ASan and UBSan. The harness builds its own objects for tests, sanitized for the C tests and unsanitized for ctypes and Rust. The rules are in [`c/ABI.md`](../../../course/contracts/c/ABI.md).

<!-- ss:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `rt.01` | [The C ABI: status codes, the error slot, the allocator hook, and a lazy ctypes loader](01-the-c-abi.md) | build | 1 |
<!-- /ss:chapters -->
