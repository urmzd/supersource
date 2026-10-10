# Part 9: Kernels in C

`libtinyllm`: the runtime (ABI, arena, thread pool) and the kernels (matmul, softmax, FlashAttention, paged attention, quantized matmul, elementwise) that the Python backend and the Rust engine call through one C ABI. Pass 1 has one chapter here: the ABI itself, the status codes, the error slot, the allocator hook, and the lazy ctypes loader every later kernel is reached through. The kernels arrive in Pass 6.

**Course passes**: 1 (rt.01, gate [MS-P1](../../../paths/course-p01-tracer/milestone.md)), 6 (rt.02, rt.03, L9.1 to L9.7, gate MS-P6).

**Build contract**: your `c/Makefile` writes `c/build/libtinyllm.{a,dylib,so}`; `SANITIZE=1` adds ASan and UBSan. The harness builds its own objects for tests, sanitized for the C tests and unsanitized for ctypes and Rust. The rules are in [`c/ABI.md`](../../../course/contracts/c/ABI.md).

<!-- ss:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `rt.01` | [The C ABI: status codes, the error slot, the allocator hook, and a lazy ctypes loader](01-the-c-abi.md) | build | 1 |
| 2 | `rt.02` | [Arena allocator with marks](02-arena-allocator-with-marks.md) | build | 6 |
| 3 | `rt.03` | [Thread pool and tl_parallel_for](03-thread-pool-and-parallel-for.md) | build | 6 |
| 4 | `L9.1` | [Cache-blocked, packed, batch-invariant matmul in C](04-tiled-batch-invariant-matmul.md) | build | 6 |
| 5 | `L9.2` | [Softmax in C: three-pass and online two-pass](05-softmax-three-pass-and-online.md) | build | 6 |
| 6 | `L9.3` | [FlashAttention forward in C](06-flash-attention-forward.md) | build | 6 |
| 7 | `L9.4` | [Paged attention for decode in C](07-paged-attention-decode.md) | build | 6 |
| 8 | `L9.5` | [Fused int4 and int8 dequantize-matmul in C](08-fused-quantized-matmul.md) | build | 6 |
| 9 | `L9.6` | [Elementwise kernels in C: RMSNorm, RoPE, SiLU-mul, embedding, add, argmax](09-elementwise-kernels.md) | build | 6 |
| 10 | `L9.7` | [The Python C backend: the Llama forward through libtinyllm, with a load-time op check](10-python-c-backend.md) | build | 6 |
<!-- /ss:chapters -->
