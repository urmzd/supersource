<!-- ss:module rt.91 -->
# Demo float sum and owned buffer

## Overview

| | |
|---|---|
| **Module** | `rt.91` · build · C · Pass 1 · 20 min |
| **You build** | `c/src/runtime/demo.c`: `tl_demo_sum_f32` and the `tl_demo_buf` constructor pair |
| **Contract** | [`contracts/c/include/tinyllm/demo.h`](../course/contracts/c/include/tinyllm/demo.h) |
| **Tests** | `course/tests/rt.91/` (what they check: section 4) |
| **Needs** | `rt.90` |
| **Used by** | `M90.2` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- Summing $n$ floats in a float accumulator loses about $\sqrt{n}$ ulps; a double accumulator keeps the float result exact for small inputs.

## How to work this chapter

```bash
ss start rt.91
ss tests rt.91
ss check rt.91
```

---

## 1. Why now

The Python normalizer in M90.2 needs a native sum it can call through ctypes, and its buffer must not leak.

## 2. Principles

Summing $n$ floats in a float accumulator loses about $\sqrt{n}$ ulps; a double accumulator keeps the float result exact for small inputs.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_i$ | the i-th input | `float` |
| $n$ | number of inputs | `int64_t` |

## 3. Worked example by hand

x = [1, 2, 3]: acc = 0 + 1 = 1, 1 + 2 = 3, 3 + 3 = 6, so out = 6.0f.

## 4. The interface

```c
tl_status tl_demo_sum_f32(const float *x, int64_t n, float *out);
tl_status tl_demo_buf_create(int64_t n, tl_demo_buf **out);
float *tl_demo_buf_data(tl_demo_buf *b);
void tl_demo_buf_destroy(tl_demo_buf *b);
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit | the chapter's worked example | M90.2 |
| `null_out_is_einval` | boundary | a NULL out pointer is a caller bug that must be reported, not a crash | M90.2 |
| `create_destroy_no_leak` | fault | every block create takes, destroy gives back (the counting allocator checks) | M90.2 |
| `alloc_failure_is_clean` | fault | when the second allocation fails, the first must be freed | M90.2 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| returning ENOMEM without freeing the first block | a leak on the failure path | `alloc_failure_is_clean` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.90` | demo abi: status codes, error slot, allocator hook |
| Forward | `M90.2` | l1 normalize through the c sum |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| demo float sum and owned buffer | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
