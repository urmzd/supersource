# C 01: Dynamic array

**Concepts:** `realloc`, growth factor, generics via macros
**Difficulty:** ⭐⭐

A growable array of `int`, then the same thing made generic with macros. This is
the exercise that teaches why `realloc` is dangerous in a way `malloc` is not:
it can move your buffer, which invalidates every pointer anyone still holds
into it.

## The contract

`vec.h` declares it and `main.c` tests it. You write `vec.c`.

| Function | Does |
|----------|------|
| `vec_init(v)` | Zero-capacity vector, no allocation yet |
| `vec_push(v, x)` | Append, growing if needed. Returns 0 on success, -1 on allocation failure |
| `vec_pop(v, out)` | Remove and return the last element. -1 if empty |
| `vec_get(v, i)` | Element at `i`. Undefined behaviour past `len`, deliberately |
| `vec_reserve(v, n)` | Ensure capacity for at least `n` without changing `len` |
| `vec_free(v)` | Release the buffer and return to the zero state |

## Rules

Standard library only. No leaks: the tests run under whatever sanitizer you
point at them, and `vec_free` must leave the struct safe to `vec_free` again.

## What to notice

**The growth factor is a real decision, not a detail.** Doubling gives
amortised O(1) push. Growing by a constant gives O(n) per push and quadratic
total cost, which you can feel on a million elements. Growing by 1.5 reuses
freed blocks better because the sum of previous allocations eventually exceeds
the next request; doubling never does. glibc, MSVC, and Rust all pick
differently here, and all three have a defensible reason.

**`realloc` can move the buffer.** Any pointer or index-derived pointer you
handed out before a `push` may now dangle. The test suite has a case that
catches exactly this, because the bug is invisible until the one push that
happens to cross a page boundary.

**`realloc(p, 0)` is not `free(p)`.** In C17 it is implementation-defined, and
in C23 it is undefined. Do not rely on it.

## Extending it

Once the `int` version passes, the macro version in `vec.h` (`VEC_DEFINE`) is
the real lesson: C has no generics, so you either write the same code per type,
cast through `void *` and lose type safety, or generate code with the
preprocessor and lose debuggability. Pick one deliberately and know what you
gave up.
