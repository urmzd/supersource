# C 08: Arena allocator

**Concepts:** bump allocation, alignment, block chaining
**Difficulty:** ⭐⭐⭐

An allocator that hands out memory by moving a pointer forward and frees
everything at once. No per-allocation `free`, no headers, no free list.

## The contract

`arena.h` declares it, `main.c` tests it, you write `arena.c`.

| Function | Does |
|----------|------|
| `arena_alloc_aligned(a, size, align)` | Bump, honouring any power-of-two alignment |
| `arena_alloc(a, size)` | Aligned for any scalar type |
| `arena_strdup(a, s)` | Copy a string in |
| `arena_mark(a)` / `arena_reset_to(a, m)` | Save a position, rewind to it |
| `arena_reset(a)` | Rewind everything, keep the memory |
| `arena_free(a)` | Return every block to the system |

`Block` is given, including its flexible array member and `_Alignas`.

## What to notice

**Align the address, not the offset within the block.** This is the bug the
reference implementation was written with, and the tests caught it. Offset
arithmetic works only while the requested alignment is no stricter than the
block's own: `data[]` starts at `max_align_t`, typically 16 bytes, so aligning
the offset silently returns misaligned memory the moment someone asks for 32 or
64. Both are ordinary requests, from SIMD types and cache-line padding
respectively. The test loops alignments from 1 to 64 for exactly this reason.

**`(n + align - 1) & ~(align - 1)` only works for powers of two.** Adding
`align-1` pushes past the boundary and the mask snaps back down to it, which
depends entirely on `align` having a single bit set. That is why every alignment
API in C, including `aligned_alloc` and `_Alignas`, insists on a power of two,
and why this one returns `NULL` rather than misbehaving when handed 3.

**Check the subtraction, not the sum.** `off + size > cap` looks like the
obvious bounds test and can wrap on a hostile size, producing a small sum that
passes the check and a write far outside the block. `size > cap - off` cannot
wrap, because `off <= cap` is already established.

**The tests write to every byte of every allocation, then read them all back.**
Checking that pointers look plausible is not enough: a wrong bump produces
overlapping regions where each individual pointer is fine and the second
allocation quietly corrupts the first. Filling and verifying is the only way to
see it.

**`reset` keeps the memory and `free` returns it, and the difference is the
entire use case.** The pattern an arena is built for is per-frame or per-request:
allocate freely, reset at the boundary, repeat. If `reset` returned memory to the
system, every cycle would pay for fresh `malloc` calls and you would have gained
nothing over the allocator you were avoiding. The test asserts that the second
round hands back *the same addresses* as the first.

**No `free` per allocation is a feature, not a limitation.** Deallocation is one
pointer write regardless of how many objects are live, there is no per-object
header, and fragmentation cannot happen. The cost is that individual lifetimes
are gone: everything in an arena dies together. That trade is why compilers,
game engines, and request-scoped server code use arenas heavily, and why nothing
with unpredictable object lifetimes does.

## Extending it

Add `arena_realloc` that grows in place when the block is the most recent
allocation and copies otherwise, which is the trick that makes a growable array
inside an arena cheap. Then try making `arena_reset` poison the reclaimed bytes
under a debug flag, so use-after-reset shows up immediately instead of reading
plausible stale data.
