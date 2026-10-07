# C 05: Merge sort

**Concepts:** pointer arithmetic, recursion, stability
**Difficulty:** ⭐⭐

A stable merge sort with one scratch allocation for the whole sort, and then the
genuinely in-place variant that uses rotations instead of a buffer.

## The contract

`msort.h` declares it, `main.c` tests it, you write `msort.c`.

| Function | Does | Space |
|----------|------|-------|
| `msort(a, n)` | Stable sort ascending by key | One buffer of `n`, allocated once |
| `msort_merge(a, left, n, scratch)` | Merge two adjacent sorted runs | Caller's scratch |
| `msort_inplace(a, n)` | Stable sort with no buffer at all | O(1), at O(n log²n) time |

Records carry a `key` and a `tag`. The tag is never compared, so it is how the
tests detect an unstable sort: tags start in input order, so equal keys that come
out with descending tags prove elements were reordered.

## What to notice

**Stability is one character.** In the merge loop, `a[i].key <= a[j].key` takes
from the left run on a tie and `<` takes from the right. Both produce a
correctly sorted array. Only one of them is stable, and the difference is
invisible until you sort records by one field having already sorted by another,
at which point the first sort's work is silently destroyed. This is why
`std::sort` and `std::stable_sort` are different functions.

**One allocation, not O(n log n) of them.** Allocating scratch inside the
recursion is correct and slow: `malloc` then dominates the runtime, and a
failure partway through leaves the array half-permuted with no way to report
which half. Allocating once up front means the sort either starts or does not.

**The `a[mid-1] <= a[mid]` check is nearly free and sometimes everything.** One
comparison per merge, and on already-sorted or nearly-sorted input the entire
sort collapses to a linear scan. Real data is very often nearly sorted, which is
the observation Timsort is built on.

**Comparison by subtraction is a bug waiting for the right input.** Writing
`a[i].key - a[j].key < 0` overflows when the keys straddle zero at the extremes,
and the sign flips. `test_extreme_keys` uses `INT_MIN` and `INT_MAX` for exactly
this reason. Compare with `<`, never with subtraction.

**"In-place merge sort" usually means rotation, and it costs a log factor.** The
buffer is what makes merging linear. Remove it and the standard approach is:
split the larger run in half, binary search for where that midpoint belongs in
the other run, rotate the block between them, and recurse. Rotation is three
reversals and no extra memory. The result is O(n log²n), meaningfully slower in
practice, and the honest answer to "can you merge sort without extra space."

**Stability in the rotating merge depends on which binary search you use, and
the two cases need different ones.** When you split the *left* run, the pivot
element came from the left, so equal keys on the right must stay after it:
`lower_bound`. When you split the *right* run, the pivot came from the right, so
equal keys on the left must come before it: `upper_bound`. Using the same bound
in both branches sorts correctly and is unstable, and it only shows on input
with duplicates spanning the seam. This exact bug is why
`test_stability_under_heavy_duplication` uses a key range of five over four
thousand elements.

## Extending it

The natural next step is a bottom-up merge sort with no recursion at all, which
is what you want when stack depth is a real constraint. After that, run
detection: scan for already-ordered runs, reverse the descending ones, and merge
only what is left. That is most of Timsort, and it turns nearly-sorted input
from O(n log n) into something close to O(n).
