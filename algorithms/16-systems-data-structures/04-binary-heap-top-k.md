<!-- ss:module ds.04 -->
# Binary heap top-k in C

## Overview

| | |
|---|---|
| **Module** | `ds.04` · build · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/ds/topk.c`: `tl_topk_f32`, the $k$ largest values of a float array and their indices, best first, by a size-$k$ min-heap that lives in the caller's output buffers |
| **Contract** | [`course/contracts/c/include/tinyllm/topk.h`](../../course/contracts/c/include/tinyllm/topk.h) · rules: [`c/ABI.md`](../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/ds.04/`: `test_topk.c` (C, under ASan and UBSan, with a property test against a full sort) and `test_topk_ctypes.py` (Python, through your `rt.01` loader, against numpy and your `L8.1` sampler) (what they check: section 4) |
| **Needs** | `rt.01` the error slot and the ctypes loader · `L8.1` your sampler's `process_logits`, the Python specification of top-k (or `--ref-deps`). Reading: `ds.06` [the first heap chapter](06-binary-heap-lazy-deletion.md) · `ds.01` (`tl_vec`, which this module does not need: the heap allocates nothing) |
| **Used by** | `L10.1` (Pass 7): your Rust engine's sampler calls `tl_topk_f32` through `tl-sys` for every request with `top_k > 0` |
| **Milestone** | `MS-P6` (inference and kernels) |
| **Optional depth** | Cormen et al., *Introduction to Algorithms* (4th ed.), chapter 6 (heaps) and section 9.2 (selection in expected linear time); Knuth, *TAOCP* vol. 3, section 5.3.3 (minimum-comparison selection) |

## Key Takeaways

- **Top-$k$ needs only a heap of size $k$**: keep the best $k$ seen so far with the *worst* of them at the root, and each new value costs one comparison against the root, plus $O(\log k)$ swaps only when it gets in (`hand_example`, `matches_a_full_sort`).
- **The order is total**: a smaller value ranks lower, and among equal values the larger index ranks lower. That one rule settles both who wins a tie at the $k$-th place (the lower index) and the output order of equal values (ascending index) (`tie_at_the_cut_goes_to_the_lower_index`, `equal_values_are_listed_by_ascending_index`).
- **NaN is rejected, $-\infty$ is a value.** NaN has no place in any order; a masked logit is $-\infty$ and simply ranks last (`nan_and_bad_k_are_einval_and_write_nothing`, `minus_infinity_is_an_ordinary_value`).
- **No allocation**: the heap is built inside `idx` and `val`, and a final in-place heap sort leaves them best first.
- **The C kernel and your Python sampler agree** on the kept set for every golden row the Rust sampler is held to (`test_matches_your_l8_1_top_k_on_fixture_logits`).

## How to work this chapter

```bash
ss start ds.04              # stubs c/src/ds/topk.c into your repo
ss tests ds.04              # read the test catalog first
ss check ds.04              # exit code is the verdict
ss check ds.04 --ref-deps   # only if your rt.01 or L8.1 is not passing yet
ss diff  ds.04              # after passing: your code against the reference
```

---

## 1. Why now

Your Python sampler (`L8.1`) implements top-$k$ by sorting every logit: `process_logits` orders all $V$ ids by (logit descending, id ascending) and keeps the first $k$. For SmolLM2, $V = 49152$, and the engine samples a token for every sequence in the batch at every step. In Pass 7 your Rust engine (`L10.1`) takes the sampler over, and its top-$k$ step will call into `libtinyllm` through `tl-sys`, the same way it calls the matmul. Sorting 49152 floats to keep 40 of them is the wrong algorithm: about $V \log_2 V \approx 770{,}000$ comparisons where roughly $V$ suffice. Right now `tl_topk_f32` is a stub that returns `TL_EUNSUPPORTED`. This module writes it, and holds it to exactly the tie rule your Python sampler uses, because a sampler that keeps a different token at the $k$-th place produces a different stream for the same seed, and the parity suite between Python and Rust (`parity/sampler`) would fail on a tie.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^n$ | the input values (logits), $x_i$ at index $i$ | `float[n]` |
| $n$ | number of values, $V$ for a logit row | `int64_t` |
| $k$ | number of values to keep, $0 \le k \le n$ | `int64_t` |
| $(v, i)$ | an entry: a value and its index | |
| $a \prec b$ | entry $a$ ranks below ("is worse than") entry $b$ | |
| $\mathit{idx}, \mathit{val}$ | the outputs: the kept indices and values, best first | `int32_t[k]`, `float[k]` |
| $h$ | the heap's current size, $h \le k$ | |

### 2.1 A total order on entries

Top-$k$ is defined by an order. Values alone do not give one: two logits can be equal, and then "the $k$ largest" is ambiguous. The sampling spec (`spec/sampling.md` step 6) breaks ties by index, so define

$$(v_a, i_a) \prec (v_b, i_b) \iff v_a < v_b \ \text{ or } \ (v_a = v_b \text{ and } i_a > i_b).$$

Read it as "$a$ is worse than $b$". Indices are distinct, so for any two different entries exactly one of $a \prec b$ and $b \prec a$ holds: the order is **total**. The top-$k$ set is then unique: the $k$ entries that are worse than no more than $k - 1$ others. Listing it best first gives values in descending order and, among equal values, ascending indices.

NaN breaks this. IEEE 754 makes every comparison with NaN false, so a NaN entry would be neither worse nor better than anything, and the result would depend on where in the array it happened to sit. The contract therefore rejects any NaN with `TL_EINVAL`. $-\infty$ is different: $-\infty < v$ for every other value, and $-\infty = -\infty$, so it fits the order and ranks last. A logit masked by constrained decoding (`L8.7`) is $-\infty$, so rejecting it would break every masked request.

### 2.2 The heap keeps the worst kept entry at the root

A **binary heap** (the `ds.06` chapter) is a complete binary tree stored in an array: the children of position $p$ are $2p + 1$ and $2p + 2$, its parent is $\lfloor (p - 1)/2 \rfloor$. Here the heap property is: **no parent is better than its children**, so the root is the worst entry in the heap. That is a min-heap under $\prec$.

The algorithm scans $x$ once, keeping the best $h \le k$ entries seen so far in the heap:

1. While $h < k$, append $(x_i, i)$ at position $h$ and **sift it up**: swap it with its parent while it is worse than the parent.
2. Once $h = k$, compare $(x_i, i)$ with the root, the worst of the $k$ kept entries. If the root is worse, the new entry replaces it and is **sifted down**: swapped with its worse child while that child is worse than it. Otherwise the new entry is not among the $k$ best seen so far, and it is dropped.

**Invariant** (`S-M05` style): after processing $x_0, \dots, x_{i}$, the heap holds exactly the top-$\min(k, i+1)$ entries of that prefix. It holds after step 1 trivially. In step 2, if the new entry ranks below the root, it ranks below all $k$ kept entries and cannot be in the top $k$; otherwise the root is now ranked below $k$ entries (the other $k - 1$ and the new one) and must leave.

Ties need no special code. The scan goes in increasing index, so a later entry with the same value as the root has a larger index: it is worse, and it stays out. The lower index wins the last place, as the spec requires.

### 2.3 Sorting the result in place

After the scan, `idx` and `val` hold the top $k$ in heap order. A **heap sort** turns that into best-first order without extra memory: swap the root (the worst) with the last position, shrink the heap by one, sift the new root down, and repeat. The worst entry ends at position $k - 1$, the second worst at $k - 2$, and position 0 ends with the best.

### 2.4 Cost

Each of the $n$ values costs one comparison with the root. An entry that gets in costs $O(\log k)$ swaps, and the final sort costs $O(k \log k)$. In the worst case (an increasing array, where every value gets in) the total is $O(n \log k)$. For a logit row the order is close to random, and then the $i$-th value gets in with probability about $k / i$, so the expected number of insertions is about $k \ln(n/k)$: for $n = 49152$ and $k = 40$, about 280 insertions and 49152 root comparisons, against $n \log_2 n \approx 770{,}000$ comparisons for a full sort. Quickselect (Hoare) finds the $k$-th value in expected $O(n)$ but reorders the input, which the contract forbids (`x` is `const`), so it would need a copy of all $n$ values.

### 2.5 Validation before the first write

The contract allows `0 <= k <= n` and requires `idx` and `val` untouched when it returns `TL_EINVAL`. So the function checks the ranges and pointers, then scans all of $x$ for NaN, and only then writes. A NaN at the end of the row is found after the whole scan; that costs one extra pass over $x$, which is cheap next to the heap work and keeps the error path simple.

## 3. Worked example by hand

$x = [1, 3, 2, 3, -1]$ (the logits of the sampling spec's worked example), $k = 3$. Entries are written $(v, i)$, the heap as an array, root first.

| Step | Entry | Action | Heap after |
|---|---|---|---|
| $i = 0$ | $(1, 0)$ | $h = 0 < 3$: append, nothing to sift | $[(1,0)]$ |
| $i = 1$ | $(3, 1)$ | append at 1; parent $(1,0)$ is worse, so no swap | $[(1,0), (3,1)]$ |
| $i = 2$ | $(2, 2)$ | append at 2; parent $(1,0)$ is worse, no swap | $[(1,0), (3,1), (2,2)]$ |
| $i = 3$ | $(3, 3)$ | full; root $(1,0) \prec (3,3)$: replace root. Sift down: children $(3,1)$ and $(2,2)$; $(2,2) \prec (3,3)$ and $(3,1)$ is not (equal value, smaller index), so swap with $(2,2)$ | $[(2,2), (3,1), (3,3)]$ |
| $i = 4$ | $(-1, 4)$ | root $(2,2)$ is not worse than $(-1, 4)$: drop | unchanged |

Heap sort: swap root and position 2, giving $[(3,3), (3,1) \mid (2,2)]$; sift down in a heap of 2: $(3,1)$ is not worse than $(3,3)$, stop. Swap root and position 1: $[(3,1) \mid (3,3), (2,2)]$.

Result: `idx = {1, 3, 2}`, `val = {3, 3, 2}`. The two 3s tie, and the lower index comes first. The spec's own trace keeps ids 1, 3, 2 in that order: the first test, `hand_example`, and its Python twin `test_hand_example_through_ctypes`.

## 4. The interface

```c
/* tinyllm/topk.h */
tl_status tl_topk_f32(const float *x, int64_t n, int64_t k, int32_t *idx, float *val);
/* The k largest values of x[0..n) and their indices, best first; equal values by
   ascending index (and the lower index wins the last place). -inf is a value.
   TL_EINVAL, idx and val untouched, for k < 0 or k > n, a NULL pointer that would
   be used, or any NaN in x. k == 0 writes nothing. Allocates nothing. */
```

From Python, through your `rt.01` loader (the v0 loader does not know this symbol yet, so declare it):

```python
lib = load()
lib.declare("tl_topk_f32", STATUS, [POINTER(c_float), c_int64, c_int64, POINTER(c_int32), POINTER(c_float)])
lib.tl_topk_f32(f32_ptr(x), x.size, k, idx.ctypes.data_as(POINTER(c_int32)), f32_ptr(val))
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3 exactly | you and the tests agree on the definition |
| `tie_at_the_cut_goes_to_the_lower_index` | boundary | $[5, 1, 1, 1, 1, 0]$, $k = 2$ keeps ids 0 and 1 | the Rust and Python samplers keep the same token |
| `equal_values_are_listed_by_ascending_index` | boundary | all-equal input comes out in index order | top-p walks the kept ids in this order |
| `k_zero_and_k_equal_n` | boundary | $k = 0$ writes nothing; $k = n$ is a full sort; $n = 0$ is fine | `top_k = 0` means off |
| `minus_infinity_is_an_ordinary_value` | boundary | $-\infty$ ranks last, ties by index | masked logits from `L8.7` |
| `nan_and_bad_k_are_einval_and_write_nothing` | boundary | NaN, $k > n$, $k < 0$, NULL pointers give `TL_EINVAL` and leave the outputs as they were | errors instead of garbage |
| `matches_a_full_sort` | property, differential | 300 random rows with many ties against `qsort` by $\prec$, every $k$ | heap shapes only random sizes reach |
| `test_hand_example_through_ctypes` | unit, smoke | section 3 across the boundary | how the Rust sampler calls it |
| `test_matches_numpy_full_sort` | differential | a 49152-value row on a coarse grid against numpy's stable sort, $k$ up to 1000 | a full vocabulary row |
| `test_matches_your_l8_1_top_k_on_fixture_logits` | differential | the kept set equals your `process_logits(top_k=k)` on every golden row, $k \in \{1, 5, 12, 40\}$ | P6: Python is the specification |
| `test_nan_raises_einval_through_the_loader` | boundary | NaN arrives in Python as `TlError` with `TL_EINVAL` | the error path end to end |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| replacing the root when the new value is greater **or equal** | a tie at the $k$-th place keeps the last tied index; the Rust sampler drifts from Python on ties | `tie_at_the_cut_goes_to_the_lower_index` (mutant `s01`) |
| comparing values only inside the heap | equal values come out in heap order, not index order | `equal_values_are_listed_by_ascending_index` (mutant `s02`) |
| no NaN check | the result depends on where the NaN sits | `nan_and_bad_k_are_einval_and_write_nothing` (mutant `s03`) |
| no `k > n` check | the heap writes past `idx` and `val` (ASan) | `nan_and_bad_k_are_einval_and_write_nothing` (mutant `s04`) |
| forgetting the final sort | the right set in heap order; top-p (step 7) walks it in the wrong order | `hand_example` (mutant `s05`) |
| a max-heap (best at the root) | the root comparison drops the wrong entries | `hand_example`, `matches_a_full_sort` (mutant `s06`) |
| treating $k = 0$ as an error | a request with `top_k = 0` (off) fails | `k_zero_and_k_equal_n` (mutant `s07`) |
| writing before validating | a rejected call has already changed the outputs | `nan_and_bad_k_are_einval_and_write_nothing` (mutant `s08`) |
| rejecting with `!isfinite` instead of `isnan` | every masked request fails | `minus_infinity_is_an_ordinary_value` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.01` | the error slot behind `TL_EINVAL`, and the ctypes loader the Python tests call through |
| Back | `L8.1` | `process_logits` is the specification the kernel is compared with |
| Back | `ds.06` | the first heap chapter: sift up, sift down, the array layout (reading) |
| Forward | `L10.1` | the Rust sampler's top-$k$ step calls `tl_topk_f32` through `tl-sys` (Pass 7) |
| Forward | `L10.1` | the parity suite `sampler` then holds Rust and Python to identical ids on the golden logits |

If you skip this module, the Rust sampler in `L10.1` has no top-$k$ kernel to call: build it, or pass `--ref-deps` there.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| size-$k$ heap over one row | PyTorch `torch.topk` (CPU) | a partial sort (`std::partial_sort` / `nth_element`) chosen by $k/n$, many rows in parallel | `aten/src/ATen/native/TopKImpl.h` |
| CPU top-$k$ | vLLM and FlashInfer GPU samplers | top-$k$ and top-$p$ fused with sampling, by rejection instead of sorting | FlashInfer `sampling.cuh` |
| one row per call | llama.cpp `llama_sampler_top_k` | partial sort over the candidate array, reused by every sampler in the chain | `src/llama-sampling.cpp` |
