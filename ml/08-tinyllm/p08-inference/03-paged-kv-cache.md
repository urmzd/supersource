<!-- ss:module L8.3 -->
# Paged KV cache in Python over the C block pool

## Overview

| | |
|---|---|
| **Module** | `L8.3` · build · Python · Pass 6 · 4 to 6 h |
| **You build** | `python/tinyllm/infer/paged.py`: `PagedKVCache` (`add_seq`, `fork`, `append`, `block_table`, `seq_len`, `gather`, `free`, `stats`, `num_free_blocks`, `pool`, `close`) and `OutOfBlocks` |
| **Contract** | [`course/contracts/py/tinyllm/infer/paged.pyi`](../../../course/contracts/py/tinyllm/infer/paged.pyi); the pool underneath is [`tinyllm/kv_pool.h`](../../../course/contracts/c/include/tinyllm/kv_pool.h) |
| **Tests** | `course/tests/L8.3/test_paged.py`, 10 tests (what they check: section 4) · your own tests in `python/tests/l8-3-paged/`, rung R4, graded by mutation (threshold 0.80) |
| **Needs** | `L8.2` [the contiguous KV cache](02-kv-cache-and-generate.md) (the oracle) · `rt.04` [the paged KV block pool](08-paged-kv-block-pool.md) · `rt.01` [the ctypes loader](../p09-kernels/01-the-c-abi.md) |
| **Used by** | later: `L9.4` (the paged attention kernel is checked against `gather`), `L10.4` (the Rust block manager implements the same semantics), `L8.2`'s `generate(cache="paged")` |
| **Milestone** | `MS-L8` (step 1: `generate --cache paged` equals `--cache contiguous` with `--kv-dtype f16`) |
| **Optional depth** | Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), section 4; Silberschatz et al., *Operating System Concepts*, chapter 9 (paging) |

## Key Takeaways

- A paged cache is a block table per sequence: position $p$ lives in slot $p \bmod B$ of block `table[p // B]`, so memory grows one block at a time (`test_hand_example_block_table`).
- Python owns only tables and lengths; every K and V value sits in the C pool, viewed from numpy without copies, in the `[heads][block_size][d_head]` layout the C kernels read (`test_slab_layout_is_heads_tokens_dims`).
- A fork copies a table and takes references; the first write into a shared block copies that block alone (`test_fork_shares_until_a_write`).
- Paged and contiguous float16 caches hold the same numbers, so a decoder gives the same logits and the same greedy tokens from either (`test_paged_matches_contiguous_decoding`).
- An append that cannot get its blocks raises `OutOfBlocks` before anything visible changes, so the scheduler can preempt and retry (`test_out_of_blocks_changes_nothing`).

## How to work this chapter

```bash
ss start L8.3               # stubs python/tinyllm/infer/paged.py
ss tests L8.3               # read the test catalog first
ss check L8.3               # exit code is the verdict; then grades your tests by mutation
ss check L8.3 --ref-deps    # only if L8.2, rt.04, or rt.01 is not passing yet
ss diff  L8.3               # after passing: your code against the reference
```

Start with `add_seq`, `append` for one token at a time, and `gather`; then multi-token chunks that cross block boundaries; then `fork` with copy on write; then `free` and `OutOfBlocks`.

---

## 1. Why now

`L8.2`'s `KVCache` preallocates `max_len` positions per sequence and layer. With a 2,048-token context and SmolLM2's 30 layers, 3 KV heads, and 64 dimensions, that is $2048 \cdot 30 \cdot 2 \cdot 3 \cdot 64 \cdot 2 = 47$ MB per sequence in float16, whether the sequence ends after 20 tokens or 2,000; and beam search with 4 beams copies the whole prompt 4 times. You now have a C pool of fixed-size blocks with reference counts and copy on write (`rt.04`). This module puts a Python cache on top of it, with the same answers as the contiguous cache, so `generate` can switch to paged memory and the later C kernel (`L9.4`) and Rust engine (`L10.4`) have a reference to be checked against.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | `block_size`, positions per block | `int` |
| $L$, $H$, $D$ | layers, KV heads, head dimension | `int` |
| $\ell_l$ | `seq_len(seq, l)`: positions layer $l$ holds | `int` |
| table | the sequence's block ids in position order | `list[int]` |
| $T$ | positions appended by one call | `int` |

**Block tables.** Position $p$ of a sequence lives in slot $p \bmod B$ of block `table[p // B]`. A sequence of $n$ positions needs $\lceil n / B \rceil$ blocks; the last one is usually partial. Appending $T$ positions starting at $\ell$ touches blocks $\lfloor \ell / B \rfloor$ through $\lfloor (\ell + T - 1) / B \rfloor$ and allocates the ones past the end of the table.

**Layers append separately.** A forward pass appends layer 0's keys and values, then layer 1's, and so on, so mid-pass the layers hold different lengths. Each block holds all layers, and its **fill** (what `tl_kv_set_fill` records and `tl_kv_export` sends) must count only positions that every layer has written: $\text{fill}_i = \min(B, \max(0, \min_l \ell_l - iB))$.

**The slab view.** `tl_kv_block_ptr(pool, id, layer, is_v)` returns the address of a `[H][B][D]` float16 array. `numpy.ctypeslib.as_array` turns that address into a numpy array that shares the memory, so `slab[:, slot:slot + n, :] = k[:, i:i + n, :]` writes straight into C memory. Values are converted to float16 (round to nearest even) on the way in, so `gather` returns exactly what a contiguous float16 cache holds.

**Copy on write.** `fork(parent, child)` copies the table and calls `tl_kv_ref` on every block. Before `append` writes into any block it already holds, it calls `tl_kv_cow`, which returns the same id for a block only this sequence holds and a fresh copy for a shared one. The parent and child then diverge in their last block only; every full block of the shared prompt stays shared.

**Failure without side effects.** `append` first copies the shared blocks it will touch, then allocates every new block it needs in one `tl_kv_alloc` (all or nothing), and only then writes and moves the length. If the pool cannot supply the blocks, it raises `OutOfBlocks`, and `gather` and `seq_len` return what they did before.

## 3. Worked example by hand

$B = 2$, one layer, one head, $D = 2$. Append five tokens one at a time, with K of token $t$ = `[t + 0.5, -t]` and V = `-K`:

| After token | Length | Table | Where the token went | Pool (free, used) of 4 |
|---|---|---|---|---|
| 0 | 1 | `[b0]` | `b0` slot 0 | (3, 1) |
| 1 | 2 | `[b0]` | `b0` slot 1 | (3, 1) |
| 2 | 3 | `[b0, b1]` | `b1` slot 0 | (2, 2) |
| 3 | 4 | `[b0, b1]` | `b1` slot 1 | (2, 2) |
| 4 | 5 | `[b0, b1, b2]` | `b2` slot 0 | (1, 3) |

Reading `b2`'s K slab from C memory gives `[4.5, -4.0]` at head 0, slot 0, and `gather(0, 0)` gives K positions `0.5 1.5 2.5 3.5 4.5` in the first dimension. This is `test_hand_example_block_table`. A fork now shares `b0`, `b1`, `b2` (each at refcount 2, still 3 used); the child's sixth token goes into `b2` slot 1, so `b2` is copied first (4 used), and the parent's view is unchanged.

## 4. The interface

```python
class OutOfBlocks(RuntimeError): ...

class PagedKVCache:
    def __init__(self, lib, num_blocks: int, block_size: int, n_layers: int, n_kv_heads: int, d_head: int): ...
    @property
    def pool(self) -> ctypes.c_void_p                 # the tl_kv_pool * for C kernels (L9.4)
    def add_seq(self, seq_id: int) -> None
    def fork(self, parent: int, child: int) -> None
    def append(self, seq_id: int, layer: int, k, v) -> None        # k, v: [n_kv_heads, T, d_head]
    def block_table(self, seq_id: int) -> NDArray                  # int32
    def seq_len(self, seq_id: int, layer: int = 0) -> int
    def gather(self, seq_id: int, layer: int) -> tuple[NDArray, NDArray]   # float16 [H, len, D]
    def free(self, seq_id: int) -> None
    def stats(self) -> dict[str, int]                              # free, used, cached, evictions
    def num_free_blocks(self) -> int
    def close(self) -> None                                        # also __exit__ and garbage collection
```

`lib` is the `Lib` from `tinyllm.ffi.libtinyllm.load()`; the cache declares the `tl_kv_*` signatures it needs with `Lib.declare`, and a status other than `TL_OK` arrives as `TlError` (`TL_EFULL` becomes `OutOfBlocks`). Shapes have no batch axis: one sequence per call.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_block_table` | unit | section 3: a block only when a position needs it; the slot read from C memory; pool counts | you and the tests agree on the mapping |
| `test_slab_layout_is_heads_tokens_dims` | unit | chunk appends land at `slab[h, p % B, d]` of block `table[p // B]`, read raw from C | L9.4's kernel reads the same memory |
| `test_gather_matches_contiguous_float16` | differential | chunks of 1 to 7 tokens, 3 layers, 2 interleaved sequences equal a contiguous float16 cache exactly | positions in order across block boundaries |
| `test_fork_shares_until_a_write` | unit | a fork allocates nothing; the first write into the shared partial block copies it; full blocks stay shared | beam search and parallel sampling |
| `test_free_restores_the_pool` | unit | a forked block returns only when both holders free it; freeing all restores every block | KV usage back to baseline after each request |
| `test_out_of_blocks_changes_nothing` | fault | a multi-block append that cannot be satisfied leaves length, values, and pool counts as they were | the scheduler preempts and retries safely |
| `test_fill_counts_positions_every_layer_holds` | unit | fills follow the slowest layer mid-pass | an export never ships half-written positions |
| `test_bad_arguments` | boundary | bad shapes, empty chunks, unknown or reused ids, bad layers, sizes below 1 | engine bugs fail loudly |
| `test_paged_matches_contiguous_decoding` | differential | a 2-layer toy decoder: logits within 1e-5 and the same greedy ids from the paged cache and from your `L8.2` `KVCache(dtype=float16)`, through a prompt and two forked branches | the equivalence MS-L8 checks on a real model |
| `test_random_ops_conserve_blocks` | property | $10^4$ seeded add, fork, append, free operations against a numpy oracle; the pool's counts add up after each; freeing all restores every block | no refcount drifts or leaks over a long run |

**Your tests (rung R4).** Write `python/tests/l8-3-paged/test_*.py`, importing only names from `contracts/py` (`tinyllm.infer.paged`, `tinyllm.ffi.libtinyllm`), with properties in prose turned into code (Hypothesis is allowed): "gather after any sequence of appends equals the concatenation of what was appended, in float16", "a fork then an append to the child never changes the parent", "freeing every sequence returns every block", "an `OutOfBlocks` append changes nothing". The planted bugs "a fork copies the table without references" (`s06`) and "a shared block is written in place" (`s07`) are required; overall 80%.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Allocating a block when a write ends exactly on a block boundary | one block per sequence wasted; tables longer than $\lceil n / B \rceil$ | `test_hand_example_block_table` (mutant `s01`) |
| Gathering whole blocks | the partial last block returns stale positions | `test_hand_example_block_table` (mutant `s02`) |
| Viewing the slab as `[B][H][D]` | your own gather round-trips, the C kernel reads garbage | `test_slab_layout_is_heads_tokens_dims` (mutant `s03`) |
| Writing K into the V slab | attention mixes the wrong values | `test_gather_matches_contiguous_float16` (mutant `s04`) |
| A chunk ignoring its starting slot | the write runs past the block's slab | `test_gather_matches_contiguous_float16` (mutant `s05`) |
| Forking without `tl_kv_ref` | the parent's free returns blocks the child still reads | `test_fork_shares_until_a_write`, `test_random_ops_conserve_blocks` (mutant `s06`) |
| Writing into a shared block in place | the parent sees the child's tokens | `test_fork_shares_until_a_write`, `test_paged_matches_contiguous_decoding` (mutant `s07`) |
| Freeing the table without unref | blocks leak on every request | `test_free_restores_the_pool` (mutant `s08`) |
| Unref of the first block only | leaks all but one block per request | `test_free_restores_the_pool` (mutant `s09`) |
| Moving the length before allocating | a failed append still grows the sequence | `test_out_of_blocks_changes_nothing` (mutant `s10`) |
| Allocating new blocks one at a time | a failed multi-block append keeps what it took | `test_out_of_blocks_changes_nothing` (mutant `s11`) |
| Fill from the fastest layer | exports and prefix registration see unwritten positions | `test_fill_counts_positions_every_layer_holds` (mutant `s12`) |
| Accepting an empty chunk | silent no-ops hide engine bugs | `test_bad_arguments` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.2` | the contiguous float16 cache is the oracle; `generate(cache="paged")` takes this cache |
| Back | `rt.04` | every block, reference, copy, and fill goes through the pool |
| Back | `rt.01` | the ctypes loader: `Lib.declare`, `TlError` |
| Forward | `L9.4` | `tl_paged_attn_decode_f32` reads `cache.pool` and `block_table` directly; its tests compare with attention over `gather` |
| Forward | `L10.4` | the Rust block manager keeps the same tables, forks, and fills over the same C pool |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `PagedKVCache` | vLLM's `KVCacheManager` | block tables on the GPU, prefix-cache hits at allocation, sliding-window and hybrid layouts | [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) |
| `fork` with copy on write | vLLM v0 `BlockSpaceManager.fork` | the same refcounted fork for beam search | [`vllm/core/block_manager.py`](https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/core/block_manager.py) |
| float16 slabs | FlashInfer paged KV | page tables consumed directly by fused attention kernels | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `include/flashinfer/page.cuh` |
