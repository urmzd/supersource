# contracts/py/tinyllm/infer/paged.pyi (L8.3): a paged KV cache over the C block pool
# chapter: ml/08-tinyllm/p08-inference/03-paged-kv-cache.md
#
# The cache keeps no K or V of its own: every value lives in a block of
# libtinyllm's tl_kv_pool (rt.04, tinyllm/kv_pool.h), reached through the
# rt.01 ctypes loader. A sequence is a block table, the list of block ids
# holding its positions in order: position p of a sequence is slot p % B of
# block table[p // B], B = block_size. Inside a block, the K (or V) slab of a
# layer is [n_kv_heads][block_size][d_head] float16, the layout
# tl_kv_block_ptr documents and the paged attention kernel (L9.4) reads.
#
# Words used below:
#   seq_id     any int naming a live sequence
#   len(l)     how many positions of layer l a sequence holds; every layer
#              is appended separately, so the lengths may differ mid-step
#   shared     a block that more than one sequence's table holds (after fork)
#
# Values are stored as float16 (round to nearest even), so gather returns
# exactly what a contiguous float16 cache (L8.2, dtype=np.float16) holds.
# Writes never touch a shared block: append copies it first (tl_kv_cow), so
# a fork's parent and child diverge without seeing each other's tokens.
from typing import Any

from numpy.typing import ArrayLike, NDArray

class OutOfBlocks(RuntimeError):
    """The pool has too few free (or cached) blocks for an append or a fork.
    Raised before anything visible changes: gather and len return what they
    returned before the call."""

class PagedKVCache:
    num_blocks: int
    block_size: int
    n_layers: int
    n_kv_heads: int
    d_head: int
    lib: Any  # the rt.01 Lib the pool lives in

    def __init__(
        self, lib: Any, num_blocks: int, block_size: int, n_layers: int, n_kv_heads: int, d_head: int
    ) -> None:
        """Declare the kv_pool.h signatures on lib (Lib.declare) and create a
        format 1 (TL_F16) pool of num_blocks blocks. ValueError for a size
        below 1; TlError when tl_kv_pool_create fails."""

    @property
    def pool(self) -> Any:
        """The tl_kv_pool * (a ctypes c_void_p) for C kernels that read the
        cache directly (tl_paged_attn_decode_f32, L9.4)."""

    def add_seq(self, seq_id: int) -> None:
        """A new empty sequence. ValueError when seq_id is live."""

    def fork(self, parent: int, child: int) -> None:
        """child becomes a copy of parent that shares every block (one more
        reference each, tl_kv_ref) and has the same lengths. KeyError for an
        unknown parent; ValueError when child is live."""

    def append(self, seq_id: int, layer: int, k: ArrayLike, v: ArrayLike) -> None:
        """Write T new positions at the end of `layer`: k and v are
        [n_kv_heads, T, d_head] (T >= 1), converted to float16. Allocates the
        blocks the new positions need and copies any shared block before
        writing into it. Afterwards each block's fill (tl_kv_set_fill) is the
        number of its positions that every layer holds. KeyError for an
        unknown sequence; ValueError for a bad layer or shape; OutOfBlocks
        when the pool is exhausted."""

    def block_table(self, seq_id: int) -> NDArray:
        """int32 block ids in position order: ceil(max over layers of len / B)
        entries. A copy; KeyError for an unknown sequence."""

    def seq_len(self, seq_id: int, layer: int = 0) -> int:
        """len(layer) of the sequence. KeyError, ValueError as append."""

    def gather(self, seq_id: int, layer: int) -> tuple[NDArray, NDArray]:
        """(K, V), each float16 [n_kv_heads, len(layer), d_head]: the layer's
        positions in order, copied out of the blocks."""

    def free(self, seq_id: int) -> None:
        """Drop the sequence's reference to each of its blocks (tl_kv_unref);
        a block nobody else holds returns to the pool. KeyError when unknown."""

    def stats(self) -> dict[str, int]:
        """tl_kv_stats_get: {"free", "used", "cached", "evictions"}; free +
        used + cached == num_blocks."""

    def num_free_blocks(self) -> int:
        """stats()["free"] + stats()["cached"]: blocks an allocation can take."""

    def close(self) -> None:
        """Destroy the pool (every sequence is gone). Idempotent; also runs
        on __exit__ and garbage collection."""

    def __enter__(self) -> "PagedKVCache": ...
    def __exit__(self, *exc: Any) -> None: ...
