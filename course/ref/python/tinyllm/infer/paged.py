"""A paged KV cache over the C block pool (L8.3).

Contract: contracts/py/tinyllm/infer/paged.pyi. The pool is rt.04's
tl_kv_pool, reached through the rt.01 ctypes loader; this module owns only
the block tables (one list of block ids per sequence) and the per-layer
lengths. Every K and V value lives in C memory, viewed from numpy without a
copy while it is written.

A block holds block_size positions of every layer. Position p of a sequence
is slot p % B of block table[p // B]; the slab of (block, layer, K or V) is
[n_kv_heads][block_size][d_head] float16.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.ffi.libtinyllm import TlError

# C ABI constants (tinyllm/abi.h, tinyllm/kv_pool.h): fixed by the ABI, so
# they are spelled here rather than imported.
_TL_EFULL = 5
_TL_F16 = 1
_FORMAT_V1 = 1


class _KvCfg(ctypes.Structure):
    _fields_ = [
        ("n_blocks", ctypes.c_uint32),
        ("block_tokens", ctypes.c_uint32),
        ("n_layers", ctypes.c_uint32),
        ("n_kv_heads", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("dtype", ctypes.c_int32),
        ("format", ctypes.c_uint32),
    ]


class _KvStats(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("free", "used", "cached", "evictions")]


def _declare(lib: Any) -> None:
    """The kv_pool.h signatures this module calls (rt.04). Status-returning
    functions are declared with a plain int32 result and checked by _check,
    so this module needs nothing from the loader but Lib.declare and TlError."""
    # SOLUTION-BEGIN L8.3
    vp, u32, u32p, st = ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32
    sigs = {
        "tl_kv_pool_create": (st, [ctypes.POINTER(_KvCfg), ctypes.POINTER(vp)]),
        "tl_kv_pool_destroy": (None, [vp]),
        "tl_kv_alloc": (st, [vp, u32, u32p]),
        "tl_kv_ref": (None, [vp, u32]),
        "tl_kv_unref": (st, [vp, u32]),
        "tl_kv_cow": (st, [vp, u32, u32p]),
        "tl_kv_set_fill": (st, [vp, u32, u32]),
        "tl_kv_block_ptr": (vp, [vp, u32, u32, ctypes.c_int]),
        "tl_kv_stats_get": (None, [vp, ctypes.POINTER(_KvStats)]),
    }
    for name, (restype, argtypes) in sigs.items():
        lib.declare(name, restype, argtypes)
    # SOLUTION-END


def _check(lib: Any, fn: str, status: int) -> None:
    """TL_EFULL becomes OutOfBlocks; any other failure TlError."""
    # SOLUTION-BEGIN L8.3
    if status == 0:
        return
    if status == _TL_EFULL:
        raise OutOfBlocks(f"{fn}: the KV pool has no free or cached block left")
    raise TlError(fn, status, lib.last_error())
    # SOLUTION-END


class OutOfBlocks(RuntimeError):
    """The pool has too few free or cached blocks for this append or fork."""


class PagedKVCache:
    def __init__(
        self, lib: Any, num_blocks: int, block_size: int, n_layers: int, n_kv_heads: int, d_head: int
    ) -> None:
        # SOLUTION-BEGIN L8.3
        dims = (num_blocks, block_size, n_layers, n_kv_heads, d_head)
        if any(int(x) < 1 for x in dims):
            raise ValueError(f"PagedKVCache: every size must be at least 1, got {dims}")
        self.lib = lib
        self.num_blocks, self.block_size, self.n_layers, self.n_kv_heads, self.d_head = map(int, dims)
        _declare(lib)
        cfg = _KvCfg(self.num_blocks, self.block_size, self.n_layers, self.n_kv_heads, self.d_head,
                     _TL_F16, _FORMAT_V1)
        self._pool = ctypes.c_void_p()
        _check(lib, "tl_kv_pool_create", lib.tl_kv_pool_create(ctypes.byref(cfg), ctypes.byref(self._pool)))
        self._tables: dict[int, list[int]] = {}
        self._lens: dict[int, list[int]] = {}
        # SOLUTION-END

    @property
    def pool(self) -> Any:
        # SOLUTION-BEGIN L8.3
        return self._pool
        # SOLUTION-END

    # -- sequences --------------------------------------------------------------

    def add_seq(self, seq_id: int) -> None:
        # SOLUTION-BEGIN L8.3
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already exists")
        self._tables[seq_id] = []
        self._lens[seq_id] = [0] * self.n_layers
        # SOLUTION-END

    def fork(self, parent: int, child: int) -> None:
        # SOLUTION-BEGIN L8.3
        table = self._table(parent)
        if child in self._tables:
            raise ValueError(f"sequence {child} already exists")
        for b in table:  # the child holds every block too: no copy until a write
            self.lib.tl_kv_ref(self._pool, b)
        self._tables[child] = list(table)
        self._lens[child] = list(self._lens[parent])
        # SOLUTION-END

    def free(self, seq_id: int) -> None:
        # SOLUTION-BEGIN L8.3
        table = self._table(seq_id)
        for b in table:
            _check(self.lib, "tl_kv_unref", self.lib.tl_kv_unref(self._pool, b))
        del self._tables[seq_id], self._lens[seq_id]
        # SOLUTION-END

    def _table(self, seq_id: int) -> list[int]:
        # SOLUTION-BEGIN L8.3
        try:
            return self._tables[seq_id]
        except KeyError:
            raise KeyError(f"unknown sequence {seq_id}") from None
        # SOLUTION-END

    def _layer(self, layer: int) -> int:
        # SOLUTION-BEGIN L8.3
        if not 0 <= int(layer) < self.n_layers:
            raise ValueError(f"layer {layer} is out of range 0..{self.n_layers - 1}")
        return int(layer)
        # SOLUTION-END

    # -- blocks -----------------------------------------------------------------

    def _slab(self, block: int, layer: int, is_v: int) -> NDArray:
        """A float16 [n_kv_heads, block_size, d_head] view of C memory."""
        # SOLUTION-BEGIN L8.3
        ptr = self.lib.tl_kv_block_ptr(self._pool, block, layer, is_v)
        if not ptr:
            raise TlError("tl_kv_block_ptr", 1, self.lib.last_error())
        shape = (self.n_kv_heads, self.block_size, self.d_head)
        u16 = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16))
        return np.ctypeslib.as_array(u16, shape=shape).view(np.float16)
        # SOLUTION-END

    def _alloc(self, n: int) -> list[int]:
        # SOLUTION-BEGIN L8.3
        if n == 0:
            return []
        ids = (ctypes.c_uint32 * n)()
        _check(self.lib, "tl_kv_alloc", self.lib.tl_kv_alloc(self._pool, n, ids))
        return list(ids)
        # SOLUTION-END

    def _private(self, table: list[int], i: int) -> None:
        """Make table[i] a block this sequence alone holds (copy on write)."""
        # SOLUTION-BEGIN L8.3
        out = ctypes.c_uint32()
        _check(self.lib, "tl_kv_cow", self.lib.tl_kv_cow(self._pool, table[i], ctypes.byref(out)))
        table[i] = out.value
        # SOLUTION-END

    def append(self, seq_id: int, layer: int, k: ArrayLike, v: ArrayLike) -> None:
        # SOLUTION-BEGIN L8.3
        table = self._table(seq_id)
        layer = self._layer(layer)
        k16 = np.asarray(k, dtype=np.float16)
        v16 = np.asarray(v, dtype=np.float16)
        want = (self.n_kv_heads, k16.shape[1] if k16.ndim == 3 else -1, self.d_head)
        if k16.ndim != 3 or k16.shape != want or v16.shape != want or want[1] < 1:
            raise ValueError(
                f"append: k and v must be [n_kv_heads={self.n_kv_heads}, T >= 1, d_head={self.d_head}], "
                f"got {k16.shape} and {v16.shape}"
            )
        B, T = self.block_size, want[1]
        start = self._lens[seq_id][layer]
        end = start + T
        first, last = start // B, (end - 1) // B  # blocks this write touches
        # 1. Copy every shared block the write touches; nothing is written yet.
        for i in range(first, min(last + 1, len(table))):
            self._private(table, i)
        # 2. Allocate the new blocks, all or nothing.
        table.extend(self._alloc(max(0, last + 1 - len(table))))
        # 3. Write, one block at a time.
        p = start
        while p < end:
            b, slot = divmod(p, B)
            n = min(B - slot, end - p)
            self._slab(table[b], layer, 0)[:, slot : slot + n, :] = k16[:, p - start : p - start + n, :]
            self._slab(table[b], layer, 1)[:, slot : slot + n, :] = v16[:, p - start : p - start + n, :]
            p += n
        self._lens[seq_id][layer] = end
        # 4. A block's fill is the positions every layer holds.
        held = min(self._lens[seq_id])
        for i in range(first, last + 1):
            fill = min(B, max(0, held - i * B))
            _check(self.lib, "tl_kv_set_fill", self.lib.tl_kv_set_fill(self._pool, table[i], fill))
        # SOLUTION-END

    # -- reads ------------------------------------------------------------------

    def block_table(self, seq_id: int) -> NDArray:
        # SOLUTION-BEGIN L8.3
        return np.array(self._table(seq_id), dtype=np.int32)
        # SOLUTION-END

    def seq_len(self, seq_id: int, layer: int = 0) -> int:
        # SOLUTION-BEGIN L8.3
        self._table(seq_id)
        return self._lens[seq_id][self._layer(layer)]
        # SOLUTION-END

    def gather(self, seq_id: int, layer: int) -> tuple[NDArray, NDArray]:
        # SOLUTION-BEGIN L8.3
        table = self._table(seq_id)
        layer = self._layer(layer)
        n = self._lens[seq_id][layer]
        B = self.block_size
        K = np.empty((self.n_kv_heads, n, self.d_head), dtype=np.float16)
        V = np.empty_like(K)
        for b in range((n + B - 1) // B):
            m = min(B, n - b * B)
            K[:, b * B : b * B + m, :] = self._slab(table[b], layer, 0)[:, :m, :]
            V[:, b * B : b * B + m, :] = self._slab(table[b], layer, 1)[:, :m, :]
        return K, V
        # SOLUTION-END

    def stats(self) -> dict[str, int]:
        # SOLUTION-BEGIN L8.3
        s = _KvStats()
        self.lib.tl_kv_stats_get(self._pool, ctypes.byref(s))
        return {"free": s.free, "used": s.used, "cached": s.cached, "evictions": s.evictions}
        # SOLUTION-END

    def num_free_blocks(self) -> int:
        # SOLUTION-BEGIN L8.3
        s = self.stats()
        return s["free"] + s["cached"]
        # SOLUTION-END

    # -- lifetime -----------------------------------------------------------------

    def close(self) -> None:
        # SOLUTION-BEGIN L8.3
        pool = getattr(self, "_pool", None)
        if pool:
            self.lib.tl_kv_pool_destroy(pool)
        self._pool = ctypes.c_void_p()
        self._tables, self._lens = {}, {}
        # SOLUTION-END

    def __enter__(self) -> "PagedKVCache":
        # SOLUTION-BEGIN L8.3
        return self
        # SOLUTION-END

    def __exit__(self, *exc: Any) -> None:
        # SOLUTION-BEGIN L8.3
        self.close()
        # SOLUTION-END

    def __del__(self) -> None:
        # SOLUTION-BEGIN L8.3
        try:
            self.close()
        except Exception:  # noqa: BLE001  (interpreter shutdown: the library may be gone)
            pass
        # SOLUTION-END
