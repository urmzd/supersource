"""Course tests for L8.3: the paged KV cache in Python over your C block pool.

PagedKVCache keeps block tables in Python and every K and V value in rt.04's
tl_kv_pool, reached through your rt.01 loader. The oracle is a contiguous
float16 cache: your L8.2 KVCache(dtype=np.float16) in the decoding
differential (the catalog's E test), and a small numpy one written here for
the tests that fork and free sequences, which KVCache does not model. Paged
and contiguous round K and V identically, so they must agree exactly.
Random inputs come from the frozen PCG32 (course/tests/_lib).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32

from tinyllm.ffi.libtinyllm import load
from tinyllm.infer.kvcache import KVCache
from tinyllm.infer.paged import OutOfBlocks, PagedKVCache

SEED = int(os.environ.get("SS_SEED", "0"))


def lib():
    L = load()
    L.declare("tl_kv_fill", ctypes.c_uint32, [ctypes.c_void_p, ctypes.c_uint32])
    L.declare("tl_kv_block_ptr", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int])
    return L


def raw_slab(cache: PagedKVCache, block: int, layer: int, is_v: int) -> np.ndarray:
    """The slab exactly as kv_pool.h lays it out: [n_kv_heads][block_size][d_head] f16."""
    ptr = cache.lib.tl_kv_block_ptr(cache.pool, int(block), layer, is_v)
    n = cache.n_kv_heads * cache.block_size * cache.d_head
    buf = (ctypes.c_uint16 * n).from_address(ptr)
    return np.frombuffer(buf, dtype=np.float16).reshape(cache.n_kv_heads, cache.block_size, cache.d_head).copy()


def rand(g: PCG32, shape, scale=1.0) -> np.ndarray:
    return (scale * g.uniform_array(shape, -1.0, 1.0)).astype(np.float32)


class Contiguous:
    """The oracle: one float16 [H, n, D] array per (sequence, layer)."""

    def __init__(self, n_layers: int, H: int, D: int):
        self.L, self.H, self.D = n_layers, H, D
        self.k: dict = {}
        self.v: dict = {}

    def add(self, s):
        for layer in range(self.L):
            self.k[s, layer] = np.zeros((self.H, 0, self.D), np.float16)
            self.v[s, layer] = np.zeros((self.H, 0, self.D), np.float16)

    def fork(self, p, c):
        for layer in range(self.L):
            self.k[c, layer] = self.k[p, layer].copy()
            self.v[c, layer] = self.v[p, layer].copy()

    def append(self, s, layer, k, v):
        self.k[s, layer] = np.concatenate([self.k[s, layer], np.asarray(k, np.float16)], axis=1)
        self.v[s, layer] = np.concatenate([self.v[s, layer], np.asarray(v, np.float16)], axis=1)

    def free(self, s):
        for layer in range(self.L):
            del self.k[s, layer], self.v[s, layer]


def test_hand_example_block_table():
    # WHY: the chapter's worked example. block_size 2, one layer, one head,
    #      d_head 2; five tokens appended one at a time need ceil(5/2) = 3
    #      blocks, and token 4 sits in slot 0 of the third block. Reading
    #      that slot straight from C memory checks the address arithmetic.
    # KIND: unit, smoke
    # CATCHES: s01, s02
    # CHAPTER: L8.3 section 3
    with PagedKVCache(lib(), 4, 2, 1, 1, 2) as c:
        c.add_seq(0)
        for t in range(5):
            k = np.array([[[t + 0.5, -t]]], np.float32)
            c.append(0, 0, k, -k)
            assert len(c.block_table(0)) == t // 2 + 1  # a block only when a position needs it
        table = c.block_table(0)
        assert table.dtype == np.int32 and len(table) == 3 and len(set(table.tolist())) == 3
        assert c.seq_len(0) == 5
        assert raw_slab(c, table[2], 0, 0)[0, 0].tolist() == [4.5, -4.0]
        assert raw_slab(c, table[2], 0, 1)[0, 0].tolist() == [-4.5, 4.0]
        assert raw_slab(c, table[0], 0, 0)[0, 1].tolist() == [1.5, -1.0]
        K, V = c.gather(0, 0)
        assert K.dtype == np.float16 and K.shape == (1, 5, 2)
        assert K[0, :, 0].tolist() == [0.5, 1.5, 2.5, 3.5, 4.5]
        assert c.stats() == {"free": 1, "used": 3, "cached": 0, "evictions": 0}
        assert c.num_free_blocks() == 1


def test_slab_layout_is_heads_tokens_dims():
    # WHY: L9.4's paged attention kernel reads the slab as
    #      [n_kv_heads][block_size][d_head]. Writing heads and positions in
    #      another order still round-trips through your own gather, so this
    #      test reads C memory directly: head h, position p, dim d of a
    #      chunk append lands at slab[h, p % B, d] of block table[p // B].
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: L8.3 section 2
    g = PCG32(SEED, 11)
    H, B, D = 3, 4, 5
    with PagedKVCache(lib(), 4, B, 2, H, D) as c:
        c.add_seq(7)
        for layer in range(2):
            k, v = rand(g, (H, 6, D)), rand(g, (H, 6, D))
            c.append(7, layer, k, v)
            t = c.block_table(7)
            for p in range(6):
                kk = raw_slab(c, t[p // B], layer, 0)[:, p % B, :]
                vv = raw_slab(c, t[p // B], layer, 1)[:, p % B, :]
                assert (kk == k[:, p, :].astype(np.float16)).all()
                assert (vv == v[:, p, :].astype(np.float16)).all()


def test_gather_matches_contiguous_float16():
    # WHY: chunks of 1 to 7 tokens across block boundaries, several layers,
    #      two sequences interleaved: gather must return exactly the float16
    #      values a contiguous cache holds, in position order.
    # KIND: differential
    # CATCHES: s02, s04, s05
    # CHAPTER: L8.3 section 4
    g = PCG32(SEED, 12)
    L_, H, B, D = 3, 2, 4, 3
    ref = Contiguous(L_, H, D)
    with PagedKVCache(lib(), 32, B, L_, H, D) as c:
        for s in (1, 2):
            c.add_seq(s)
            ref.add(s)
        for step in range(12):
            s = 1 + step % 2
            T = 1 + g.below(7)
            for layer in range(L_):
                k, v = rand(g, (H, T, D), 3.0), rand(g, (H, T, D), 3.0)
                c.append(s, layer, k, v)
                ref.append(s, layer, k, v)
        for s in (1, 2):
            for layer in range(L_):
                K, V = c.gather(s, layer)
                assert K.dtype == np.float16
                assert (K == ref.k[s, layer]).all() and (V == ref.v[s, layer]).all()


def test_fork_shares_until_a_write():
    # WHY: a fork must not copy anything: the child references the parent's
    #      blocks. The first write into the shared partial block copies it
    #      (one more used block), and the parent never sees the child's
    #      tokens. Full shared blocks stay shared.
    # KIND: unit
    # CATCHES: s06, s07
    # CHAPTER: L8.3 section 3
    with PagedKVCache(lib(), 8, 2, 1, 1, 1) as c:
        c.add_seq(0)
        for t in range(3):  # blocks: [t0 t1] [t2 .]
            c.append(0, 0, [[[t]]], [[[t]]])
        c.fork(0, 1)
        assert c.stats()["used"] == 2
        assert c.block_table(1).tolist() == c.block_table(0).tolist()
        c.append(1, 0, [[[9]]], [[[9]]])  # writes slot 1 of the shared block
        assert c.stats()["used"] == 3
        t0, t1 = c.block_table(0).tolist(), c.block_table(1).tolist()
        assert t0[0] == t1[0] and t0[1] != t1[1]
        assert c.gather(0, 0)[0][0, :, 0].tolist() == [0, 1, 2]
        assert c.gather(1, 0)[0][0, :, 0].tolist() == [0, 1, 2, 9]
        c.append(0, 0, [[[5]]], [[[5]]])  # the parent's block is its own again: no copy
        assert c.stats()["used"] == 3
        assert c.gather(0, 0)[0][0, :, 0].tolist() == [0, 1, 2, 5]


def test_free_restores_the_pool():
    # WHY: every block a sequence held comes back when the last holder
    #      frees it, so an engine's free count returns to its baseline after
    #      each request (the `--kv-stats` check of MS-L8). A forked block is
    #      returned only when both holders are gone.
    # KIND: unit
    # CATCHES: s08, s09
    # CHAPTER: L8.3 section 4
    with PagedKVCache(lib(), 6, 2, 2, 1, 2) as c:
        c.add_seq(0)
        for layer in range(2):
            c.append(0, layer, np.ones((1, 5, 2)), np.ones((1, 5, 2)))
        c.fork(0, 1)
        c.free(0)
        assert c.stats()["used"] == 3
        assert c.gather(1, 1)[0].shape == (1, 5, 2)
        c.free(1)
        assert c.stats() == {"free": 6, "used": 0, "cached": 0, "evictions": 0}
        with pytest.raises(KeyError):
            c.free(1)


def test_out_of_blocks_changes_nothing():
    # WHY: the scheduler (L10.2) treats an exhausted pool as "preempt
    #      someone and retry", which is only safe if the failed append left
    #      the sequence exactly as it was: same length, same values, no
    #      half-allocated blocks.
    # KIND: fault
    # CATCHES: s10, s11
    # CHAPTER: L8.3 section 5, Pitfalls
    with PagedKVCache(lib(), 4, 2, 1, 1, 1) as c:
        c.add_seq(0)
        c.append(0, 0, [[[1], [2], [3]]], [[[1], [2], [3]]])  # 2 blocks used, 2 free
        before = c.stats()
        six = np.arange(4, 10, dtype=np.float32).reshape(1, 6, 1)
        with pytest.raises(OutOfBlocks):
            c.append(0, 0, six, six)  # positions 3..8 need 3 new blocks
        assert c.seq_len(0) == 3
        assert c.gather(0, 0)[0][0, :, 0].tolist() == [1, 2, 3]
        assert c.stats() == before
        c.append(0, 0, six[:, :5], six[:, :5])  # positions 3..7 need exactly 2
        assert c.gather(0, 0)[0][0, :, 0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    with PagedKVCache(lib(), 2, 2, 1, 1, 1) as c:
        c.add_seq(0)
        c.append(0, 0, [[[1], [2], [3], [4]]], [[[1], [2], [3], [4]]])
        c.fork(0, 1)
        with pytest.raises(OutOfBlocks):
            c.append(1, 0, [[[7]]], [[[7]]])  # a new block for position 4, none left
        assert c.seq_len(1) == 4 and c.stats()["used"] == 2


def test_fill_counts_positions_every_layer_holds():
    # WHY: a block's fill (what tl_kv_export sends and a prefix cache may
    #      register once full) is the positions EVERY layer has written. Mid
    #      step, layer 0 is ahead of layer 1; the fill must follow the
    #      slowest layer, or an export would ship garbage for layer 1.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L8.3 section 2
    L = lib()
    with PagedKVCache(L, 4, 4, 2, 1, 1) as c:
        c.add_seq(0)
        c.append(0, 0, np.ones((1, 6, 1)), np.ones((1, 6, 1)))
        t = c.block_table(0)
        assert [L.tl_kv_fill(c.pool, int(b)) for b in t] == [0, 0]
        c.append(0, 1, np.ones((1, 5, 1)), np.ones((1, 5, 1)))
        assert [L.tl_kv_fill(c.pool, int(b)) for b in t] == [4, 1]
        c.append(0, 1, np.ones((1, 1, 1)), np.ones((1, 1, 1)))
        assert [L.tl_kv_fill(c.pool, int(b)) for b in t] == [4, 2]


def test_bad_arguments():
    # WHY: the cache is driven by an engine loop, where a wrong shape or a
    #      stale sequence id is a bug to report at once, not memory to
    #      corrupt: shapes other than [n_kv_heads, T >= 1, d_head], unknown
    #      ids, a reused id, a bad layer, and sizes below 1 are refused.
    # KIND: boundary
    # CATCHES: s13
    # CHAPTER: L8.3 section 4
    with pytest.raises(ValueError):
        PagedKVCache(lib(), 0, 2, 1, 1, 1)
    with PagedKVCache(lib(), 4, 2, 2, 2, 3) as c:
        c.add_seq(0)
        with pytest.raises(ValueError):
            c.add_seq(0)
        with pytest.raises(ValueError):
            c.append(0, 0, np.ones((2, 3)), np.ones((2, 3)))
        with pytest.raises(ValueError):
            c.append(0, 0, np.ones((3, 1, 3)), np.ones((3, 1, 3)))
        with pytest.raises(ValueError):
            c.append(0, 0, np.ones((2, 0, 3)), np.ones((2, 0, 3)))
        with pytest.raises(ValueError):
            c.append(0, 2, np.ones((2, 1, 3)), np.ones((2, 1, 3)))
        with pytest.raises(KeyError):
            c.append(5, 0, np.ones((2, 1, 3)), np.ones((2, 1, 3)))
        with pytest.raises(KeyError):
            c.gather(5, 0)
        with pytest.raises(KeyError):
            c.fork(5, 6)
        with pytest.raises(ValueError):
            c.fork(0, 0)
        assert c.stats()["used"] == 0


class ToyDecoder:
    """Two attention-only layers with residuals: enough to make every cached
    K and V matter for the next token's logits."""

    def __init__(self, g: PCG32, V=13, d=8, H=2, D=4, L=2):
        self.V, self.d, self.H, self.D, self.L = V, d, H, D, L
        self.E = rand(g, (V, d), 1.5)
        self.W = [[rand(g, (d, H * D)) for _ in range(4)] for _ in range(L)]  # q, k, v, o^T
        self.U = rand(g, (d, V), 2.0)

    def step(self, tok, put, get) -> np.ndarray:
        x = self.E[tok].astype(np.float32)
        for layer, (Wq, Wk, Wv, Wo) in enumerate(self.W):
            q = (x @ Wq).reshape(self.H, self.D)
            put(layer, (x @ Wk).reshape(self.H, 1, self.D), (x @ Wv).reshape(self.H, 1, self.D))
            K, V = get(layer)
            K, V = K.astype(np.float32), V.astype(np.float32)
            s = np.einsum("hd,htd->ht", q, K) / np.sqrt(self.D)
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            o = np.einsum("ht,htd->hd", p, V).reshape(-1)
            x = x + o @ Wo.T
        return x @ self.U


class Contiguous82:
    """L8.2's KVCache (batch 1, float16) behind the decoder's put/get hooks;
    clone() replays the appends into a fresh cache (a fork, the slow way)."""

    def __init__(self, L, H, D):
        self.dims = (L, H, D)
        self.c = KVCache(L, H, D, max_len=64, batch=1, dtype=np.float16)
        self.last: dict = {}
        self.log: list = []

    def put(self, layer, k, v):
        K, V = self.c.update(layer, k[None], v[None])
        self.last[layer] = (K[0], V[0])
        self.log.append((layer, k, v))

    def get(self, layer):
        return self.last[layer]

    def clone(self):
        new = Contiguous82(*self.dims)
        for layer, k, v in self.log:
            new.put(layer, k, v)
        return new


def test_paged_matches_contiguous_decoding():
    # WHY: the catalog's differential: a decoder fed from the paged cache and
    #      the same decoder fed from your contiguous L8.2 KVCache with
    #      dtype=float16 give logits within 1e-5 and identical greedy ids at
    #      every step, for a 10-token prompt and for two branches forked
    #      from it mid-block (one greedy, one forced onto another first
    #      token), as beam search and parallel sampling use it. MS-L8 checks
    #      the same equivalence end to end on a real model.
    # KIND: differential
    # CATCHES: s02, s07
    # CHAPTER: L8.3 section 4
    g = PCG32(SEED, 13)
    m = ToyDecoder(g)
    prompt = [g.below(m.V) for _ in range(10)]  # 9 fed before the fork: block 2 is shared and partial
    refs = {0: Contiguous82(m.L, m.H, m.D)}
    with PagedKVCache(lib(), 16, 4, m.L, m.H, m.D) as c:

        def step(seq, tok, i):
            r = refs[seq]
            lp = m.step(tok, lambda l_, k, v: c.append(seq, l_, k, v), lambda l_: c.gather(seq, l_))
            lc = m.step(tok, r.put, r.get)
            assert_close(lp, lc, rtol=0.0, atol=1e-5, msg=f"sequence {seq}, step {i}")
            assert int(np.argmax(lp)) == int(np.argmax(lc))
            return int(np.argmax(lp))

        def generate(seq, tok, n, force=None):
            ids = []
            for i in range(n):
                nxt = step(seq, tok, i)
                tok = force if (i == 0 and force is not None) else nxt
                ids.append(tok)
            return ids

        c.add_seq(0)
        for i, tok in enumerate(prompt[:-1]):
            step(0, tok, i)
        c.fork(0, 1)
        refs[1] = refs[0].clone()
        a = generate(0, prompt[-1], 6)
        b = generate(1, prompt[-1], 6, force=(a[0] + 1) % m.V)
        assert len(a) == 6 and len(b) == 6 and a != b
        for s in (0, 1):
            assert c.seq_len(s, m.L - 1) == len(prompt) - 1 + 6
            for layer in range(m.L):
                K, V = c.gather(s, layer)
                RK, RV = refs[s].get(layer)
                assert (K.astype(np.float32) == RK).all() and (V.astype(np.float32) == RV).all()


def test_random_ops_conserve_blocks():
    # WHY: the catalog's property: 10^4 seeded add, fork, append, and free
    #      operations. After every one, free + used + cached equals the pool
    #      size; every live sequence gathers exactly what the contiguous
    #      oracle holds (checked for one random sequence per step and all of
    #      them at the end); and freeing everything restores every block, so
    #      no refcount went negative or leaked.
    # KIND: property
    # CATCHES: s04, s05, s06, s07, s08, s09
    # CHAPTER: L8.3 section 4
    g = PCG32(SEED, 14)
    L_, H, B, D, N = 2, 1, 3, 2, 24
    ref = Contiguous(L_, H, D)
    live: list[int] = []
    nxt = 0
    with PagedKVCache(lib(), N, B, L_, H, D) as c:
        for _ in range(10_000):
            op = g.below(10)
            if op < 2 or not live:
                c.add_seq(nxt)
                ref.add(nxt)
                live.append(nxt)
                nxt += 1
            elif op < 3:
                p = live[g.below(len(live))]
                try:
                    c.fork(p, nxt)
                    ref.fork(p, nxt)
                    live.append(nxt)
                except OutOfBlocks:
                    pass
                nxt += 1
            elif op < 8:
                s = live[g.below(len(live))]
                T = 1 + g.below(4)
                k = np.full((H, T, D), float(g.below(1000)), np.float32)
                k[0, :, 0] += np.arange(T)
                try:
                    for layer in range(L_):
                        c.append(s, layer, k, -k)
                        ref.append(s, layer, k, -k)
                except OutOfBlocks:  # an engine preempts: free the sequence
                    c.free(s)
                    ref.free(s)
                    live.remove(s)
            else:
                s = live.pop(g.below(len(live)))
                c.free(s)
                ref.free(s)
            st = c.stats()
            assert st["free"] + st["used"] + st["cached"] == N
            if live:
                s = live[g.below(len(live))]
                layer = g.below(L_)
                K, V = c.gather(s, layer)
                assert (K == ref.k[s, layer]).all() and (V == ref.v[s, layer]).all()
        for s in live:
            for layer in range(L_):
                K, _ = c.gather(s, layer)
                assert (K == ref.k[s, layer]).all()
        for s in live:
            c.free(s)
        assert c.stats()["free"] == N
