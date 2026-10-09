"""Toy causal LMs and a toy tokenizer for the L8.6 course tests.

BagLM is built so that chunking cannot change a single bit: each position's
key and value are its token's embedding (stored in the KVCache as float32),
and the logits at absolute position t are W @ tanh(mean of the cached keys
0..t + a position embedding), computed row by row in float64. So the logits
at a position depend only on the tokens up to it, whether the target scores
one token or five, and a stale key left in the cache after a rejected draft
changes them. That makes "speculative greedy == plain greedy" an exact test.
"""

from __future__ import annotations

import numpy as np
from _lib.pcg32 import PCG32


class BagLM:
    def __init__(self, vocab: int = 8, d: int = 6, max_len: int = 96, seed: int = 0, scale: float = 3.0) -> None:
        rng = PCG32(seed)
        self.vocab, self.d = vocab, d
        self.n_layers, self.n_kv_heads, self.d_head, self.max_len = 1, 1, d, max_len
        self.emb = rng.normal_array((vocab, d))
        self.pos = 0.5 * rng.normal_array((max_len, d))
        self.W = scale * rng.normal_array((vocab, d))
        self.calls: list[tuple[list[int], list[int]]] = []  # (positions, ids) per forward

    def forward(self, ids, positions=None, cache=None):
        ids = np.asarray(ids, dtype=np.int64)
        assert ids.ndim == 2 and ids.shape[0] == 1
        T = ids.shape[1]
        pos = np.arange(T) if positions is None else np.asarray(positions, dtype=np.int64)
        self.calls.append((pos.tolist(), ids[0].tolist()))
        k = self.emb[ids[0]].astype(np.float32)[None, None]  # [1, 1, T, d]
        if cache is not None:
            K, _ = cache.update(0, k, k)
            base = int(pos[0])
            assert K.shape[2] == base + T, "the cache must hold exactly the positions before this chunk"
        else:
            K, base = k, 0
            assert int(pos[0]) == 0
        keys = np.asarray(K, dtype=np.float64)[0, 0]
        out = np.empty((1, T, self.vocab))
        for j in range(T):
            t = base + j
            h = np.tanh(keys[: t + 1].mean(axis=0) + self.pos[t])
            out[0, j] = [float(np.dot(w, h)) for w in self.W]
        return out


class Letters:
    """ids 0..V-1 are the letters a, b, c, ...; no special tokens."""

    def __init__(self, vocab: int = 8) -> None:
        self.vocab_size = vocab
        self.special_ids: dict[str, int] = {}
        self.unk_id = None

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        return [ord(c) - 97 for c in text]

    def decode(self, ids, skip_special: bool = False) -> str:
        return "".join(chr(97 + int(i)) for i in ids)


def greedy_reference(model: BagLM, prompt: list[int], n: int) -> list[int]:
    """Plain greedy decoding without any cache: argmax, ties to the lowest id."""
    ids = list(prompt)
    for _ in range(n):
        if len(ids) >= model.max_len:
            break
        logits = model.forward(np.asarray([ids]))[0, -1]
        ids.append(int(np.argmax(logits)))
    return ids[len(prompt) :]
