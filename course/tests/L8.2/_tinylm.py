"""Test models for the L8.2 course tests (a helper module: no tests here).

TinyLM is a two-layer Llama-shaped decoder in float32 numpy, written for the
tests and independent of your L7 modules: token embedding, RMSNorm,
grouped-query attention (4 query heads over 2 kv heads, head dim 8) with
RoPE (half layout, base 10000) at absolute positions, a SiLU-gated MLP, and
a tied output head. With a cache it calls cache.update(layer, k, v) once per
layer and attends to what comes back, and it reads positions and the mask
from the cache (cache.positions, cache.mask) when the caller passes none,
exactly the contract L7.9's model follows. Weights come from the frozen
PCG32, so every run is identical.

ScriptLM ignores attention and writes logits that make a fixed script of ids
come out greedily: at absolute position j it puts margin 8 on script[j + 1].
It still calls cache.update on every layer, so caches advance as with a real
model.
"""

from __future__ import annotations

import numpy as np
from _lib.pcg32 import PCG32


def rms(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w).astype(
        np.float32
    )


def rope(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """x [B, H, T, dh], half layout: pairs (i, i + dh/2) rotated by pos * base^(-2i/dh)."""
    dh = x.shape[-1]
    inv = (10000.0 ** (-np.arange(0, dh, 2, dtype=np.float32) / dh)).astype(np.float32)
    ang = pos.astype(np.float32)[:, None] * inv[None, :]
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[..., : dh // 2], x[..., dh // 2 :]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1).astype(np.float32)


class TinyLM:
    def __init__(
        self,
        vocab: int,
        seed: int = 0,
        d: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        d_head: int = 8,
        d_ff: int = 48,
        max_len: int = 128,
    ):
        g = PCG32(seed, 77)
        r = lambda *shape: (g.normal_array(shape) * 0.3).astype(np.float32)  # noqa: E731
        self.vocab, self.d, self.n_heads = vocab, d, n_heads
        self.n_layers, self.n_kv_heads, self.d_head, self.max_len = (
            n_layers,
            n_kv_heads,
            d_head,
            max_len,
        )
        self.emb = r(vocab, d)
        self.layers = [
            dict(
                n1=1 + r(d) * 0.1,
                wq=r(d, n_heads * d_head),
                wk=r(d, n_kv_heads * d_head),
                wv=r(d, n_kv_heads * d_head),
                wo=r(n_heads * d_head, d),
                n2=1 + r(d) * 0.1,
                wg=r(d, d_ff),
                wu=r(d, d_ff),
                wd=r(d_ff, d),
            )
            for _ in range(n_layers)
        ]
        self.nf = 1 + r(d) * 0.1
        self.calls = []  # (T, positions) per forward call

    def forward(self, ids, positions=None, cache=None):
        ids = np.asarray(ids)
        B, T = ids.shape
        if positions is None:
            positions = cache.positions(T) if cache is not None else np.arange(T)
        positions = np.asarray(positions)
        mask = (
            cache.mask(T) if cache is not None else np.tril(np.ones((T, T), dtype=bool))
        )
        self.calls.append((T, positions.tolist()))
        H, Hkv, dh = self.n_heads, self.n_kv_heads, self.d_head
        x = self.emb[ids]
        for li, L in enumerate(self.layers):
            h = rms(x, L["n1"])
            q = (h @ L["wq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            k = (h @ L["wk"]).reshape(B, T, Hkv, dh).transpose(0, 2, 1, 3)
            v = (h @ L["wv"]).reshape(B, T, Hkv, dh).transpose(0, 2, 1, 3)
            q, k = rope(q, positions), rope(k, positions)
            if cache is not None:
                k, v = cache.update(li, k, v)
            k = np.repeat(k, H // Hkv, axis=1)
            v = np.repeat(v, H // Hkv, axis=1)
            s = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(dh))
            s = np.where(mask, s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w = (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
            o = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, H * dh)
            x = (x + o @ L["wo"]).astype(np.float32)
            h = rms(x, L["n2"])
            gate = h @ L["wg"]
            x = (x + ((gate / (1 + np.exp(-gate))) * (h @ L["wu"])) @ L["wd"]).astype(
                np.float32
            )
        return (rms(x, self.nf) @ self.emb.T).astype(np.float32)


class ScriptLM:
    """Greedy output = script[len(prompt):], one id per step."""

    def __init__(
        self, vocab: int, script: list[int], n_layers: int = 2, max_len: int = 128
    ):
        self.vocab, self.script = vocab, list(script)
        self.n_layers, self.n_kv_heads, self.d_head, self.max_len = (
            n_layers,
            1,
            2,
            max_len,
        )

    def forward(self, ids, positions=None, cache=None):
        ids = np.asarray(ids)
        B, T = ids.shape
        if positions is None:
            positions = cache.positions(T) if cache is not None else np.arange(T)
        if cache is not None:
            for li in range(self.n_layers):
                cache.update(
                    li,
                    np.zeros((B, 1, T, 2), np.float32),
                    np.zeros((B, 1, T, 2), np.float32),
                )
        out = np.zeros((B, T, self.vocab), dtype=np.float32)
        for t, pos in enumerate(np.asarray(positions).tolist()):
            nxt = pos + 1
            if nxt < len(self.script):
                out[:, t, self.script[nxt]] = 8.0
        return out
