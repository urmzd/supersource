"""Course tests for L7.7: sliding windows and attention sinks (tinyllm/modern/window.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.7), and the chapter section it comes
from.

The worked example of the chapter (section 3): window 2 and one sink token
over 5 positions. Query 3 reads keys 0 (the sink), 2, and 3; query 4 reads
0, 3, and 4. A SinkWindowCache(1, 2) fed one token at a time returns the
keys at positions [0], [0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 4] and then
holds [0, 4]. One query at position 1 with scores (0, ln 2) over values (4, 8) and a sink
logit 0 has weights (1/4, 1/2), gives 1/4 to the sink, and outputs 5;
lse = ln 4.

The golden fixture (course/fixtures/L7.7/window_hf.npz) holds a
transformers 5.19.0 MistralAttention layer with sliding_window 3 and
gpt-oss's eager attention with learned sinks under window and causal
masks, float32, from course/oracle/L7.7/window_hf.py.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.gqa import ConcatKVCache, GQAttention
from tinyllm.modern.rope import RopeSpec
from tinyllm.modern.window import (
    SinkWindowCache,
    attention_mask,
    sink_window_mask,
    windowed_attention,
)
from tinyllm.xfmr.masks import causal_mask, sliding_window_mask

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.7", "window_hf.npz")


def ladder(r: int, base: float = 10000.0) -> np.ndarray:
    return base ** (-np.arange(0, r, 2, dtype=np.float64) / r)


def gqa(
    seed: int = 0, d: int = 16, H: int = 4, Hkv: int = 2, dh: int = 4
) -> GQAttention:
    spec = RopeSpec(
        inv_freq=ladder(dh), attention_scaling=1.0, layout="half", rotary_dim=dh
    )
    m = GQAttention(d, H, Hkv, dh, spec)
    g = PCG32(seed=200 + seed)
    for _, p in m.named_parameters():
        p.data = 0.5 * g.normal_array(p.shape)
    return m


def stream(m: GQAttention, x: np.ndarray, cache, chunks, window, n_sink) -> np.ndarray:
    """Run x through m chunk by chunk with cache and the window/sink mask."""
    outs, t = [], 0
    for n in chunks:
        mask = attention_mask(cache, 0, n, window, n_sink)
        outs.append(
            m(
                Tensor(x[:, t : t + n], dtype=x.dtype),
                np.arange(t, t + n),
                mask=mask,
                cache=cache,
            ).data
        )
        t += n
    return np.concatenate(outs, axis=1)


# --- the worked example ------------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: the mask with window 2 and one
    #      sink, what a SinkWindowCache returns and keeps at every step, and
    #      one learned-sink softmax row worked on paper.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, s07
    # CHAPTER: L7.7 section 3, Worked example by hand
    want = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(sink_window_mask(5, 5, window=2, n_sink=1), want)
    cache = SinkWindowCache(1, 2)
    seen = []
    for t in range(5):
        k = np.full((1, 1, 1, 1), float(t))
        ka, _ = cache.update(0, k, k)
        seen.append(cache.positions(0).tolist())
        assert ka[0, 0, :, 0].tolist() == [float(p) for p in seen[-1]]
    assert seen == [[0], [0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 4]]
    assert cache.held(0) == 2 and cache.seq_len(0) == 5
    q = np.array([[[[1.0]]]])
    k = np.array([[[[0.0], [math.log(2.0)]]]])
    v = np.array([[[[4.0], [8.0]]]])
    out, lse = windowed_attention(
        q, k, v, sink_logits=[0.0], q_offset=1
    )  # the query sits at position 1
    assert_close(out, [[[[5.0]]]], rtol=1e-12, atol=1e-12)
    assert_close(lse, [[[math.log(4.0)]]], rtol=1e-12, atol=1e-12)


# --- masks ---------------------------------------------------------------------------------


def test_mask_agrees_with_l5_2():
    # WHY: without sinks the rule is Mistral's sliding window (L5.2's
    #      sliding_window_mask), and without a window it is the causal mask;
    #      a decode row at q_offset obeys the same rule as a prefill row.
    # KIND: differential
    # CATCHES: s01, s02
    # CHAPTER: L7.7 section 2.1, The visibility rule
    for Tq, Tk, w, off in ((6, 6, 3, 0), (2, 9, 4, 7), (1, 12, 1, 11), (5, 8, 20, 3)):
        assert np.array_equal(
            sink_window_mask(Tq, Tk, w, 0, off), sliding_window_mask(Tq, Tk, w, off)
        )
        assert np.array_equal(
            sink_window_mask(Tq, Tk, None, 3, off), causal_mask(Tq, Tk, off)
        )
    m = sink_window_mask(9, 9, 3, 2)
    assert m.sum(axis=1).max() <= 2 + 3 and m[8].tolist() == [1, 1, 0, 0, 0, 0, 1, 1, 1]


# --- against Hugging Face ----------------------------------------------------------------------


@pytest.mark.parametrize("case", ["win3", "chunk", "full"])
def test_learned_sinks_golden(case):
    # WHY: gpt-oss's eager attention: kv heads repeated in blocks
    #      (repeat_kv), one learned sink logit per head in every row's
    #      denominator, under a window of 3, a decode-like chunk of the last
    #      3 queries (q_offset 4), and the plain causal mask. L9.3's kernel
    #      is checked against this function, its lse included.
    # KIND: golden
    # CATCHES: s01, s05, s06, s10
    # CHAPTER: L7.7 section 2.3, Learned sinks
    f = np.load(FIX)
    window, off, q = {
        "win3": (3, 0, f["gptoss.q"]),
        "chunk": (3, 4, f["gptoss.q"][:, :, 4:]),
        "full": (None, 0, f["gptoss.q"]),
    }[case]
    out, lse = windowed_attention(
        q,
        f["gptoss.k"],
        f["gptoss.v"],
        window=window,
        sink_logits=f["gptoss.sinks"],
        q_offset=off,
    )
    assert out.dtype == np.float64 and out.shape == f[f"gptoss.{case}.out"].shape
    assert_close(out, f[f"gptoss.{case}.out"], rtol=1e-4, atol=1e-5, msg="out")
    assert_close(lse, f[f"gptoss.{case}.lse"], rtol=1e-5, atol=1e-5, msg="lse")


def test_mistral_sliding_window_golden_streamed():
    # WHY: HF's MistralAttention with sliding_window 3 over 9 tokens. Your
    #      GQAttention (L7.5) with this module's mask gives the same output in
    #      one pass, and token by token through a SinkWindowCache(0, 3) that
    #      never holds more than 2 keys between steps.
    # KIND: golden
    # CATCHES: s01, s03, s08, s11
    # CHAPTER: L7.7 section 2.2, A cache that forgets
    f = np.load(FIX)
    spec = RopeSpec(
        inv_freq=f["mistral.inv_freq"],
        attention_scaling=1.0,
        layout="half",
        rotary_dim=6,
    )
    m = GQAttention(24, 4, 2, 6, spec)
    m.load_state_dict(
        {
            k[len("mistral.") :]: f[k]
            for k in f.files
            if k.startswith("mistral.") and k.endswith("weight")
        }
    )
    x = f["mistral.x"]
    full = m(Tensor(x), np.arange(9), mask=sink_window_mask(9, 9, 3)).data
    assert_close(full, f["mistral.out"], rtol=1e-4, atol=1e-5, msg="one pass")
    cache = SinkWindowCache(0, 3)
    streamed = stream(m, x, cache, [1] * 9, 3, 0)
    assert cache.held(0) == 2
    assert_close(streamed, f["mistral.out"], rtol=1e-4, atol=1e-5, msg="token by token")


# --- streaming equals full attention ------------------------------------------------------------


@pytest.mark.parametrize("chunks", [[1] * 12, [5, 3, 4], [2, 1, 6, 3]])
def test_streaming_with_sinks_equals_full_attention(chunks):
    # WHY: StreamingLLM's promise, made exact: with 2 sink tokens and a
    #      window of 4, decoding token by token or prefilling in chunks
    #      through the bounded cache gives the outputs of full attention
    #      under the sink-window mask (float64, GQA with 4 query and 2 kv
    #      heads).
    # KIND: differential
    # CATCHES: s01, s02, s03, s04, s08, s11
    # CHAPTER: L7.7 section 2.2, A cache that forgets
    m = gqa(seed=1)
    x = PCG32(seed=21).normal_array((2, 12, 16))
    full = m(
        Tensor(x, dtype=np.float64), np.arange(12), mask=sink_window_mask(12, 12, 4, 2)
    ).data
    cache = SinkWindowCache(2, 4)
    got = stream(m, x, cache, chunks, 4, 2)
    assert_close(got, full, rtol=1e-9, atol=1e-11)
    assert cache.held(0) <= 2 + 4 - 1


def test_plain_cache_uses_absolute_positions():
    # WHY: a cache that keeps every key (L7.5's ConcatKVCache, L8.2's
    #      KVCache) needs the window mask placed at the chunk's absolute
    #      positions (q_offset = what the cache holds); the result must equal
    #      the bounded cache's.
    # KIND: differential
    # CATCHES: s09
    # CHAPTER: L7.7 section 4, The interface
    m = gqa(seed=2)
    x = PCG32(seed=22).normal_array((1, 10, 16))
    full = m(
        Tensor(x, dtype=np.float64), np.arange(10), mask=sink_window_mask(10, 10, 3, 1)
    ).data
    got = stream(m, x, ConcatKVCache(), [4, 1, 5], 3, 1)
    assert_close(got, full, rtol=1e-9, atol=1e-11)


# --- the cache ------------------------------------------------------------------------------------


def test_cache_is_bounded():
    # WHY: the reason to evict: 200 tokens through SinkWindowCache(4, 8)
    #      never hold more than 4 + 8 - 1 keys between steps, and what each
    #      step returns is exactly the keys the last query may read (the
    #      sinks and the window). The cache keeps copies of what it is given.
    # KIND: property
    # CATCHES: s01, s02, s03, s04, s07
    # CHAPTER: L7.7 section 2.2, A cache that forgets
    cache = SinkWindowCache(4, 8)
    g = PCG32(seed=23)
    for t in range(200):
        k = g.normal_array((1, 2, 1, 3))
        ka, va = cache.update(0, k, k + 1.0)
        k[:] = 99.0
        pos = cache.positions(0)
        assert np.array_equal(
            pos, np.flatnonzero(sink_window_mask(1, t + 1, 8, 4, t)[0])
        )
        assert ka.shape[2] == pos.size and va.shape[2] == pos.size
        assert cache.held(0) <= 4 + 8 - 1 and cache.seq_len(0) == t + 1
    assert float(np.max(ka)) < 99.0


def test_attention_mask_dispatch():
    # WHY: one function gives every attention layer its mask: none without a
    #      window, the full-sequence mask without a cache, the bounded
    #      cache's own mask, or the mask at the chunk's offset for a cache
    #      that keeps everything.
    # KIND: unit
    # CATCHES: s09, s11
    # CHAPTER: L7.7 section 4, The interface
    assert attention_mask(None, 0, 5, None, 2) is None
    assert np.array_equal(
        attention_mask(None, 0, 5, 3, 1), sink_window_mask(5, 5, 3, 1)
    )
    sw = SinkWindowCache(1, 3)
    for _ in range(4):
        sw.update(0, np.zeros((1, 1, 1, 2)), np.zeros((1, 1, 1, 2)))
    assert np.array_equal(attention_mask(sw, 0, 2, 3, 1), sw.chunk_mask(0, 2))
    assert sw.chunk_mask(0, 2).shape == (2, sw.held(0) + 2)
    plain = ConcatKVCache()
    plain.update(0, np.zeros((1, 1, 3, 2)), np.zeros((1, 1, 3, 2)))
    assert np.array_equal(
        attention_mask(plain, 0, 2, 3, 1), sink_window_mask(2, 5, 3, 1, 3)
    )
    with pytest.raises(ValueError):
        attention_mask(object(), 0, 2, 3, 1)


def test_sink_mass_and_empty_rows():
    # WHY: a learned sink takes p_sink = exp(sink - lse) of every row, so the
    #      weights sum to 1 - p_sink (with values of ones the output IS that
    #      sum); a row that sees no key gives 0 with lse -inf without a sink,
    #      and 0 with lse = the sink logit with one, never NaN.
    # KIND: property
    # CATCHES: s05
    # CHAPTER: L7.7 section 2.3, Learned sinks
    g = PCG32(seed=24)
    q, k = g.normal_array((1, 2, 4, 3)), g.normal_array((1, 1, 4, 3))
    sinks = np.array([0.5, -1.0])
    out, lse = windowed_attention(
        q, k, np.ones((1, 1, 4, 1)), window=2, sink_logits=sinks
    )
    assert_close(
        out[..., 0], 1.0 - np.exp(sinks[None, :, None] - lse), rtol=1e-12, atol=1e-12
    )
    plain, _ = windowed_attention(q, k, np.ones((1, 1, 4, 1)), window=2)
    assert_close(plain, np.ones_like(plain), rtol=1e-12, atol=1e-12)
    # a query at position 5 with window 2 sees nothing among keys 0..2
    out, lse = windowed_attention(
        q[:, :, :1], k[:, :, :3], np.ones((1, 1, 3, 1)), window=2, q_offset=5
    )
    assert np.isfinite(out).all() and np.all(out == 0) and np.all(np.isneginf(lse))
    out, lse = windowed_attention(
        q[:, :, :1],
        k[:, :, :3],
        np.ones((1, 1, 3, 1)),
        window=2,
        q_offset=5,
        sink_logits=sinks,
    )
    assert np.all(out == 0) and np.array_equal(lse[0, :, 0], sinks)


def test_validation():
    # WHY: a zero window, negative sinks, mismatched head counts, and a sink
    #      vector of the wrong length are config bugs; fail loudly.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: L7.7 section 4, The interface
    for args in ((0, 3, 2), (3, 0, 2), (3, 3, 0), (3, 3, 2, -1)):
        with pytest.raises(ValueError):
            sink_window_mask(*args)
    for bad in ((-1, 4), (2, 0), (1.5, 4)):
        with pytest.raises(ValueError):
            SinkWindowCache(*bad)
    q, k = np.zeros((1, 3, 2, 4)), np.zeros((1, 2, 2, 4))
    with pytest.raises(ValueError):
        windowed_attention(q, k, k)  # 3 query heads over 2 kv heads
    with pytest.raises(ValueError):
        windowed_attention(q[:, :2], k, k, sink_logits=[0.0])
    cache = SinkWindowCache(1, 2)
    cache.update(0, np.zeros((1, 2, 1, 4)), np.zeros((1, 2, 1, 4)))
    with pytest.raises(ValueError):
        cache.update(0, np.zeros((1, 2, 1, 5)), np.zeros((1, 2, 1, 5)))
    with pytest.raises(ValueError):
        cache.chunk_mask(0, 0)
