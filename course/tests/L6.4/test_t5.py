"""Course tests for L6.4 (optional): T5 span corruption and relative position
buckets (tinyllm/obj/t5.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L6.4), and the chapter section it comes
from.

The worked examples of the chapter (section 3):
  * 8 tokens 10..17, noise_density 0.25, mean span 2: n_noise = 2,
    n_spans = 1, so the last two tokens are the span: inputs
    (10 11 12 13 14 15 S0), targets (S0 16 17), with S0 = sentinel_start_id.
  * 10 tokens 0..9, noise_density 0.4, mean span 2, and the scripted draws
    below(3) = 0, below(2) = 1 (noise lengths 3, 1), below(5) = 2,
    below(4) = 0, below(3) = 2, below(2) = 0 (keep lengths 4, 2): inputs
    (0 1 2 3 S0 7 8 S1), targets (S0 4 5 6 S1 9), S1 = S0 - 1.
  * buckets, bidirectional, 32 buckets, max_distance 128: rel -3 -> 3,
    +3 -> 19, -20 -> 10, +200 -> 31, 0 -> 0; causal (32 buckets): +5 -> 0,
    -5 -> 5.

The golden fixture (course/fixtures/L6.4/t5_buckets.npz) holds Hugging Face
transformers 5.19.0 T5Attention._relative_position_bucket over -300..300 for
five settings and compute_bias with copied weights for an encoder and a
decoder with past tokens (course/oracle/L6.4/t5_golden.py).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.obj.t5 import (
    T5RelativeBias,
    noise_span_counts,
    random_spans_noise_mask,
    span_corrupt,
    t5_relative_bucket,
)

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L6.4", "t5_buckets.npz")
S0 = 99


class Rng:
    """The frozen PCG32 behind the generator API of M06.3; counts draws."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)
        self.calls: list[int] = []

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def below(self, n: int) -> int:
        self.calls.append(n)
        return self.g.below(n)


class Scripted:
    """below(n) returns the scripted values in order and records each n."""

    def __init__(self, values) -> None:
        self.values = list(values)
        self.calls: list[int] = []

    def below(self, n: int) -> int:
        self.calls.append(n)
        v = self.values.pop(0)
        assert 0 <= v < n
        return v


def restore(inputs, targets, start: int) -> list[int]:
    """The test's own inverse: each sentinel in inputs becomes the tokens
    that follow the same sentinel in targets."""
    spans, cur = {}, None
    for t in targets.tolist():
        if start - 32 < t <= start:
            cur = t
            spans[cur] = []
        else:
            spans[cur].append(t)
    out = []
    for t in inputs.tolist():
        out.extend(spans[t] if t in spans else [t])
    return out


# --- the worked examples ---------------------------------------------------------------


def test_hand_example_one_span():
    # WHY: the chapter's first worked example: with one span there is no
    #      choice to make, the text keeps its first tokens and the last
    #      n_noise tokens are the span, behind sentinel 0.
    # KIND: unit
    # CATCHES: s05, s06, m04
    # CHAPTER: L6.4 section 3, Worked example by hand
    ids = np.arange(10, 18)
    assert noise_span_counts(8, 0.25, 2.0) == (2, 1)
    inp, tgt = span_corrupt(ids, 0.25, 2.0, S0, Rng(0))
    assert inp.tolist() == [10, 11, 12, 13, 14, 15, S0]
    assert tgt.tolist() == [S0, 16, 17]
    inp, tgt = span_corrupt(ids, 0.25, 2.0, S0, Rng(0), eos_id=1)
    assert inp.tolist() == [10, 11, 12, 13, 14, 15, S0, 1]
    assert tgt.tolist() == [S0, 16, 17, 1]


def test_hand_example_two_spans_scripted_draws():
    # WHY: the second worked example pins the whole algorithm: the noise
    #      lengths are segmented first, then the kept lengths, each by
    #      Fisher-Yates from the end (below(i + 1) for i = m - 2 down to 1),
    #      and sentinel ids count DOWN from sentinel_start_id.
    # KIND: unit
    # CATCHES: s04, s05, s06, s07, s08
    # CHAPTER: L6.4 section 3, Worked example by hand
    r = Scripted([0, 1, 2, 0, 2, 0])
    inp, tgt = span_corrupt(np.arange(10), 0.4, 2.0, S0, r)
    assert r.calls == [3, 2, 5, 4, 3, 2]
    assert inp.tolist() == [0, 1, 2, 3, S0, 7, 8, S0 - 1]
    assert tgt.tolist() == [S0, 4, 5, 6, S0 - 1, 9]


def test_hand_example_buckets():
    # WHY: the chapter's bucket table: exact buckets for |rel| < 8, the
    #      logarithmic bucket 10 for rel = -20, keys after the query in the
    #      upper half, the cap at the last bucket, and a causal decoder that
    #      folds every future key into bucket 0.
    # KIND: unit
    # CATCHES: s02, s09, m01, m02
    # CHAPTER: L6.4 section 3, Worked example by hand
    got = t5_relative_bucket(np.array([-3, 3, -20, 200, 0]), bidirectional=True)
    assert got.tolist() == [3, 19, 10, 31, 0]
    assert got.dtype == np.int64
    assert t5_relative_bucket(np.array([5, -5]), bidirectional=False).tolist() == [0, 5]


# --- span corruption -------------------------------------------------------------------


def test_inputs_and_targets_restore_the_text():
    # WHY: span corruption loses nothing: putting each span back in place of
    #      its sentinel gives the original ids, for any length, density, and
    #      seed. This is what makes the objective self-supervised.
    # KIND: property
    # CATCHES: s04, s05
    # CHAPTER: L6.4 section 2.1, Span corruption
    for seed in range(40):
        g = PCG32(seed=seed + 100)
        L = 2 + g.below(60)
        dens = [0.15, 0.3, 0.5][seed % 3]
        ids = np.array([g.below(50) for _ in range(L)])
        inp, tgt = span_corrupt(ids, dens, 3.0, S0, Rng(seed))
        assert restore(inp, tgt, S0) == ids.tolist(), (seed, L)


def test_counts_and_lengths():
    # WHY: the number of noise tokens and spans is fixed by L, the density,
    #      and the mean span (round half to EVEN, as numpy and T5), and the
    #      mask has exactly those: it starts with a kept token and ends with
    #      a noise span; the sequence lengths follow.
    # KIND: property
    # CATCHES: s03, s05, s06
    # CHAPTER: L6.4 section 2.1, Span corruption
    assert noise_span_counts(10, 0.25, 3.0) == (2, 1)  # 2.5 rounds to 2
    assert noise_span_counts(14, 0.25, 3.0) == (4, 1)  # 3.5 rounds to 4
    assert noise_span_counts(512, 0.15, 3.0) == (77, 26)
    assert noise_span_counts(2, 0.9, 3.0) == (1, 1)
    for L, dens, mean in [
        (512, 0.15, 3.0),
        (37, 0.3, 2.0),
        (9, 0.5, 1.0),
        (3, 0.5, 5.0),
    ]:
        n_noise, n_spans = noise_span_counts(L, dens, mean)
        m = random_spans_noise_mask(L, dens, mean, Rng(L))
        assert m.dtype == bool and m.shape == (L,)
        assert int(m.sum()) == n_noise
        starts = int(np.sum(m[1:] & ~m[:-1]) + m[0])
        assert starts == n_spans
        assert not m[0] and m[-1]
        inp, tgt = span_corrupt(np.arange(L), dens, mean, 1000, Rng(L))
        assert len(inp) == L - n_noise + n_spans
        assert len(tgt) == n_noise + n_spans


def test_short_texts_never_make_empty_spans():
    # WHY: T5's formula asks for round(n_noise / mean) spans, which on a
    #      short text can exceed the tokens available for spans or for the
    #      gaps between them; the count is clamped so every span and every
    #      gap holds at least one token.
    # KIND: boundary
    # CATCHES: s06, s10
    # CHAPTER: L6.4 section 5, Pitfalls
    assert noise_span_counts(10, 0.9, 1.0) == (9, 1)
    assert noise_span_counts(6, 0.5, 1.0) == (3, 3)
    for seed in range(10):
        m = random_spans_noise_mask(10, 0.9, 1.0, Rng(seed))
        assert m.tolist() == [False] + [True] * 9
        m = random_spans_noise_mask(6, 0.5, 1.0, Rng(seed))
        assert m.tolist() == [False, True] * 3


def test_draws_follow_the_spec():
    # WHY: one rng.below per Fisher-Yates swap, noise lengths first: the
    #      same seed gives the same corruption in every implementation, and
    #      a data loader can replay a batch from its seed.
    # KIND: unit
    # CATCHES: s07, s08
    # CHAPTER: L6.4 section 2.1, Span corruption
    r = Rng(3)
    random_spans_noise_mask(20, 0.3, 2.0, r)  # n_noise 6, n_spans 3
    assert r.calls == [5, 4, 3, 2] + list(range(13, 1, -1))
    a = span_corrupt(np.arange(30), 0.2, 3.0, S0, Rng(5))
    b = span_corrupt(np.arange(30), 0.2, 3.0, S0, Rng(5))
    assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()


def test_sentinels_and_validation():
    # WHY: sentinel k is sentinel_start_id - k, so sentinels never collide
    #      with real tokens only if the ids stay out of that range: a
    #      collision would make the targets ambiguous, so it fails loudly.
    # KIND: boundary
    # CATCHES: s04, s05, m05
    # CHAPTER: L6.4 section 5, Pitfalls
    inp, tgt = span_corrupt(np.arange(40), 0.3, 2.0, 200, Rng(1))
    sent = [t for t in tgt.tolist() if t > 150]
    assert sent == list(range(200, 200 - len(sent), -1))
    assert [t for t in inp.tolist() if t > 150] == sent
    with pytest.raises(ValueError):
        span_corrupt(np.array([5, 6, 199, 7, 8, 9]), 0.5, 1.0, 199, Rng(0))
    with pytest.raises(ValueError):
        span_corrupt(np.arange(10), 0.5, 1.0, 1, Rng(0))
    for bad in [
        dict(noise_density=0.0),
        dict(noise_density=1.0),
        dict(mean_noise_span=0.5),
    ]:
        kw = dict(noise_density=0.3, mean_noise_span=2.0) | bad
        with pytest.raises(ValueError):
            span_corrupt(np.arange(10), sentinel_start_id=99, rng=Rng(0), **kw)
    with pytest.raises(ValueError):
        span_corrupt(np.array([3]), 0.5, 1.0, 99, Rng(0))


# --- relative position buckets ------------------------------------------------------------


@pytest.mark.parametrize(
    "bi,nb,md",
    [
        (True, 32, 128),
        (False, 32, 128),
        (True, 8, 20),
        (False, 16, 64),
        (True, 64, 256),
    ],
)
def test_buckets_match_hf(bi, nb, md):
    # WHY: a T5 checkpoint's bias table is indexed by THESE buckets; one
    #      bucket off anywhere in -300..300 and pretrained attention reads
    #      the wrong bias. Compared with Hugging Face's own function.
    # KIND: golden
    # CATCHES: s02, s09, m01, m02
    # CHAPTER: L6.4 section 2.2, Relative position buckets
    f = np.load(FIX)
    got = t5_relative_bucket(
        f["rel"], bidirectional=bi, num_buckets=nb, max_distance=md
    )
    want = f[f"bucket.{int(bi)}.{nb}.{md}"]
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, (
        f"rel {f['rel'][bad[:5]].tolist()}: got {got[bad[:5]].tolist()}, HF {want[bad[:5]].tolist()}"
    )


def test_bucket_properties():
    # WHY: buckets are a monotone coarsening of distance: exact while small,
    #      never decreasing as |rel| grows, every bucket in [0, n), and the
    #      two directions use disjoint halves when bidirectional.
    # KIND: property
    # CATCHES: s02, s09, m01
    # CHAPTER: L6.4 section 2.2, Relative position buckets
    r = np.arange(0, 1000)
    for nb, md in [(32, 128), (16, 50)]:
        past = t5_relative_bucket(-r, True, nb, md)
        fut = t5_relative_bucket(r, True, nb, md)
        assert np.all(np.diff(past) >= 0) and np.all(np.diff(fut[1:]) >= 0)
        assert (
            past.max() == nb // 2 - 1
            and fut[1:].min() == nb // 2 + 1
            and fut.max() == nb - 1
        )
        assert past[: nb // 4].tolist() == list(range(nb // 4))
        causal = t5_relative_bucket(-r, False, nb, md)
        assert causal.min() == 0 and causal.max() == nb - 1
        assert np.all(t5_relative_bucket(r[1:], False, nb, md) == 0)
    with pytest.raises(ValueError):
        t5_relative_bucket(r, True, num_buckets=2)


@pytest.mark.parametrize("case", ["encoder", "decoder"])
def test_relative_bias_matches_hf(case):
    # WHY: the bias a head adds, [heads, q, k], equals HF's compute_bias with
    #      the same weights: bidirectional for the encoder, causal with
    #      past_seen_tokens (our q_offset) for a decoding step.
    # KIND: golden
    # CATCHES: s01, s02, s09, m02, m03, m06
    # CHAPTER: L6.4 section 2.2, Relative position buckets
    f = np.load(FIX)
    q, k, past = f[f"{case}.shape"].tolist()
    m = T5RelativeBias(
        3, num_buckets=8, max_distance=20, bidirectional=case == "encoder", rng=Rng(0)
    )
    assert [n for n, _ in m.named_parameters()] == ["relative_attention_bias.weight"]
    m.relative_attention_bias.weight.data[...] = f[f"{case}.weight"]
    got = m(q, k, q_offset=past)
    assert got.shape == (3, q, k)
    assert_close(got.data, f[f"{case}.bias"], dtype="float32")


def test_relative_bias_gradient_lands_on_buckets():
    # WHY: training the table: d sum(bias * g) / d weight[b, h] is the sum
    #      of g[h, i, j] over the (i, j) whose offset falls in bucket b, so
    #      every head learns its own table and far offsets share one entry.
    # KIND: property
    # CATCHES: s01, s02, m03
    # CHAPTER: L6.4 section 2.2, Relative position buckets
    m = T5RelativeBias(2, num_buckets=8, max_distance=16, rng=Rng(1))
    g = PCG32(seed=4).uniform_array((2, 6, 9), -1, 1)
    F.sum(m(6, 9) * Tensor(g.astype(np.float32))).backward()
    b = t5_relative_bucket(np.arange(9)[None, :] - np.arange(6)[:, None], True, 8, 16)
    want = np.zeros((8, 2))
    for h in range(2):
        np.add.at(want[:, h], b.ravel(), g[h].ravel())
    assert_close(m.relative_attention_bias.weight.grad, want, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        m(0, 3)
