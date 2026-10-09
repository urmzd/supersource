"""Course tests for L5.4: positional encodings, sinusoidal and learned
(tinyllm/xfmr/pos.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L5.4), and the chapter section it comes
from.

The worked example of the chapter (section 3), d = 4, base = 10000:
omega = (1, 0.01). PE(0) = (0, 1, 0, 1); PE(1) = (sin 1, cos 1, sin 0.01,
cos 0.01) = (0.841471, 0.540302, 0.010000, 0.999950); PE(2) = (0.909297,
-0.416147, 0.019999, 0.999800). Turning each pair of PE(1) by -1 * omega_i
gives PE(2).

The golden fixture (course/fixtures/L5.4/pe_mpmath.json) holds PE at 50
digits from mpmath 1.3.0 (course/oracle/L5.4/pe_mpmath.py), positions up to
65535.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.xfmr.pos import (
    LearnedPE,
    SinusoidalPE,
    shift_pe,
    sinusoidal_freqs,
    sinusoidal_pe,
)

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L5.4", "pe_mpmath.json")


class Rng:
    """The frozen PCG32 behind the generator API of M06.3."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.g.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        return self.g.below(n)


HAND = [
    [0.0, 1.0, 0.0, 1.0],
    [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)],
    [math.sin(2), math.cos(2), math.sin(0.02), math.cos(0.02)],
]


# --- the worked example -------------------------------------------------------------------


def test_hand_example_table():
    # WHY: the chapter's worked example, number for number: d = 4 gives the
    #      frequencies 1 and 0.01, sin in the even columns, cos in the odd
    #      ones, and PE(0) = (0, 1, 0, 1).
    # KIND: unit
    # CATCHES: s01, s02, s03, s05, m01, m02, m03, m05
    # CHAPTER: L5.4 section 3, Worked example by hand
    assert_close(sinusoidal_freqs(4), [1.0, 0.01], rtol=1e-12, atol=0.0)
    pe = sinusoidal_pe(3, 4)
    assert pe.dtype == np.float32 and pe.shape == (3, 4)
    assert_close(pe, HAND, dtype="float32")


def test_hand_example_shift():
    # WHY: moving one position turns each pair by its own frequency: the
    #      rotation of PE(1) by -omega_i is PE(2), the second row of the
    #      worked example, with no knowledge of p.
    # KIND: unit
    # CATCHES: s02, s04, s06, m01, m02, m04
    # CHAPTER: L5.4 section 3, Worked example by hand
    got = shift_pe(np.array(HAND[1]), 1.0, 4)
    assert_close(got, HAND[2], rtol=1e-12, atol=1e-12)


# --- against the 50-digit oracle ------------------------------------------------------------


@pytest.mark.parametrize("case", [0, 1])
def test_golden_mpmath(case):
    # WHY: the whole table against mpmath at 50 digits, for the paper's
    #      d = 16, base = 10000 and for d = 8, base = 500, at positions up to
    #      65535. A float32 angle is off by about p * 6e-8 rad, so at p = 10^4
    #      it misses by more than float32's tolerance: angles are float64.
    # KIND: golden
    # CATCHES: s01, s02, s03, s05, s07, m01, m02, m03, m05
    # CHAPTER: L5.4 section 2.1, The sinusoidal table
    c = json.load(open(FIX))["cases"][case]
    d, base = c["d"], c["base"]
    assert_close(sinusoidal_freqs(d, base), c["freqs"], rtol=1e-13, atol=0.0)
    full = sinusoidal_pe(max(c["positions"]) + 1, d, base)
    assert_close(full[c["positions"]], c["pe"], dtype="float32")


# --- the laws ----------------------------------------------------------------------------------


def test_shift_is_a_fixed_rotation():
    # WHY: PE(p + k) = R(k) PE(p) with R(k) independent of p: shifting the
    #      rows of four different positions by the same k gives exactly the
    #      table k rows later. This is the property that lets attention read
    #      relative positions from absolute ones, and that RoPE (L7.3) builds
    #      into the attention scores.
    # KIND: property
    # CATCHES: s01, s03, s04, s06, m03, m04, m05
    # CHAPTER: L5.4 section 2.2, A shift is a rotation
    d = 12
    table = sinusoidal_pe(200, d).astype(np.float64)
    g = PCG32(seed=3)
    for _ in range(5):
        k = g.below(60)
        ps = [g.below(140) for _ in range(4)]
        got = shift_pe(table[ps], k, d)
        assert_close(got, table[[p + k for p in ps]], rtol=1e-6, atol=1e-6)
    # A negative shift goes back.
    assert_close(shift_pe(table[50], -7, d), table[43], rtol=1e-6, atol=1e-6)


def test_dot_product_depends_only_on_the_offset():
    # WHY: PE(p) . PE(p + k) = sum_i cos(k omega_i) for every p: the
    #      similarity of two positions is a function of their distance only.
    # KIND: property
    # CATCHES: m03, m05
    # CHAPTER: L5.4 section 2.2, A shift is a rotation
    d = 16
    t = sinusoidal_pe(300, d).astype(np.float64)
    w = sinusoidal_freqs(d)
    for k in (1, 5, 40):
        want = np.sum(np.cos(k * w))
        dots = [t[p] @ t[p + k] for p in (0, 17, 123, 250)]
        assert_close(dots, [want] * 4, rtol=1e-5, atol=1e-5)


def test_every_pair_is_on_the_unit_circle():
    # WHY: each pair is (sin a, cos a), so its norm is 1 at every position
    #      and the whole row has norm sqrt(d / 2): the signal never grows
    #      with p, unlike adding p itself.
    # KIND: property
    # CATCHES: s01, m03, m05
    # CHAPTER: L5.4 section 2.1, The sinusoidal table
    t = sinusoidal_pe(1000, 10).astype(np.float64)
    norms = t[:, 0::2] ** 2 + t[:, 1::2] ** 2
    assert_close(norms, np.ones_like(norms), rtol=1e-6, atol=1e-6)


def test_offset_continues_the_table():
    # WHY: a table that starts at position `offset` equals the tail of one
    #      that starts at 0; incremental decoding (L8.2) asks for row t only.
    # KIND: property
    # CATCHES: s08, s09, m06, m09
    # CHAPTER: L5.4 section 4, The interface
    assert np.array_equal(sinusoidal_pe(5, 8, offset=11), sinusoidal_pe(16, 8)[11:])
    pe = SinusoidalPE(32, 8)
    x = Tensor(np.zeros((2, 3, 8)))
    assert_close(
        pe(x, offset=4).data,
        np.broadcast_to(sinusoidal_pe(7, 8)[4:], (2, 3, 8)),
        rtol=0.0,
        atol=0.0,
    )


# --- the modules ----------------------------------------------------------------------------------


def test_sinusoidal_module_has_no_parameters():
    # WHY: the sinusoidal table is a function, not a weight: nothing to
    #      train, nothing in the checkpoint, and the gradient passes through
    #      the addition unchanged to the embeddings below.
    # KIND: unit
    # CATCHES: s09, m09
    # CHAPTER: L5.4 section 4, The interface
    pe = SinusoidalPE(16, 6)
    assert pe.state_dict() == {} and list(pe.parameters()) == []
    x = Tensor(PCG32(seed=1).normal_array((2, 5, 6)), requires_grad=True)
    y = pe(x)
    assert_close(y.data, x.data + sinusoidal_pe(5, 6), dtype="float32")
    g = PCG32(seed=2).normal_array((2, 5, 6))
    F.sum(y * Tensor(g)).backward()
    assert_close(x.grad, g, dtype="float32")


def test_learned_pe_adds_rows_and_gets_their_gradient():
    # WHY: a learned table adds row offset + t at position t, and only the
    #      rows that were used receive a gradient (summed over the batch);
    #      GPT-2 (L6.1) and BERT (L6.2) train it like any embedding.
    # KIND: unit
    # CATCHES: s10, s11, m07
    # CHAPTER: L5.4 section 2.3, Learned positions
    pe = LearnedPE(10, 4, rng=Rng(5))
    x = Tensor(np.zeros((3, 4, 4)), requires_grad=True)
    y = pe(x, offset=2)
    assert_close(
        y.data, np.broadcast_to(pe.weight.data[2:6], (3, 4, 4)), rtol=0.0, atol=0.0
    )
    F.sum(y * Tensor(np.ones((3, 4, 4)))).backward()
    want = np.zeros((10, 4))
    want[2:6] = 3.0
    assert_close(pe.weight.grad, want, rtol=0.0, atol=0.0)
    assert_close(
        pe.positions(2, offset=8).data, pe.weight.data[8:10], rtol=0.0, atol=0.0
    )


def test_learned_pe_init():
    # WHY: the table starts at N(0, std^2) drawn from the given PCG32, so a
    #      seed fixes it in every language; the default std is GPT-2's 0.02.
    # KIND: statistical
    # CATCHES: s12, m08
    # CHAPTER: L5.4 section 2.3, Learned positions
    a, b = LearnedPE(64, 32, rng=Rng(7)), LearnedPE(64, 32, rng=Rng(7))
    assert np.array_equal(a.weight.data, b.weight.data) and a.weight.dtype == np.float32
    assert [n for n, _ in a.named_parameters()] == ["weight"] and a.weight.shape == (
        64,
        32,
    )
    w = a.weight.data.astype(np.float64)
    # 2048 draws: the sample std is within 6% of 0.02 with overwhelming probability.
    assert abs(w.std() - 0.02) < 0.0012 and abs(w.mean()) < 0.0015
    w2 = LearnedPE(64, 32, std=0.5, rng=Rng(7)).weight.data
    assert_close(w2, a.weight.data * 25.0, dtype="float32")


def test_validation():
    # WHY: an odd width has no last pair, a base <= 1 makes no ladder, and a
    #      position past the table is a bug the model must not hide by
    #      wrapping around.
    # KIND: boundary
    # CATCHES: s10, s13
    # CHAPTER: L5.4 section 4, The interface
    with pytest.raises(ValueError):
        sinusoidal_freqs(5)
    with pytest.raises(ValueError):
        sinusoidal_freqs(8, base=1.0)
    with pytest.raises(ValueError):
        sinusoidal_pe(-1, 8)
    with pytest.raises(ValueError):
        SinusoidalPE(4, 8)(Tensor(np.zeros((1, 5, 8))))
    with pytest.raises(ValueError):
        SinusoidalPE(8, 8)(Tensor(np.zeros((1, 3, 6))))
    with pytest.raises(ValueError):
        LearnedPE(6, 4)(Tensor(np.zeros((1, 3, 4))), offset=4)
    with pytest.raises(ValueError):
        shift_pe(np.zeros(6), 1, 8)
