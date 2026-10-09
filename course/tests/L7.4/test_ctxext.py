"""Course tests for L7.4: context extension for RoPE (tinyllm/modern/ctxext.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.4), and the chapter section it comes
from. The ALiBi tests (the optional part, DESIGN D31) are in test_alibi.py.

The worked example of the chapter (section 3), d_rot = 4 and 8, base 1e4:
theta = (1, 0.01) for d_rot 4; linear x2 gives (0.5, 0.005); NTK x2 gives
base' = 1e4 * 2^2 = 4e4 and (1, 0.005). YaRN with d_rot 8 (theta = 1, 0.1,
0.01, 0.001), factor 4, original context 2048: the correction range is
[1, 3], the ramp (0, 0, 0.5, 1), so inv_freq (1, 0.1, 0.00625, 0.00025) and
attention scaling 0.1 ln 4 + 1 = 1.138629. Llama-3 with the same theta,
factor 8, original 1024, low 1, high 4: wavelengths (6.28, 62.8, 628.3,
6283); pairs 0 and 1 keep theta, pair 3 is divided by 8, pair 2 blends with
s = (1024 / 628.3 - 1) / 3 = 0.209916: 0.01 (0.790084 / 8 + 0.209916) =
0.003087.

The golden fixture (course/fixtures/L7.4/rope_scaling_hf.json) holds
transformers 5.19.0 ROPE_INIT_FUNCTIONS outputs (float32) for every kind,
from course/oracle/L7.4/rope_scaling_hf.py.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from tinyllm.modern.ctxext import rope_inv_freq_scaled

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.4", "rope_scaling_hf.json")


def theta(r: int, base: float) -> np.ndarray:
    return base ** (-np.arange(0, r, 2, dtype=np.float64) / r)


def hf_cases() -> list[dict]:
    with open(FIX) as f:
        return json.load(f)["rope"]


# --- the worked example ------------------------------------------------------------------


def test_hand_example_linear_and_ntk():
    # WHY: the chapter's worked example: position interpolation divides
    #      every frequency by the factor; NTK-aware scaling raises the base
    #      so the fastest pair keeps its frequency and only the slowest is
    #      divided by exactly the factor.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L7.4 section 3, Worked example by hand
    inv, s = rope_inv_freq_scaled(4, 1e4, "linear", 2.0)
    assert_close(inv, [0.5, 0.005], rtol=1e-12, atol=0)
    assert s == 1.0
    inv, s = rope_inv_freq_scaled(4, 1e4, "ntk", 2.0)
    assert_close(inv, [1.0, 0.005], rtol=1e-12, atol=0)
    assert s == 1.0
    inv, s = rope_inv_freq_scaled(4, 1e4, "default", 7.0)
    assert_close(inv, [1.0, 0.01], rtol=1e-12, atol=0)
    assert s == 1.0


def test_hand_example_yarn():
    # WHY: YaRN by hand: pairs that turn more than beta_fast = 32 times over
    #      the original 2048 positions keep theta, pairs that turn fewer than
    #      beta_slow = 1 time are divided by the factor, pair 2 sits half way
    #      on the ramp; the attention temperature is 0.1 ln 4 + 1.
    # KIND: unit
    # CATCHES: s03, s04, s06
    # CHAPTER: L7.4 section 3, Worked example by hand
    inv, s = rope_inv_freq_scaled(8, 1e4, "yarn", 4.0, 2048)
    assert_close(inv, [1.0, 0.1, 0.00625, 0.00025], rtol=1e-12, atol=0)
    assert_close(s, 0.1 * math.log(4.0) + 1.0, rtol=1e-12, atol=0)


def test_hand_example_llama3():
    # WHY: Llama 3 by hand: short wavelengths keep theta, long ones are
    #      divided by the factor, and the band between blends linearly in
    #      L0 / wavelength.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: L7.4 section 3, Worked example by hand
    inv, s = rope_inv_freq_scaled(8, 1e4, "llama3", 8.0, 1024, low_freq_factor=1, high_freq_factor=4)
    sm = (1024 / (2 * math.pi / 0.01) - 1) / 3
    assert_close(inv, [1.0, 0.1, 0.01 * ((1 - sm) / 8 + sm), 0.001 / 8], rtol=1e-12, atol=0)
    assert s == 1.0


# --- against Hugging Face ---------------------------------------------------------------------


@pytest.mark.parametrize("i", range(10))
def test_golden_hf_rope_init_functions(i):
    # WHY: L7.9 builds RoPE from a checkpoint's `rope_scaling`; for every
    #      kind (linear, NTK through HF's dynamic rule, YaRN with default and
    #      custom betas and partial rotary, Llama 3.1 and 3.2) the
    #      frequencies and the attention scaling must equal Hugging Face's.
    #      HF computes in float32, the course in float64: rtol 2e-6.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, s07
    # CHAPTER: L7.4 section 4, The interface
    c = hf_cases()[i]
    inv, s = rope_inv_freq_scaled(**c["args"])
    assert inv.dtype == np.float64 and inv.shape == (c["args"]["d_rot"] // 2,)
    assert_close(inv, c["inv_freq"], rtol=2e-6, atol=0, msg=c["name"])
    assert_close(s, c["attention_scaling"], rtol=1e-6, atol=0, msg=c["name"])


# --- properties ---------------------------------------------------------------------------------


def test_linear_maps_long_positions_onto_trained_angles():
    # WHY: position interpolation means position p * factor of the extended
    #      model has exactly the angles position p had in training: nothing
    #      the model sees is new.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: L7.4 section 2.2, Position interpolation
    base = theta(64, 1e5)
    inv, _ = rope_inv_freq_scaled(64, 1e5, "linear", 4.0)
    for p in (1, 100, 2047):
        assert_close(4 * p * inv, p * base, rtol=1e-12, atol=0)


@pytest.mark.parametrize("r,factor", [(16, 2.0), (64, 4.0), (128, 8.0), (128, 32.0)])
def test_ntk_keeps_fastest_divides_slowest(r, factor):
    # WHY: the NTK-aware base change is chosen so pair 0 (the finest
    #      position detail) is untouched and the slowest pair is divided by
    #      exactly the factor; every pair in between is divided by less.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: L7.4 section 2.3, NTK-aware scaling
    base = theta(r, 1e4)
    inv, _ = rope_inv_freq_scaled(r, 1e4, "ntk", factor)
    assert_close(inv[0], 1.0, rtol=1e-12, atol=0)
    assert_close(inv[-1], base[-1] / factor, rtol=1e-10, atol=0)
    ratio = base / inv
    assert np.all(np.diff(ratio) > 0) and np.all(ratio >= 1 - 1e-12) and np.all(ratio <= factor * (1 + 1e-10))


def test_yarn_bands():
    # WHY: YaRN leaves the high-frequency pairs exactly as trained, divides
    #      the low-frequency ones by exactly the factor, and blends between;
    #      factor 1 is the default ladder with temperature 1.
    # KIND: property
    # CATCHES: s03, s04, s07
    # CHAPTER: L7.4 section 2.4, YaRN
    r, base, L0, f = 128, 1e6, 32768, 8.0
    th = theta(r, base)
    inv, s = rope_inv_freq_scaled(r, base, "yarn", f, L0)
    turns = L0 * th / (2 * math.pi)  # rotations of each pair over the original context
    fast, slow = turns > 32, turns < 1
    assert fast.sum() > 10 and slow.sum() > 10
    assert_close(inv[fast], th[fast], rtol=1e-12, atol=0)
    assert_close(inv[slow], th[slow] / f, rtol=1e-12, atol=0)
    mid = ~fast & ~slow
    assert np.all(inv[mid] <= th[mid]) and np.all(inv[mid] >= th[mid] / f)
    assert_close(s, 0.1 * math.log(8) + 1, rtol=1e-12, atol=0)
    inv1, s1 = rope_inv_freq_scaled(r, base, "yarn", 1.0, L0)
    assert_close(inv1, th, rtol=1e-12, atol=0)
    assert s1 == 1.0


def test_llama3_bands_keep_frequencies_ordered():
    # WHY: Llama 3's rule must not reorder frequencies: after scaling,
    #      inv_freq still decreases with the pair index, and it is the
    #      default ladder for short wavelengths and theta / factor for long.
    # KIND: property
    # CATCHES: s05
    # CHAPTER: L7.4 section 2.5, Llama 3 scaling
    th = theta(128, 5e5)
    inv, _ = rope_inv_freq_scaled(128, 5e5, "llama3", 8.0, 8192)
    w = 2 * math.pi / th
    assert np.all(np.diff(inv) < 0)
    assert_close(inv[w < 8192 / 4], th[w < 8192 / 4], rtol=1e-12, atol=0)
    assert_close(inv[w > 8192], th[w > 8192] / 8, rtol=1e-12, atol=0)


def test_validation():
    # WHY: a typo in `rope_scaling` (an unknown type, a factor below 1, a
    #      missing original context, swapped Llama 3 factors) must fail when
    #      the model is built, not produce a model that degrades silently.
    # KIND: boundary
    # CATCHES: m01, m02, m03
    # CHAPTER: L7.4 section 4, The interface
    bad = [
        dict(d_rot=64, base=1e4, kind="longrope", factor=2.0, original_max_pos=4096),
        dict(d_rot=63, base=1e4, kind="default"),
        dict(d_rot=64, base=1.0, kind="default"),
        dict(d_rot=64, base=1e4, kind="linear", factor=0.5),
        dict(d_rot=64, base=1e4, kind="yarn", factor=4.0),
        dict(d_rot=64, base=1e4, kind="llama3", factor=8.0, original_max_pos=8192, low_freq_factor=4, high_freq_factor=1),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            rope_inv_freq_scaled(**kw)
