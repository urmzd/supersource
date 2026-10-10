"""Course tests for L7.1: RMSNorm and the Pre-LN residual step
(tinyllm/modern/norm.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.1), and the chapter section it comes
from.

The worked example of the chapter (section 3): x = (3, 4, 0, 0), eps = 0.
mean(x^2) = 25 / 4 = 6.25, rms = 2.5, x / rms = (1.2, 1.6, 0, 0); with the
gain (1, 0.5, 2, 2) the output is (1.2, 0.8, 0, 0).

The golden fixture (course/fixtures/L7.1/rmsnorm_hf.npz) holds transformers
5.19.0 LlamaRMSNorm and GemmaRMSNorm float32 outputs and gradients, from
course/oracle/L7.1/rmsnorm_hf.py.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.norm import RMSNorm, pre_norm_residual
from tinyllm.nn.layers import Linear

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.1", "rmsnorm_hf.npz")


def golden(name: str, eps: float, offset: float):
    f = np.load(FIX)
    n = RMSNorm(16, eps=eps, offset=offset)
    n.load_state_dict({"weight": f[f"{name}.weight"]})
    x = Tensor(f[f"{name}.x"], requires_grad=True)
    y = n(x)
    F.sum(y * Tensor(f[f"{name}.g"])).backward()
    return f, n, x, y


# --- the worked example ------------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: (3, 4, 0, 0) has root mean square
    #      2.5, so it normalizes to (1.2, 1.6, 0, 0), and the gain
    #      (1, 0.5, 2, 2) scales each feature: (1.2, 0.8, 0, 0). No mean is
    #      subtracted (LayerNorm would give a different vector).
    # KIND: unit
    # CATCHES: s02, s05, s07
    # CHAPTER: L7.1 section 3, Worked example by hand
    n = RMSNorm(4, eps=0.0)
    assert_close(
        n(Tensor([[3.0, 4.0, 0.0, 0.0]])).data, [[1.2, 1.6, 0.0, 0.0]], dtype="float32"
    )
    n.load_state_dict({"weight": [1.0, 0.5, 2.0, 2.0]})
    assert_close(
        n(Tensor([[3.0, 4.0, 0.0, 0.0]])).data, [[1.2, 0.8, 0.0, 0.0]], dtype="float32"
    )


# --- against Hugging Face ------------------------------------------------------------------


def test_golden_hf_llama():
    # WHY: Llama's RMSNorm with copied weights: the same output and the same
    #      gradients for x and the gain, so SmolLM2's 61 norms (L7.9) compute
    #      what Hugging Face computes.
    # KIND: golden
    # CATCHES: s02, s05, s07, s09
    # CHAPTER: L7.1 section 4, The interface
    f, n, x, y = golden("llama", 1e-5, 0.0)
    assert_close(y.data, f["llama.y"], rtol=1e-5, atol=1e-6)
    assert_close(x.grad, f["llama.grad.x"], rtol=1e-4, atol=1e-6, msg="x")
    assert_close(
        n.weight.grad, f["llama.grad.weight"], rtol=1e-4, atol=1e-6, msg="weight"
    )


def test_golden_hf_gemma_offset():
    # WHY: Gemma stores gain - 1 (zero at init) and multiplies by 1 + weight:
    #      offset = 1 must reproduce GemmaRMSNorm, output and gradients.
    # KIND: golden
    # CATCHES: s03, s09
    # CHAPTER: L7.1 section 2.2, The gain and Gemma's offset
    f, n, x, y = golden("gemma", 1e-6, 1.0)
    assert_close(y.data, f["gemma.y"], rtol=1e-5, atol=1e-6)
    assert_close(x.grad, f["gemma.grad.x"], rtol=1e-4, atol=1e-6, msg="x")
    assert_close(
        n.weight.grad, f["gemma.grad.weight"], rtol=1e-4, atol=1e-6, msg="weight"
    )


def test_eps_inside_the_root():
    # WHY: on activations of size 1e-3, mean(x^2) ~ 1e-6 is smaller than
    #      eps = 1e-5, so where eps goes decides the output: Llama computes
    #      x / sqrt(mean(x^2) + eps), not x / (sqrt(mean(x^2)) + eps).
    # KIND: golden
    # CATCHES: s01
    # CHAPTER: L7.1 section 5, Pitfalls, item 1
    f, n, x, y = golden("llama-small", 1e-5, 0.0)
    assert_close(y.data, f["llama-small.y"], rtol=1e-5, atol=1e-7)
    assert_close(x.grad, f["llama-small.grad.x"], rtol=1e-4, atol=1e-6, msg="x")


def test_gradcheck_x_and_gain():
    # WHY: backward reaches the input (every earlier layer learns through
    #      the norm) and the gain, checked against the frozen central
    #      differences in float64, for both offsets.
    # KIND: gradcheck
    # CATCHES: s09
    # CHAPTER: L7.1 section 2.3, Backward
    g = PCG32(seed=3)
    x0, gy = g.normal_array((3, 5)), g.normal_array((3, 5))
    for offset in (0.0, 1.0):
        n = RMSNorm(5, eps=1e-3, offset=offset)
        w0 = g.normal_array((5,))
        n.weight.data = w0.copy()

        def f(x, w):
            n.weight.data = w
            return float(np.sum(n(Tensor(x, dtype=np.float64)).data * gy))

        n.weight.data = w0.astype(np.float64)
        x = Tensor(x0, requires_grad=True, dtype=np.float64)
        F.sum(n(x) * gy).backward()
        gradcheck(
            f,
            [x0, w0],
            [x.grad, n.weight.grad],
            names=["x", f"weight (offset {offset})"],
        )


# --- properties ------------------------------------------------------------------------------


def test_unit_rms_and_scale_invariance():
    # WHY: with unit gain and eps = 0 every output vector has root mean
    #      square exactly 1, so RMSNorm(c x) = RMSNorm(x) for any c > 0: the
    #      next layer sees the direction of x, not its size.
    # KIND: property
    # CATCHES: s05, s07
    # CHAPTER: L7.1 section 2.1, Normalizing by the root mean square
    x = PCG32(seed=4).normal_array((2, 3, 8))
    n = RMSNorm(8, eps=0.0)
    y = n(Tensor(x, dtype=np.float64)).data
    assert_close(
        np.sqrt(np.mean(y**2, axis=-1)), np.ones((2, 3)), rtol=1e-12, atol=1e-12
    )
    for c in (1e-3, 7.0, 1e4):
        assert_close(n(Tensor(c * x, dtype=np.float64)).data, y, rtol=1e-10, atol=1e-12)


def test_no_mean_is_subtracted():
    # WHY: RMSNorm only rescales: a constant vector stays constant (all
    #      ones for unit gain), where LayerNorm would map it to zeros.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L7.1 section 2.1, Normalizing by the root mean square
    y = RMSNorm(6, eps=0.0)(Tensor(np.full((2, 6), 3.0))).data
    assert_close(y, np.ones((2, 6)), dtype="float32")


def test_parameter_names_and_init():
    # WHY: the gain is the only parameter, named `weight` (the safetensors
    #      key of every Llama norm, L7.9), float32, ones for Llama and zeros
    #      for Gemma's offset form, so a fresh norm is the identity scaling.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L7.1 section 4, The interface
    for offset, init in ((0.0, 1.0), (1.0, 0.0)):
        n = RMSNorm(5, offset=offset)
        names = [k for k, _ in n.named_parameters()]
        assert names == ["weight"]
        assert n.weight.data.dtype == np.float32 and n.weight.data.shape == (5,)
        assert np.all(n.weight.data == init)
        assert n.eps == 1e-6 and n.offset == offset
    x = PCG32(seed=5).normal_array((4, 5))
    assert_close(
        RMSNorm(5, eps=0.0, offset=1.0)(Tensor(x)).data,
        RMSNorm(5, eps=0.0)(Tensor(x)).data,
        dtype="float32",
    )


def test_pre_norm_residual_identity_path():
    # WHY: Pre-LN keeps the residual stream un-normalized: when the sublayer
    #      outputs zeros the step is exactly the identity, in value and in
    #      gradient (d out / d x = I). That clean path from the loss to the
    #      embedding is why deep Pre-LN stacks train without warmup tricks.
    # KIND: property
    # CATCHES: s04, s08
    # CHAPTER: L7.1 section 2.4, Pre-LN
    x0 = 50.0 * PCG32(seed=6).normal_array((2, 3, 4))
    zero = Linear(4, 4, bias=False)
    zero.weight.data = np.zeros((4, 4), dtype=np.float32)
    x = Tensor(x0, requires_grad=True)
    out = pre_norm_residual(x, RMSNorm(4), zero)
    assert np.array_equal(out.data, x.data)
    F.sum(out).backward()
    assert np.array_equal(x.grad, np.ones_like(x0, dtype=np.float32))


def test_pre_norm_residual_adds_the_branch():
    # WHY: with a real sublayer the step adds sublayer(norm(x)) to x, and
    #      the branch sees the normalized input (a large x does not blow up
    #      the branch), not norm(x + sublayer(x)) (post-LN) or
    #      norm(x) + sublayer(norm(x)).
    # KIND: unit
    # CATCHES: s04, s08
    # CHAPTER: L7.1 section 2.4, Pre-LN
    g = PCG32(seed=7)
    x0 = 100.0 * g.normal_array((3, 4))
    lin = Linear(4, 4, bias=False)
    lin.weight.data = g.normal_array((4, 4)).astype(np.float32)
    norm = RMSNorm(4)
    out = pre_norm_residual(Tensor(x0), norm, lin)
    branch = lin(norm(Tensor(x0))).data
    assert_close(out.data, x0.astype(np.float32) + branch, dtype="float32")
    assert np.abs(branch).max() < 20.0


@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 5)])
def test_normalizes_each_vector_on_its_own(shape):
    # WHY: the statistic is over the last axis only: each token is
    #      normalized by its own RMS, so a batch of tokens gives the same
    #      result as one token at a time (no leakage across positions).
    # KIND: property
    # CATCHES: s05, s07
    # CHAPTER: L7.1 section 2.1, Normalizing by the root mean square
    x = PCG32(seed=8).normal_array(shape) * np.arange(1, shape[-1] + 1)
    n = RMSNorm(5, eps=1e-6)
    n.weight.data = np.linspace(0.5, 1.5, 5).astype(np.float32)
    y = n(Tensor(x)).data
    flat_x, flat_y = x.reshape(-1, 5), y.reshape(-1, 5)
    for row_x, row_y in zip(flat_x, flat_y):
        assert_close(n(Tensor(row_x[None])).data[0], row_y, dtype="float32")
    want = (
        x
        / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        * np.linspace(0.5, 1.5, 5)
    )
    assert_close(y, want, rtol=1e-5, atol=1e-6)


def test_validation():
    # WHY: a norm built for width 8 applied to width 6 is a wiring bug
    #      (a wrong hidden size in a config); fail loudly, as do a zero
    #      width and a negative eps.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: L7.1 section 4, The interface
    with pytest.raises(ValueError):
        RMSNorm(8)(Tensor(np.ones((2, 6))))
    with pytest.raises(ValueError):
        RMSNorm(0)
    with pytest.raises(ValueError):
        RMSNorm(4, eps=-1e-6)
