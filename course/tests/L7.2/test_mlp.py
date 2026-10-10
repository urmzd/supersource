"""Course tests for L7.2: gated MLPs, SwiGLU and GeGLU (tinyllm/modern/mlp.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.2), and the chapter section it comes
from.

The worked example of the chapter (section 3): d = d_ff = 1, gate weight 1,
up weight 2, down weight 3, x = 1. SwiGLU: silu(1) = 0.731059, times
up(x) = 2 gives 1.462117, times 3 gives 4.386352. GeGLU: gelu_tanh(1) =
0.841192, so 5.047152.

The golden fixture (course/fixtures/L7.2/gated_mlp_hf.npz) holds
transformers 5.19.0 LlamaMLP and GemmaMLP float32 outputs and gradients,
from course/oracle/L7.2/gated_mlp_hf.py.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.mlp import GatedMLP, llama_ffn_dim
from tinyllm.nn.layers import Linear

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.2", "gated_mlp_hf.npz")


class Rng:
    """The frozen PCG32 behind the generator API of M06.3, so no verdict here
    depends on your PCG32."""

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


def hand(act: str) -> GatedMLP:
    m = GatedMLP(1, 1, act=act, rng=Rng(0))
    m.load_state_dict(
        {
            "gate_proj.weight": [[1.0]],
            "up_proj.weight": [[2.0]],
            "down_proj.weight": [[3.0]],
        }
    )
    return m


def golden(name: str, act: str, bias: bool):
    f = np.load(FIX)
    m = GatedMLP(8, 12, act=act, bias=bias, rng=Rng(1))
    p = f"{name}.param."
    m.load_state_dict({k[len(p) :]: f[k] for k in f.files if k.startswith(p)})
    x = Tensor(f[f"{name}.x"], requires_grad=True)
    y = m(x)
    F.sum(y * Tensor(f[f"{name}.g"])).backward()
    assert_close(y.data, f[f"{name}.y"], rtol=1e-5, atol=1e-6)
    assert_close(x.grad, f[f"{name}.grad.x"], rtol=1e-4, atol=1e-6, msg="x")
    for n, prm in m.named_parameters():
        assert_close(prm.grad, f[f"{name}.grad.{n}"], rtol=1e-4, atol=1e-6, msg=n)


# --- the worked example ------------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: the activation of the GATE
    #      projection multiplies the UP projection, then down projects:
    #      3 * silu(1) * 2 = 4.386352, and with GeGLU 3 * gelu_tanh(1) * 2 =
    #      5.047152.
    # KIND: unit
    # CATCHES: s01, s02, s04
    # CHAPTER: L7.2 section 3, Worked example by hand
    silu1 = 1.0 / (1.0 + math.exp(-1.0))
    gelu1 = 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (1.0 + 0.044715)))
    assert_close(hand("silu")(Tensor([[1.0]])).data, [[6.0 * silu1]], dtype="float32")
    assert_close(
        hand("gelu_tanh")(Tensor([[1.0]])).data, [[6.0 * gelu1]], dtype="float32"
    )


# --- against Hugging Face ------------------------------------------------------------------


def test_golden_hf_swiglu():
    # WHY: Llama's MLP with copied weights: the same output and the same
    #      gradients for x and gate, up, down, so SmolLM2's 30 MLPs (L7.9)
    #      compute what Hugging Face computes.
    # KIND: golden
    # CATCHES: s01, s04, s08
    # CHAPTER: L7.2 section 4, The interface
    golden("swiglu", "silu", False)


def test_golden_hf_swiglu_bias():
    # WHY: `mlp_bias = true` checkpoints give all three projections a bias;
    #      each must be applied and trained.
    # KIND: golden
    # CATCHES: s06
    # CHAPTER: L7.2 section 4, The interface
    golden("swiglu-bias", "silu", True)


def test_golden_hf_geglu():
    # WHY: Gemma's MLP uses the tanh approximation of GELU
    #      (gelu_pytorch_tanh); the exact erf form differs by up to 5e-4,
    #      outside float32 tolerance.
    # KIND: golden
    # CATCHES: s02
    # CHAPTER: L7.2 section 5, Pitfalls, item 2
    golden("geglu", "gelu_tanh", False)


def test_gradcheck_every_parameter():
    # WHY: backward reaches x and all six parameters (weights and biases of
    #      gate, up, down) for both activations, checked against the frozen
    #      central differences in float64.
    # KIND: gradcheck
    # CATCHES: s08
    # CHAPTER: L7.2 section 2.3, Backward
    g = PCG32(seed=2)
    x0, gy = g.normal_array((2, 3)), g.normal_array((2, 3))
    for act in ("silu", "gelu_tanh"):
        m = GatedMLP(3, 4, act=act, bias=True, rng=Rng(3))
        params = [p for _, p in m.named_parameters()]
        for p in params:
            p.data = g.normal_array(p.data.shape)
        start = [p.data.copy() for p in params]

        def f(x, *ps):
            for p, a in zip(params, ps):
                p.data = a
            return float(np.sum(m(Tensor(x, dtype=np.float64)).data * gy))

        x = Tensor(x0, requires_grad=True, dtype=np.float64)
        F.sum(m(x) * gy).backward()
        gradcheck(
            f,
            [x0] + start,
            [x.grad] + [p.grad for p in params],
            names=["x"] + [n for n, _ in m.named_parameters()],
        )


# --- structure -------------------------------------------------------------------------------


def test_parameter_names_order_and_shapes():
    # WHY: the names and order are the safetensors keys of a Llama MLP
    #      (`model.layers.N.mlp.gate_proj.weight` ...): gate, up, down,
    #      [d_ff, d], [d_ff, d], [d, d_ff], float32; biases only when asked.
    # KIND: unit
    # CATCHES: s05, m03
    # CHAPTER: L7.2 section 4, The interface
    m = GatedMLP(6, 10, rng=Rng(4))
    assert [(n, p.data.shape) for n, p in m.named_parameters()] == [
        ("gate_proj.weight", (10, 6)),
        ("up_proj.weight", (10, 6)),
        ("down_proj.weight", (6, 10)),
    ]
    assert all(p.data.dtype == np.float32 for p in m.parameters())
    mb = GatedMLP(6, 10, bias=True, rng=Rng(4))
    assert [n for n, _ in mb.named_parameters()] == [
        "gate_proj.weight",
        "gate_proj.bias",
        "up_proj.weight",
        "up_proj.bias",
        "down_proj.weight",
        "down_proj.bias",
    ]
    assert m.act == "silu" and GatedMLP(2, 3, act="gelu_tanh").act == "gelu_tanh"


def test_rng_draw_order():
    # WHY: one seed fixes every weight in every language only if the draws
    #      happen in a fixed order: gate, then up, then down, from the one
    #      rng, exactly as three Linear layers built in that order.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L7.2 section 4, The interface
    m = GatedMLP(4, 5, rng=Rng(5))
    r = Rng(5)
    want = [
        Linear(4, 5, bias=False, rng=r),
        Linear(4, 5, bias=False, rng=r),
        Linear(5, 4, bias=False, rng=r),
    ]
    for (n, p), lin in zip(m.named_parameters(), want):
        assert np.array_equal(p.data, lin.weight.data), n
    assert np.array_equal(
        GatedMLP(4, 5).gate_proj.weight.data, GatedMLP(4, 5).gate_proj.weight.data
    )


def test_closed_gate_blocks_the_unit():
    # WHY: the activation sits on the GATE: when a unit's gate projection
    #      is strongly negative, silu and gelu are about 0 and the unit
    #      contributes nothing however large its up projection is. The gate
    #      decides, per token, which units pass.
    # KIND: property
    # CATCHES: s01
    # CHAPTER: L7.2 section 2.2, Gating
    g = PCG32(seed=6)
    for act in ("silu", "gelu_tanh"):
        m = GatedMLP(4, 3, act=act, rng=Rng(6))
        m.gate_proj.weight.data[1] = -100.0
        x = (np.abs(g.normal_array((5, 4))) + 0.1).astype(np.float32)
        y1 = m(Tensor(x)).data
        m.up_proj.weight.data[1] = 1e3
        assert_close(m(Tensor(x)).data, y1, rtol=1e-5, atol=1e-6)


# --- sizes -------------------------------------------------------------------------------------


def test_parameter_count_matches_a_4d_mlp():
    # WHY: three d x d_ff matrices at d_ff = 8d/3 hold 8 d^2 numbers, the
    #      same as the 2017 MLP's two d x 4d: the gated MLP is compared with
    #      the plain one at equal parameters, and Meta's width rule with no
    #      rounding is exactly that 8d/3.
    # KIND: property
    # CATCHES: m01
    # CHAPTER: L7.2 section 2.4, Sizing d_ff
    for d in (3, 6, 24):
        m = GatedMLP(d, 8 * d // 3, rng=Rng(7))
        assert sum(p.data.size for p in m.parameters()) == 8 * d * d == 2 * d * 4 * d
        assert llama_ffn_dim(d, multiple_of=1) == 8 * d // 3


def test_llama_ffn_dim():
    # WHY: Meta's rule, 2/3 of 4d, times an optional multiplier, rounded UP
    #      to a multiple: Llama-2 7B 11008, Llama-3 8B 14336, Llama-3.2 1B
    #      8192, SmolLM2-135M 1536. Rounding down gives 10752 for Llama-2 and
    #      the checkpoint no longer loads.
    # KIND: unit
    # CATCHES: s03, s07, m01
    # CHAPTER: L7.2 section 2.4, Sizing d_ff
    assert llama_ffn_dim(4096) == 11008
    assert llama_ffn_dim(4096, 1024, 1.3) == 14336
    assert llama_ffn_dim(2048, 256, 1.5) == 8192
    assert llama_ffn_dim(576) == 1536
    assert llama_ffn_dim(10, 1) == 26
    with pytest.raises(ValueError):
        llama_ffn_dim(0)
    with pytest.raises(ValueError):
        llama_ffn_dim(64, 0)
    with pytest.raises(ValueError):
        llama_ffn_dim(64, 8, 0.0)


def test_validation():
    # WHY: an unknown activation name in a config (`gelu_new`, `relu`) must
    #      fail at construction, not silently pick one; sizes below 1 too.
    # KIND: boundary
    # CATCHES: m02
    # CHAPTER: L7.2 section 4, The interface
    with pytest.raises(ValueError):
        GatedMLP(4, 8, act="relu")
    with pytest.raises(ValueError):
        GatedMLP(0, 8)
    with pytest.raises(ValueError):
        GatedMLP(4, 0)
