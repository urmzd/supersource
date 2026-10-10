"""Course tests for L6.6: LoRA with PiSSA initialization and merge
(tinyllm/obj/lora.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L6.6), and the chapter section it comes
from.

The worked example of the chapter (section 3): W = [[1, 2], [3, 4]],
b = (0.5, -0.5), r = 1, alpha = 2 (so s = 2), A = [[1, -1]], B = [[0.5], [1]],
x = (2, 1). The frozen path gives W x + b = (4.5, 9.5); A x = 1, B A x =
(0.5, 1), times s = (1, 2); y = (5.5, 11.5). Merging gives
W' = W + s B A = [[2, 1], [5, 2]] and W' x + b = (5.5, 11.5) again. PiSSA on
W = diag(3, 1) with r = 1, alpha = 4 (s = 4): W_1 = diag(3, 0), so
B = (sqrt(3) / 2, 0)^T, A = (sqrt(3) / 2, 0), and the frozen residual is
diag(0, 1).

The oracle for PiSSA is numpy's LAPACK SVD (np.linalg.svd), independent of
your M03.5 Jacobi SVD. PEFT itself is not run here: its adapter key names
(adapter_model.safetensors) are spelled out in test_peft_key_names.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding, LayerNorm, Linear, ReLU, Sequential
from tinyllm.nn.module import Module
from tinyllm.obj.lora import (
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_lora,
    trainable_fraction,
)


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


def lin(W, b=None) -> Linear:
    W = np.asarray(W, dtype=np.float32)
    m = Linear(W.shape[1], W.shape[0], bias=b is not None, rng=Rng(0))
    m.weight.data[...] = W
    if b is not None:
        m.bias.data[...] = np.asarray(b, dtype=np.float32)
    return m


def x_of(shape, s=1) -> Tensor:
    return Tensor(PCG32(seed=s).uniform_array(shape, -1.0, 1.0).astype(np.float32))


class Block(Module):
    """A tiny model with the names LoRA usually targets."""

    def __init__(self, s: int = 0) -> None:
        super().__init__()
        r = Rng(s)
        self.emb = Embedding(11, 8, rng=r)
        self.q_proj = Linear(8, 8, rng=r)
        self.v_proj = Linear(8, 8, rng=r)
        self.mlp = Sequential(Linear(8, 16, rng=r), ReLU(), Linear(16, 8, rng=r))
        self.norm = LayerNorm(8)
        self.head = Linear(8, 5, rng=r)

    def forward(self, ids):
        h = self.emb(ids)
        h = h + self.v_proj(F.tanh(self.q_proj(h)))
        h = h + self.mlp(self.norm(h))
        return self.head(h)


IDS = np.array([[1, 4, 7, 2], [3, 3, 9, 10]])


def attn_and_mlp(name, m):
    return name.endswith(("q_proj", "v_proj")) or name.startswith("mlp.")


# --- the worked example --------------------------------------------------------------


def test_hand_example_forward_and_merge():
    # WHY: the chapter's worked example: the adapted output is the frozen
    #      path plus s * B A x with s = alpha / r, and merging W' = W + s B A
    #      gives the same output from a plain Linear.
    # KIND: unit
    # CATCHES: s06, s07, m03
    # CHAPTER: L6.6 section 3, Worked example by hand
    m = LoRALinear(lin([[1, 2], [3, 4]], [0.5, -0.5]), r=1, alpha=2.0, rng=Rng(1))
    assert m.scaling == 1.0 * 2.0 / 1
    m.lora_A.weight.data[...] = [[1.0, -1.0]]
    m.lora_B.weight.data[...] = [[0.5], [1.0]]
    x = Tensor(np.array([2.0, 1.0], dtype=np.float32))
    assert_close(m(x).data, [5.5, 11.5], dtype="float32")
    assert_close(m.delta_weight(), [[1.0, -1.0], [2.0, -2.0]], dtype="float32")
    model = Sequential(m)
    assert merge_lora(model) == ["0"]
    plain = model[0]
    assert type(plain) is Linear
    assert_close(plain.weight.data, [[2.0, 1.0], [5.0, 2.0]], dtype="float32")
    assert_close(model(x).data, [5.5, 11.5], dtype="float32")


def test_hand_example_pissa_split():
    # WHY: PiSSA on W = diag(3, 1), r = 1, alpha = 4: the adapter holds the
    #      top singular part, s B A = diag(3, 0), with B and A each carrying
    #      sqrt(3 / s); the frozen weight keeps the residual diag(0, 1).
    # KIND: unit
    # CATCHES: s04, s05, s06
    # CHAPTER: L6.6 section 3, Worked example by hand
    m = LoRALinear(lin([[3, 0], [0, 1]]), r=1, alpha=4.0, init="pissa")
    h = math.sqrt(3.0) / 2.0
    B, A = m.lora_B.weight.data, m.lora_A.weight.data
    sign = np.sign(A[0, 0])  # (B, A) and (-B, -A) are the same split
    assert_close(sign * B, [[h], [0.0]], rtol=1e-6, atol=1e-6)
    assert_close(sign * A, [[h, 0.0]], rtol=1e-6, atol=1e-6)
    assert_close(m.weight.data, [[0.0, 0.0], [0.0, 1.0]], rtol=1e-6, atol=1e-6)
    assert_close(m.delta_weight(), [[3.0, 0.0], [0.0, 0.0]], rtol=1e-6, atol=1e-6)


# --- default init and the frozen path --------------------------------------------------


def test_default_init_equals_base_bitwise():
    # WHY: with B = 0 the update s B A x is exactly zero, so the adapted
    #      model's output equals the base model's bit for bit: fine-tuning
    #      starts from the pretrained function, not near it.
    # KIND: differential
    # CATCHES: s02, m02
    # CHAPTER: L6.6 section 2.2, Initialization
    base = Block(3)
    x = IDS
    want = base(x).data.copy()
    names = inject_lora(base, attn_and_mlp, r=2, alpha=8.0, rng=Rng(5))
    assert names == ["q_proj", "v_proj", "mlp.0", "mlp.2"]
    got = base(x).data
    assert got.dtype == want.dtype
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    for n in names:
        mod = dict(base.named_modules())[n]
        assert not mod.lora_B.weight.data.any()
        assert mod.lora_A.weight.data.any()


def test_scaling_is_alpha_over_r():
    # WHY: s = alpha / r keeps the update's size roughly independent of r,
    #      so changing r does not force a new learning rate. Doubling r at
    #      fixed alpha halves s.
    # KIND: unit
    # CATCHES: s01, s06
    # CHAPTER: L6.6 section 2.1, The update
    for r, alpha in [(1, 1.0), (2, 8.0), (4, 8.0), (3, 1.5)]:
        m = LoRALinear(Linear(6, 5, rng=Rng(r)), r=r, alpha=alpha, rng=Rng(9))
        assert m.scaling == pytest.approx(alpha / r, rel=0, abs=0)
        m.lora_B.weight.data[...] = PCG32(seed=r).uniform_array((5, r), -1, 1)
        A, B = (
            m.lora_A.weight.data.astype(np.float64),
            m.lora_B.weight.data.astype(np.float64),
        )
        assert_close(m.delta_weight(), (alpha / r) * B @ A, dtype="float32")
        x = x_of((3, 6), r)
        base = x.data @ m.weight.data.T + m.bias.data
        assert_close(
            m(x).data, base + x.data @ m.delta_weight().T, rtol=1e-5, atol=1e-5
        )


def test_only_adapter_parameters_get_gradients():
    # WHY: the point of LoRA is that only A and B train. The frozen weight
    #      and bias stay registered (they are in the checkpoint) but get no
    #      gradient, so the optimizer state is r (in + out) numbers per layer.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: L6.6 section 2.3, Freezing
    m = LoRALinear(Linear(6, 4, rng=Rng(2)), r=2, alpha=2.0, rng=Rng(3))
    m.lora_B.weight.data[...] = 0.3
    names = [n for n, _ in m.named_parameters()]
    assert names == ["weight", "bias", "lora_A.weight", "lora_B.weight"]
    flags = {n: p.requires_grad for n, p in m.named_parameters()}
    assert flags == {
        "weight": False,
        "bias": False,
        "lora_A.weight": True,
        "lora_B.weight": True,
    }
    F.sum(m(x_of((5, 6), 4)) ** 2).backward()
    assert m.weight.grad is None and m.bias.grad is None
    assert m.lora_A.weight.grad is not None and np.abs(m.lora_A.weight.grad).sum() > 0
    assert m.lora_B.weight.grad is not None and np.abs(m.lora_B.weight.grad).sum() > 0


def test_inject_freezes_everything_else():
    # WHY: inject_lora freezes the WHOLE model, not just the wrapped layers:
    #      embeddings, norms, and untargeted Linears stay as pretrained.
    #      Only adapters train, and the trainable fraction is small.
    # KIND: property
    # CATCHES: s03, s09, m02
    # CHAPTER: L6.6 section 2.3, Freezing
    model = Block(1)
    names = inject_lora(
        model, lambda n, m: n in ("q_proj", "v_proj"), r=2, alpha=4.0, rng=Rng(2)
    )
    assert names == ["q_proj", "v_proj"]
    trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert trainable == [
        "q_proj.lora_A.weight",
        "q_proj.lora_B.weight",
        "v_proj.lora_A.weight",
        "v_proj.lora_B.weight",
    ]
    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            mod.lora_B.weight.data[...] = 0.1
    F.sum(model(IDS)).backward()
    for n, p in model.named_parameters():
        assert (p.grad is not None) == p.requires_grad, n
    with pytest.raises(ValueError):
        inject_lora(Block(1), lambda n, m: False, r=2, alpha=4.0)


def test_adapter_gradcheck():
    # WHY: the gradients A and B receive are the ones of the formula
    #      y = x W^T + b + s B A x: checked against central differences of
    #      sum(y * g) in float64 (the frozen gradcheck, never yours).
    # KIND: gradcheck
    # CATCHES: s01
    # CHAPTER: L6.6 section 2.1, The update
    g = PCG32(seed=11)
    W, b = g.uniform_array((4, 5), -1, 1), g.uniform_array((4,), -1, 1)
    A0, B0 = g.uniform_array((2, 5), -1, 1), g.uniform_array((4, 2), -1, 1)
    x, w = g.uniform_array((3, 5), -1, 1), g.uniform_array((3, 4), -1, 1)
    m = LoRALinear(lin(W, b), r=2, alpha=3.0, rng=Rng(0))
    m.weight.data, m.bias.data = W.copy(), b.copy()
    m.lora_A.weight.data, m.lora_B.weight.data = A0.copy(), B0.copy()
    F.sum(m(Tensor(x, dtype=np.float64)) * Tensor(w, dtype=np.float64)).backward()

    def f(A, B):
        return float(np.sum((x @ W.T + b + 1.5 * (x @ A.T) @ B.T) * w))

    gradcheck(
        f, [A0, B0], [m.lora_A.weight.grad, m.lora_B.weight.grad], names=["A", "B"]
    )


# --- PiSSA -----------------------------------------------------------------------------


def test_pissa_reproduces_the_base():
    # WHY: PiSSA moves the principal part of W into the adapter and leaves
    #      the residual frozen, so at step 0 the model computes the same
    #      function as the base (to float32 rounding), whatever r and alpha.
    # KIND: differential
    # CATCHES: s04, s05, s06
    # CHAPTER: L6.6 section 2.2, Initialization
    for r, alpha in [(1, 1.0), (3, 8.0), (4, 2.0)]:
        base = Linear(6, 5, rng=Rng(r + 20))
        x = x_of((4, 6), r)
        want = base(x).data.copy()
        W = base.weight.data.astype(np.float64).copy()
        m = LoRALinear(base, r=r, alpha=alpha, init="pissa")
        assert_close(m(x).data, want, rtol=1e-5, atol=1e-5)
        assert_close(
            m.weight.data.astype(np.float64) + m.delta_weight(), W, rtol=1e-5, atol=1e-6
        )


def test_pissa_adapter_is_the_top_singular_part():
    # WHY: s B A must be the best rank-r approximation of W (Eckart-Young),
    #      computed here with LAPACK's SVD, and B's columns must span the top
    #      left singular vectors: that is what makes PiSSA train the
    #      principal directions first.
    # KIND: golden
    # CATCHES: s01, s04, s05, s06
    # CHAPTER: L6.6 section 2.2, Initialization
    base = Linear(7, 6, rng=Rng(42))
    W = base.weight.data.astype(np.float64).copy()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    r, alpha = 3, 12.0
    m = LoRALinear(base, r=r, alpha=alpha, init="pissa")
    Wr = (U[:, :r] * S[:r]) @ Vt[:r]
    assert_close(m.delta_weight(), Wr, rtol=1e-5, atol=1e-5)
    assert_close(m.weight.data, W - Wr, rtol=1e-5, atol=1e-5)
    B = m.lora_B.weight.data.astype(np.float64)
    # Columns of B are sqrt(S_r / s) times +-U_r.
    norms = np.linalg.norm(B, axis=0)
    assert_close(norms, np.sqrt(S[:r] / (alpha / r)), rtol=1e-5, atol=1e-6)
    cos = np.abs(np.sum(B / norms * U[:, :r], axis=0))
    assert_close(cos, np.ones(r), rtol=1e-5, atol=1e-5)


# --- merge -----------------------------------------------------------------------------


def test_merge_equals_unmerged():
    # WHY: serving a fine-tuned model must not pay for the adapter: merging
    #      folds s B A into W, and the merged plain Linears give the same
    #      outputs as the adapted model (within 1e-5), for default and PiSSA
    #      adapters after a training-like change to A and B.
    # KIND: differential
    # CATCHES: s06, s07, m02, m03
    # CHAPTER: L6.6 section 2.4, Merging
    for init in ("default", "pissa"):
        model = Block(7)
        inject_lora(model, attn_and_mlp, r=2, alpha=6.0, init=init, rng=Rng(8))
        g = PCG32(seed=13)
        for _, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                for t in (mod.lora_A.weight, mod.lora_B.weight):
                    t.data[...] += g.uniform_array(t.shape, -0.3, 0.3).astype(
                        np.float32
                    )
        want = model(IDS).data.copy()
        assert merge_lora(model) == ["q_proj", "v_proj", "mlp.0", "mlp.2"]
        assert not any(isinstance(mod, LoRALinear) for _, mod in model.named_modules())
        assert_close(model(IDS).data, want, rtol=1e-5, atol=1e-5)
        assert merge_lora(model) == []


def test_merge_restores_the_base_keys():
    # WHY: a merged model is a plain checkpoint: the same state_dict keys,
    #      in the same order, as the model before inject_lora, so every
    #      loader (the zoo, the engine) reads it unchanged. With B = 0 the
    #      merged weights equal the base weights exactly.
    # KIND: property
    # CATCHES: s02, s07
    # CHAPTER: L6.6 section 2.4, Merging
    model = Block(2)
    before = model.state_dict()
    inject_lora(model, attn_and_mlp, r=1, alpha=1.0, rng=Rng(4))
    keys = [n for n, _ in model.named_parameters()]
    assert "q_proj.weight" in keys and "q_proj.lora_A.weight" in keys
    merge_lora(model)
    after = model.state_dict()
    assert list(after) == list(before)
    for k in before:
        assert np.array_equal(after[k], before[k]), k


# --- adapter files -------------------------------------------------------------------------


def test_peft_key_names():
    # WHY: adapters are shared as PEFT's adapter_model.safetensors, keyed
    #      "base_model.model.<module>.lora_A.weight" with A [r, in] and
    #      B [out, r]: the names a later multi-adapter server (sq.multi-lora)
    #      and the HF ecosystem expect.
    # KIND: golden
    # CATCHES: s08, m02
    # CHAPTER: L6.6 section 4, The interface
    model = Block(5)
    inject_lora(
        model, lambda n, m: n in ("q_proj", "mlp.2"), r=3, alpha=6.0, rng=Rng(6)
    )
    sd = lora_state_dict(model)
    assert list(sd) == [
        "base_model.model.q_proj.lora_A.weight",
        "base_model.model.q_proj.lora_B.weight",
        "base_model.model.mlp.2.lora_A.weight",
        "base_model.model.mlp.2.lora_B.weight",
    ]
    assert sd["base_model.model.q_proj.lora_A.weight"].shape == (3, 8)
    assert sd["base_model.model.mlp.2.lora_B.weight"].shape == (8, 3)


def test_adapter_roundtrip():
    # WHY: an adapter file restores exactly the adapter it came from into a
    #      fresh model with the same injection; wrong keys fail before
    #      anything is copied.
    # KIND: property
    # CATCHES: s08, m05
    # CHAPTER: L6.6 section 4, The interface
    a, b = Block(5), Block(5)
    inject_lora(a, attn_and_mlp, r=2, alpha=2.0, rng=Rng(1))
    inject_lora(b, attn_and_mlp, r=2, alpha=2.0, rng=Rng(2))
    for _, mod in a.named_modules():
        if isinstance(mod, LoRALinear):
            mod.lora_B.weight.data[...] = 0.25
    sd = lora_state_dict(a)
    load_lora_state_dict(b, sd)
    assert np.array_equal(a(IDS).data, b(IDS).data)
    bad = dict(sd)
    bad["base_model.model.k_proj.lora_A.weight"] = bad.pop(
        "base_model.model.q_proj.lora_A.weight"
    )
    before = lora_state_dict(b)
    with pytest.raises(KeyError):
        load_lora_state_dict(b, bad)
    after = lora_state_dict(b)
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_dropout_only_on_the_adapter_path():
    # WHY: LoRA dropout perturbs the adapter's input only; the frozen path
    #      must see x untouched, and in eval mode dropout is the identity.
    #      With B = 0 nothing the dropout does can change the output.
    # KIND: property
    # CATCHES: s02, s10
    # CHAPTER: L6.6 section 2.1, The update
    base = Linear(6, 4, rng=Rng(3))
    x = x_of((5, 6), 9)
    want = base(x).data.copy()
    m = LoRALinear(base, r=2, alpha=2.0, dropout=0.5, rng=Rng(4))
    assert np.array_equal(m(x).data, want)  # training mode, B = 0
    m.lora_B.weight.data[...] = 0.5
    m.eval()
    clean = m(x).data.copy()
    assert_close(clean, want + x.data @ m.delta_weight().T, rtol=1e-5, atol=1e-5)
    m.train()
    noisy = m(x).data
    assert not np.array_equal(noisy, clean)


def test_trainable_fraction():
    # WHY: the milestone reports trainable_frac < 0.05 for a LoRA fine-tune;
    #      the fraction counts elements, frozen parameters included in the
    #      denominator.
    # KIND: unit
    # CATCHES: s03, s09, m02, m04
    # CHAPTER: L6.6 section 2.3, Freezing
    model = Block(0)
    total = sum(p.data.size for p in model.parameters())
    assert trainable_fraction(model) == 1.0
    inject_lora(model, lambda n, m: n == "q_proj", r=1, alpha=1.0, rng=Rng(0))
    assert trainable_fraction(model) == pytest.approx(16 / (total + 16), rel=1e-12)
    with pytest.raises(ValueError):
        trainable_fraction(Sequential(ReLU()))


def test_validation():
    # WHY: a rank above min(in, out) adds nothing a full update could not,
    #      and PiSSA cannot split it; alpha <= 0, a non-Linear base, and an
    #      unknown init are caller bugs that must fail loudly.
    # KIND: boundary
    # CATCHES: m01
    # CHAPTER: L6.6 section 4, The interface
    base = Linear(4, 3, rng=Rng(0))
    LoRALinear(base, r=3, alpha=1.0)
    for kw in (
        dict(r=0, alpha=1.0),
        dict(r=4, alpha=1.0),
        dict(r=1, alpha=0.0),
        dict(r=1, alpha=1.0, init="svd"),
    ):
        with pytest.raises(ValueError):
            LoRALinear(Linear(4, 3, rng=Rng(0)), **kw)
    with pytest.raises(TypeError):
        LoRALinear(LayerNorm(4), r=1, alpha=1.0)
