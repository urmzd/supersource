"""Course tests for L5.3: multi-head attention (tinyllm/xfmr/mha.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L5.3), and the chapter section it comes
from. Attention itself is L5.1's scaled_dot_product_attention.

The worked example of the chapter (section 3): d_model = 4, two heads of
width 2, every projection the identity with zero bias. The query
q = (1, 0, 0, 2) reads two keys (= values) k_1 = (1, 0, 0, 0) and
k_2 = (0, 1, 0, 2). Head 0 sees q = (1, 0) and keys (1, 0), (0, 1): scores
(1, 0) / sqrt 2, weights (0.669762, 0.330238), context (0.669762, 0.330238).
Head 1 sees q = (0, 2) and keys (0, 0), (0, 2): scores (0, 4) / sqrt 2,
weights (0.055817, 0.944183), context (0, 1.888366). The output is the two
contexts side by side: (0.669762, 0.330238, 0, 1.888366).

The golden fixture (course/fixtures/L5.3/mha_torch.npz) holds torch 2.14.1
nn.MultiheadAttention outputs, per-head weights, and gradients for a
self-attention case (causal and padding) and a cross-attention case
(padding), from course/oracle/L5.3/mha_torch.py.
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
from tinyllm.xfmr.mha import MultiHeadAttention

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L5.3", "mha_torch.npz")


def differs(a, b, tol: float = 1e-4) -> bool:
    """True when two arrays differ somewhere by more than tol: the opposite of
    closeness, so it needs no tolerance table."""
    return bool(
        np.max(
            np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        )
        > tol
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


def identity_mha(d=4, h=2) -> MultiHeadAttention:
    m = MultiHeadAttention(d, h, rng=Rng(0))
    eye = np.eye(d)
    m.load_state_dict(
        {n: (eye if n.endswith("weight") else np.zeros(d)) for n in m.state_dict()}
    )
    return m


def rand(seed, *shape):
    return PCG32(seed=seed).normal_array(shape).astype(np.float32)


HAND_Q = [[[1.0, 0.0, 0.0, 2.0]]]
HAND_KV = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 2.0]]]


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


# --- the worked example ----------------------------------------------------------------------


def test_hand_example_two_heads():
    # WHY: the chapter's worked example, number for number: each head
    #      compares only its own two features, scaled by 1 / sqrt(d_head) =
    #      1 / sqrt 2, and the output puts head 0's context in features 0-1
    #      and head 1's in features 2-3.
    # KIND: unit
    # CATCHES: s02, s03, s04, m03, m06
    # CHAPTER: L5.3 section 3, Worked example by hand
    m = identity_mha()
    out, w = m.attend(Tensor(HAND_Q), Tensor(HAND_KV))
    w0, w1 = sig(1 / math.sqrt(2)), sig(-4 / math.sqrt(2))
    assert w.shape == (1, 2, 1, 2)
    assert_close(w.data[0, :, 0], [[w0, 1 - w0], [w1, 1 - w1]], dtype="float32")
    assert_close(out.data, [[[w0, 1 - w0, 0.0, 2 * (1 - w1)]]], dtype="float32")


def test_hand_example_one_head_mixes_everything():
    # WHY: the same numbers with ONE head of width 4: a single score per key
    #      (q.k_1 = 1, q.k_2 = 4, scaled by 1 / 2) and one weight vector for
    #      all features. Two heads let one position attend to two places at
    #      once; one head cannot.
    # KIND: unit
    # CATCHES: s04, m01, m03
    # CHAPTER: L5.3 section 3, Worked example by hand
    m = identity_mha(4, 1)
    out, w = m.attend(Tensor(HAND_Q), Tensor(HAND_KV))
    a = sig(0.5 - 2.0)
    assert_close(w.data[0, 0, 0], [a, 1 - a], dtype="float32")
    assert_close(out.data[0, 0], [a, 1 - a, 0.0, 2 * (1 - a)], dtype="float32")


# --- against torch ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["self", "cross"])
def test_golden_torch(case):
    # WHY: torch's nn.MultiheadAttention with the same weights (its packed
    #      in_proj split into q, k, v) must give the same output, the same
    #      per-head weights, and the same gradients of sum(out * g) for the
    #      inputs and every parameter: self-attention with a causal and
    #      padding mask, cross-attention with padding.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s07, s08, m01, m02, m03, m04, m05, m06
    # CHAPTER: L5.3 section 2.2, Heads
    f = np.load(FIX)
    m = MultiHeadAttention(8, 2, rng=Rng(1))
    m.load_packed_in_proj(f["in_proj_weight"], f["in_proj_bias"])
    m.load_state_dict(
        {"out_proj.weight": f["out_proj.weight"], "out_proj.bias": f["out_proj.bias"]},
        strict=False,
    )
    xq = Tensor(f[f"{case}.x_q"], requires_grad=True)
    xkv = xq if case == "self" else Tensor(f[f"{case}.x_kv"], requires_grad=True)
    out, w = m.attend(xq, xkv, f[f"{case}.mask"])
    assert_close(out.data, f[f"{case}.out"], rtol=1e-5, atol=1e-6)
    assert_close(w.data, f[f"{case}.weights"], rtol=1e-5, atol=1e-6)
    F.sum(out * Tensor(f[f"{case}.g"])).backward()
    assert_close(xq.grad, f[f"{case}.grad.x_q"], rtol=1e-4, atol=1e-5, msg="x_q")
    if case == "cross":
        assert_close(xkv.grad, f[f"{case}.grad.x_kv"], rtol=1e-4, atol=1e-5, msg="x_kv")
    for name, p in m.named_parameters():
        assert_close(p.grad, f[f"{case}.grad.{name}"], rtol=1e-4, atol=1e-5, msg=name)


def test_gradcheck_inputs_and_parameters():
    # WHY: backward reaches both inputs and all eight parameter tensors
    #      through the split, the attention, and the merge; checked against
    #      the frozen central differences in float64 with a padding mask.
    # KIND: gradcheck
    # CATCHES: s04, s08, m02, m05
    # CHAPTER: L5.3 section 2.2, Heads
    m = MultiHeadAttention(4, 2, rng=Rng(2))
    params = [p for _, p in m.named_parameters()]
    for p in params:
        p.data = p.data.astype(np.float64)
    g = PCG32(seed=3)
    xq0, xkv0, gout = (
        g.normal_array((2, 3, 4)),
        g.normal_array((2, 5, 4)),
        g.normal_array((2, 3, 4)),
    )
    mask = (np.arange(5)[None, :] < np.array([5, 2])[:, None])[:, None, None, :]

    def f(xq, xkv, *ps):
        for p, a in zip(params, ps):
            p.data = a
        return float(
            np.sum(
                m(
                    Tensor(xq, dtype=np.float64), Tensor(xkv, dtype=np.float64), mask
                ).data
                * gout
            )
        )

    start = [p.data.copy() for p in params]
    xq = Tensor(xq0, requires_grad=True, dtype=np.float64)
    xkv = Tensor(xkv0, requires_grad=True, dtype=np.float64)
    F.sum(m(xq, xkv, mask) * gout).backward()
    analytic = [xq.grad, xkv.grad] + [p.grad for p in params]
    gradcheck(
        f,
        [xq0, xkv0] + start,
        analytic,
        names=["x_q", "x_kv"] + [n for n, _ in m.named_parameters()],
    )


# --- heads --------------------------------------------------------------------------------------------


def test_split_and_merge_layout():
    # WHY: head h owns features h * d_head .. (h + 1) * d_head - 1 of every
    #      position, and merge is the exact inverse of split. Reshaping
    #      [B, H, T, d_head] straight to [B, T, d_model] would mix positions.
    # KIND: property
    # CATCHES: s01, s02, m01, m06
    # CHAPTER: L5.3 section 2.2, Heads
    m = MultiHeadAttention(6, 3, rng=Rng(4))
    x = rand(5, 2, 4, 6)
    s = m.split_heads(Tensor(x))
    assert s.shape == (2, 3, 4, 2)
    for h in range(3):
        assert np.array_equal(s.data[:, h], x[:, :, 2 * h : 2 * h + 2])
    assert np.array_equal(m.merge_heads(s).data, x)


def test_heads_are_independent():
    # WHY: changing the query projection rows that belong to head 1 changes
    #      head 1's weights and nothing in head 0: each head is its own
    #      attention in its own subspace.
    # KIND: property
    # CATCHES: s02, s04
    # CHAPTER: L5.3 section 2.2, Heads
    m = MultiHeadAttention(4, 2, rng=Rng(6))
    xq, xkv = Tensor(rand(7, 1, 3, 4)), Tensor(rand(8, 1, 5, 4))
    _, w_before = m.attend(xq, xkv)
    m.q_proj.weight.data[2:] *= 3.0
    _, w_after = m.attend(xq, xkv)
    assert np.array_equal(w_after.data[:, 0], w_before.data[:, 0])
    assert differs(w_after.data[:, 1], w_before.data[:, 1])


def test_one_head_is_projected_attention():
    # WHY: with one head the module is softmax(Q K^T / sqrt(d) + mask) V
    #      followed by out_proj, computed here in float64 numpy from the
    #      module's own weights.
    # KIND: differential
    # CATCHES: s04, s08, m01, m02, m03, m04, m06, m07
    # CHAPTER: L5.3 section 2.1, One head
    m = MultiHeadAttention(6, 1, rng=Rng(9))
    sd = {n: v.astype(np.float64) for n, v in m.state_dict().items()}
    xq, xkv = rand(10, 2, 3, 6), rand(11, 2, 4, 6)

    def lin(x, n):
        return x @ sd[n + ".weight"].T + sd[n + ".bias"]

    q, k, v = lin(xq, "q_proj"), lin(xkv, "k_proj"), lin(xkv, "v_proj")
    e = q @ np.swapaxes(k, -1, -2) / math.sqrt(6)
    a = np.exp(e - e.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    want = lin(a @ v, "out_proj")
    assert_close(m(Tensor(xq), Tensor(xkv)).data, want, rtol=1e-5, atol=1e-5)


# --- masks --------------------------------------------------------------------------------------------


def test_mask_shapes_agree():
    # WHY: a [Tq, Tk] mask, a per-sequence [B, Tq, Tk] mask, and the full
    #      [B, H, Tq, Tk] mask saying the same thing give the same output.
    #      With B = H = 2 a [B, Tq, Tk] mask broadcast without a head axis
    #      would silently give sequence b's mask to head b.
    # KIND: property
    # CATCHES: s05, m05
    # CHAPTER: L5.3 section 2.3, Masks for every head
    m = MultiHeadAttention(4, 2, rng=Rng(12))
    xq, xkv = Tensor(rand(13, 2, 3, 4)), Tensor(rand(14, 2, 3, 4))
    causal = np.tril(np.ones((3, 3), dtype=bool))
    per_seq = np.stack([causal, np.ones((3, 3), dtype=bool)])
    full = np.broadcast_to(per_seq[:, None], (2, 2, 3, 3))
    a = m(xq, xkv, per_seq).data
    assert np.array_equal(a, m(xq, xkv, full).data)
    b = m(xq, xkv, causal).data
    assert np.array_equal(b[0], a[0]) and differs(b[1], a[1])


def test_masked_keys_are_never_read():
    # WHY: a masked key gets weight exactly 0 in every head, so whatever its
    #      value holds cannot reach the output: changing the padding changes
    #      nothing.
    # KIND: property
    # CATCHES: s02, s04, m05
    # CHAPTER: L5.3 section 2.3, Masks for every head
    m = MultiHeadAttention(4, 2, rng=Rng(15))
    xq, xkv = rand(16, 2, 3, 4), rand(17, 2, 5, 4)
    keep = np.arange(5)[None, :] < np.array([5, 2])[:, None]
    mask = keep[:, None, None, :]
    out, w = m.attend(Tensor(xq), Tensor(xkv), mask)
    assert np.all(w.data[1, :, :, 2:] == 0.0)
    xkv2 = xkv.copy()
    xkv2[1, 2:] = 100.0
    assert np.array_equal(m(Tensor(xq), Tensor(xkv2), mask).data, out.data)


def test_cross_attention_reads_keys_from_x_kv():
    # WHY: in cross-attention the queries come from x_q and BOTH keys and
    #      values from x_kv: the output has x_q's length, and permuting the
    #      positions of x_kv (a set, without positions) changes nothing, while
    #      permuting x_q permutes the output rows.
    # KIND: property
    # CATCHES: s01, s02, s04, m06
    # CHAPTER: L5.3 section 2.4, Self- and cross-attention
    m = MultiHeadAttention(4, 2, rng=Rng(18))
    xq, xkv = rand(19, 1, 3, 4), rand(20, 1, 5, 4)
    out = m(Tensor(xq), Tensor(xkv)).data
    assert out.shape == (1, 3, 4)
    perm = [3, 0, 4, 1, 2]
    assert_close(m(Tensor(xq), Tensor(xkv[:, perm])).data, out, rtol=1e-5, atol=1e-6)
    qperm = [2, 0, 1]
    assert_close(
        m(Tensor(xq[:, qperm]), Tensor(xkv)).data, out[:, qperm], rtol=1e-5, atol=1e-6
    )


# --- the interface ------------------------------------------------------------------------------------------


def test_dropout_only_in_training():
    # WHY: attention dropout is training noise: in eval mode the module
    #      equals the same weights without dropout and draws nothing; in
    #      training mode it changes the weights.
    # KIND: unit
    # CATCHES: s04, s06
    # CHAPTER: L5.3 section 4, The interface
    a = MultiHeadAttention(4, 2, dropout=0.5, rng=Rng(21))
    b = MultiHeadAttention(4, 2, dropout=0.0, rng=Rng(21))
    xq, xkv = Tensor(rand(22, 2, 3, 4)), Tensor(rand(23, 2, 6, 4))
    a.eval()
    assert np.array_equal(a(xq, xkv).data, b(xq, xkv).data)
    a.train()
    assert differs(a(xq, xkv).data, b(xq, xkv).data)


def test_parameter_names_shapes_and_init():
    # WHY: the names are the checkpoint keys every model in Parts 5 to 7
    #      builds on (q_proj, k_proj, v_proj, out_proj, in that order); a
    #      seed fixes the weights; bias=False registers no biases.
    # KIND: unit
    # CATCHES: m04, m07
    # CHAPTER: L5.3 section 4, The interface
    m = MultiHeadAttention(6, 3, rng=Rng(24))
    assert [(n, v.shape) for n, v in m.state_dict().items()] == [
        (f"{p}.{w}", (6, 6) if w == "weight" else (6,))
        for p in ("q_proj", "k_proj", "v_proj", "out_proj")
        for w in ("weight", "bias")
    ]
    assert m.d_head == 2 and m.n_heads == 3
    m2 = MultiHeadAttention(6, 3, rng=Rng(24))
    assert all(
        np.array_equal(a, b)
        for a, b in zip(m.state_dict().values(), m2.state_dict().values())
    )
    nb = MultiHeadAttention(6, 3, bias=False, rng=Rng(24))
    assert list(nb.state_dict()) == [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    ]


def test_load_packed_in_proj():
    # WHY: torch stacks the query, key, and value projections in ONE
    #      [3 d, d] weight, rows q, then k, then v; loading it in another
    #      order swaps what each projection does.
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L5.3 section 4, The interface
    m = MultiHeadAttention(2, 1, rng=Rng(25))
    w = np.arange(12, dtype=np.float32).reshape(6, 2)
    b = np.arange(6, dtype=np.float32) + 100
    m.load_packed_in_proj(w, b)
    assert np.array_equal(m.q_proj.weight.data, w[0:2]) and np.array_equal(
        m.k_proj.weight.data, w[2:4]
    )
    assert np.array_equal(m.v_proj.weight.data, w[4:6]) and np.array_equal(
        m.v_proj.bias.data, b[4:6]
    )
    with pytest.raises(ValueError):
        m.load_packed_in_proj(w[:4])


def test_validation():
    # WHY: heads must tile d_model exactly; inputs must be [B, T, d_model]
    #      with one batch size; a mask is bool and broadcasts to
    #      [B, H, Tq, Tk]. Each is a wiring bug in the model that calls it.
    # KIND: boundary
    # CATCHES: m08
    # CHAPTER: L5.3 section 4, The interface
    with pytest.raises(ValueError):
        MultiHeadAttention(6, 4)
    with pytest.raises(ValueError):
        MultiHeadAttention(4, 2, dropout=1.0)
    m = MultiHeadAttention(4, 2, rng=Rng(26))
    xq, xkv = Tensor(rand(27, 2, 3, 4)), Tensor(rand(28, 2, 5, 4))
    with pytest.raises(ValueError):
        m(xq, Tensor(rand(29, 3, 5, 4)))
    with pytest.raises(ValueError):
        m(Tensor(rand(30, 2, 3, 6)), xkv)
    with pytest.raises(ValueError):
        m(xq, xkv, np.ones((3, 5)))  # not bool
    with pytest.raises(ValueError):
        m(xq, xkv, np.ones((3, 4), dtype=bool))  # Tk is 5
    with pytest.raises(ValueError):
        m.merge_heads(Tensor(rand(31, 2, 3, 4)))
