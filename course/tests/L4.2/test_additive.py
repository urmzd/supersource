"""Course tests for L4.2: Bahdanau additive attention (tinyllm/seq2seq/additive.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L4.2), and the chapter section it comes
from.

The worked example of the chapter (section 3): every size 1, W = U = v = 1,
b = 0, query q = 0, keys k = (0, 1, 2) with the third position padding.
Scores e = (tanh 0, tanh 1, masked) = (0, 0.761594, -inf); weights
softmax = (0.318300, 0.681700, 0); context = 0.681700.

The golden fixture (course/fixtures/L4.2/additive_torch.npz) holds torch
2.14.1 float32 outputs and gradients for the same formula with copied
weights, from course/oracle/L4.2/additive_torch.py.
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
from tinyllm.seq2seq.additive import AdditiveAttention, length_mask

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L4.2", "additive_torch.npz")


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


def hand() -> AdditiveAttention:
    att = AdditiveAttention(1, 1, 1, rng=Rng(0))
    att.load_state_dict(
        {
            "query.weight": [[1.0]],
            "key.weight": [[1.0]],
            "key.bias": [0.0],
            "v.weight": [[1.0]],
        }
    )
    return att


def rand_inputs(seed: int, B=3, S=5, Dq=4, Dk=6, dtype=np.float32):
    g = PCG32(seed=seed)
    q = g.normal_array((B, Dq)).astype(dtype)
    k = g.normal_array((B, S, Dk)).astype(dtype)
    return q, k


# --- the worked example ---------------------------------------------------------------


def test_hand_example_weights_and_context():
    # WHY: the chapter's worked example, number for number: scores tanh(0)
    #      and tanh(1), the padded third position weight exactly 0 even though
    #      its score tanh(2) would be the largest, weights 0.318300 and
    #      0.681700, context 0.681700.
    # KIND: unit
    # CATCHES: s01, s03, s05, s06, s08, m01, m03
    # CHAPTER: L4.2 section 3, Worked example by hand
    att = hand()
    q = Tensor([[0.0]])
    k = Tensor([[[0.0], [1.0], [2.0]]])
    ctx, a = att(q, k, [[True, True, False]])
    w1 = math.exp(math.tanh(1.0)) / (1.0 + math.exp(math.tanh(1.0)))
    assert a.shape == (1, 3) and ctx.shape == (1, 1)
    assert a.data[0, 2] == 0.0
    assert_close(a.data, [[1.0 - w1, w1, 0.0]], dtype="float32")
    assert_close(ctx.data, [[w1]], dtype="float32")
    assert_close(
        att.scores(q, k).data, [[0.0, math.tanh(1.0), math.tanh(2.0)]], dtype="float32"
    )


def test_length_mask():
    # WHY: the mask is the one place lengths enter attention: position s of
    #      row b may be read exactly when s < lengths[b]. Lengths outside
    #      [0, S] are caller bugs.
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L4.2 section 2.3, Masking padding
    m = length_mask(np.array([3, 1, 0]), 4)
    assert m.dtype == bool
    assert m.tolist() == [
        [True, True, True, False],
        [True, False, False, False],
        [False] * 4,
    ]
    for bad in (np.array([5]), np.array([-1]), np.array([[1]]), np.array([1.5])):
        with pytest.raises(ValueError):
            length_mask(bad, 4)


# --- against torch ------------------------------------------------------------------------


def test_golden_torch():
    # WHY: the same formula in torch with the same weights must give the
    #      same context, weights, and gradients (of sum(context * gc) +
    #      sum(weights * gw)) for the query, the keys, and all four
    #      parameters, on a batch with lengths 5, 3, and 1.
    # KIND: golden
    # CATCHES: s01, s03, s04, s05, s06, s07, s08, m01, m02, m03, m06
    # CHAPTER: L4.2 section 4, The interface
    f = np.load(FIX)
    att = AdditiveAttention(4, 6, 7, rng=Rng(1))
    att.load_state_dict(
        {k[len("param.") :]: f[k] for k in f.files if k.startswith("param.")}
    )
    q = Tensor(f["query"], requires_grad=True)
    k = Tensor(f["keys"], requires_grad=True)
    ctx, a = att(q, k, length_mask(f["lengths"], f["keys"].shape[1]))
    assert_close(ctx.data, f["context"], rtol=1e-4, atol=1e-6)
    assert_close(a.data, f["weights"], rtol=1e-4, atol=1e-6)
    ((ctx * f["gc"]).numpy().sum() and None)
    loss = ctx * Tensor(f["gc"])
    total = None
    total = F.sum(ctx * Tensor(f["gc"])) + F.sum(a * Tensor(f["gw"]))
    total.backward()
    assert_close(q.grad, f["grad.query"], rtol=1e-4, atol=1e-6, msg="query")
    assert_close(k.grad, f["grad.keys"], rtol=1e-4, atol=1e-6, msg="keys")
    for name, p in att.named_parameters():
        assert_close(p.grad, f["grad." + name], rtol=1e-4, atol=1e-6, msg=name)


def test_gradcheck_every_input_and_parameter():
    # WHY: backward must reach the query (the decoder state, so the decoder
    #      learns where to look), the keys (the encoder outputs, so the
    #      encoder learns what to offer), and every parameter. Checked against
    #      the frozen central differences in float64, with one row padded.
    # KIND: gradcheck
    # CATCHES: s04, m06
    # CHAPTER: L4.2 section 2.2, Scores, weights, context
    att = AdditiveAttention(3, 2, 4, rng=Rng(2))
    params = [p for _, p in att.named_parameters()]
    for p in params:
        p.data = p.data.astype(np.float64)
    q0, k0 = rand_inputs(3, B=2, S=4, Dq=3, Dk=2, dtype=np.float64)
    mask = length_mask(np.array([4, 2]), 4)
    g = PCG32(seed=4)
    gc, gw = g.normal_array((2, 2)), g.normal_array((2, 4))

    def f(q, k, *ps):
        for p, a in zip(params, ps):
            p.data = a
        ctx, w = att(Tensor(q, dtype=np.float64), Tensor(k, dtype=np.float64), mask)
        return float(np.sum(ctx.data * gc) + np.sum(w.data * gw))

    start = [p.data.copy() for p in params]
    q, k = (
        Tensor(q0, requires_grad=True, dtype=np.float64),
        Tensor(k0, requires_grad=True, dtype=np.float64),
    )
    ctx, w = att(q, k, mask)
    (F.sum(ctx * gc) + F.sum(w * gw)).backward()
    analytic = [q.grad, k.grad] + [p.grad for p in params]
    gradcheck(
        f,
        [q0, k0] + start,
        analytic,
        names=["query", "keys"] + [n for n, _ in att.named_parameters()],
    )


# --- masking ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_masked_weights_are_exactly_zero_and_rows_sum_to_one(seed):
    # WHY: a padded position must get weight exactly 0 (not 1e-9: the next
    #      layer would read padding), and the real positions still sum to 1,
    #      because the mask is applied to the SCORES before the softmax.
    # KIND: property
    # CATCHES: s01, s03, s05, s07, m06
    # CHAPTER: L4.2 section 5, Pitfalls, item 1
    att = AdditiveAttention(4, 6, 5, rng=Rng(seed))
    q, k = rand_inputs(10 + seed)
    lens = np.array([5, 2, 4])
    _, a = att(Tensor(q), Tensor(k), length_mask(lens, 5))
    w = a.data.astype(np.float64)
    for b, n in enumerate(lens):
        assert np.all(w[b, n:] == 0.0)
        assert np.all(w[b, :n] > 0.0)
    assert_close(w.sum(axis=1), np.ones(3), rtol=1e-6, atol=1e-6)


def test_padding_content_never_matters():
    # WHY: whatever the encoder wrote at a padded position (garbage, huge
    #      values) must not change the context or the weights, bit for bit,
    #      and gets no gradient: a batch padded differently gives each
    #      sentence the same translation.
    # KIND: property
    # CATCHES: s01, s05, s07, m06
    # CHAPTER: L4.2 section 5, Pitfalls, item 1
    att = AdditiveAttention(4, 6, 5, rng=Rng(5))
    q, k = rand_inputs(20)
    lens = np.array([3, 5, 1])
    mask = length_mask(lens, 5)
    k2 = k.copy()
    k2[0, 3:] = 1e4
    k2[2, 1:] = -7.0
    c1, a1 = att(Tensor(q), Tensor(k), mask)
    kt = Tensor(k2, requires_grad=True)
    c2, a2 = att(Tensor(q), kt, mask)
    assert np.array_equal(c1.data, c2.data) and np.array_equal(a1.data, a2.data)
    F.sum(c2).backward()
    assert np.all(kt.grad[~mask] == 0.0)


def test_fully_masked_row_reads_nothing():
    # WHY: an empty source (length 0) has nothing to read: its weights and
    #      context are 0, not NaN (a softmax of an all -inf row, M09.2), and
    #      no gradient comes back from it. Filling with a large finite
    #      negative number instead of -inf gives uniform weights here.
    # KIND: boundary
    # CATCHES: s01, s02, s05, s07, m06
    # CHAPTER: L4.2 section 5, Pitfalls, item 2
    att = AdditiveAttention(4, 6, 5, rng=Rng(6))
    q, k = rand_inputs(30)
    qt, kt = Tensor(q, requires_grad=True), Tensor(k, requires_grad=True)
    ctx, a = att(qt, kt, length_mask(np.array([0, 5, 2]), 5))
    assert np.all(a.data[0] == 0.0) and np.all(ctx.data[0] == 0.0)
    assert np.isfinite(a.data).all() and np.isfinite(ctx.data).all()
    F.sum(ctx).backward()
    assert np.all(qt.grad[0] == 0.0) and np.all(kt.grad[0] == 0.0)


# --- structure ------------------------------------------------------------------------------------


def test_precomputed_keys_equal_recomputed():
    # WHY: U k_s + b does not depend on the decoder step, so a decoder
    #      computes project_keys once per sentence and passes it at every
    #      step: the result must be identical to recomputing it.
    # KIND: property
    # CATCHES: m06
    # CHAPTER: L4.2 section 2.4, Precomputing the keys
    att = AdditiveAttention(4, 6, 5, rng=Rng(7))
    q, k = rand_inputs(40)
    mask = length_mask(np.array([5, 4, 3]), 5)
    kt = Tensor(k)
    proj = att.project_keys(kt)
    assert proj.shape == (3, 5, 5)
    c1, a1 = att(Tensor(q), kt, mask)
    c2, a2 = att(Tensor(q), kt, mask, proj)
    assert np.array_equal(c1.data, c2.data) and np.array_equal(a1.data, a2.data)


def test_permuting_the_source_permutes_the_weights():
    # WHY: attention has no notion of order by itself: shuffling the source
    #      positions (and the mask with them) shuffles the weights the same
    #      way and leaves the context unchanged. Order must come from the
    #      encoder (the RNN here, position encodings in L5).
    # KIND: property
    # CATCHES: m06
    # CHAPTER: L4.2 section 2.2, Scores, weights, context
    att = AdditiveAttention(4, 6, 5, rng=Rng(8))
    q, k = rand_inputs(50)
    mask = length_mask(np.array([5, 3, 4]), 5)
    perm = np.array([3, 0, 4, 1, 2])
    c1, a1 = att(Tensor(q), Tensor(k), mask)
    c2, a2 = att(Tensor(q), Tensor(k[:, perm]), mask[:, perm])
    assert_close(a2.data, a1.data[:, perm], rtol=1e-5, atol=1e-6)
    assert_close(c2.data, c1.data, rtol=1e-5, atol=1e-6)


def test_parameter_names_and_shapes():
    # WHY: the parameter names are the safetensors keys of every seq2seq
    #      checkpoint with this attention (L4.1, the zoo). The same rng seed
    #      gives the same weights.
    # KIND: unit
    # CATCHES: m01, m03, m04
    # CHAPTER: L4.2 section 4, The interface
    att = AdditiveAttention(4, 6, 7, rng=Rng(9))
    sd = att.state_dict()
    assert list(sd) == ["query.weight", "key.weight", "key.bias", "v.weight"]
    assert [v.shape for v in sd.values()] == [(7, 4), (7, 6), (7,), (1, 7)]
    assert all(v.dtype == np.float32 for v in sd.values())
    assert att.query_from == "previous"
    again = AdditiveAttention(4, 6, 7, rng=Rng(9)).state_dict()
    assert all(np.array_equal(sd[n], again[n]) for n in sd)


def test_validation():
    # WHY: a query of the wrong width, keys without a source axis, or a mask
    #      of the wrong shape or type (a float array is not a mask) is a wiring
    #      bug in the decoder; each fails loudly instead of broadcasting into
    #      nonsense.
    # KIND: boundary
    # CATCHES: m05
    # CHAPTER: L4.2 section 4, The interface
    att = AdditiveAttention(4, 6, 5, rng=Rng(11))
    q, k = rand_inputs(60)
    mask = length_mask(np.array([5, 5, 5]), 5)
    with pytest.raises(ValueError):
        AdditiveAttention(0, 6, 5)
    with pytest.raises(ValueError):
        att(Tensor(q[:, :3]), Tensor(k), mask)
    with pytest.raises(ValueError):
        att(Tensor(q), Tensor(k[:, 0]), mask)
    with pytest.raises(ValueError):
        att(Tensor(q), Tensor(k), mask[:, :4])
    with pytest.raises(ValueError):
        att(Tensor(q), Tensor(k), np.ones((3, 5)))
    with pytest.raises(ValueError):
        att(Tensor(q), Tensor(k), mask, Tensor(np.zeros((3, 5, 4), dtype=np.float32)))
