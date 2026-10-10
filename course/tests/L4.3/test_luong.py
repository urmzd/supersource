"""Course tests for L4.3: Luong attention, dot, general, and concat scores
(tinyllm/seq2seq/luong.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L4.3), and the chapter section it comes
from. Masks come from L4.2's length_mask.

The worked example of the chapter (section 3), d = 2, dot score:
h = (1, 0), keys k_1 = (1, 0), k_2 = (0, 1), k_3 = (2, 0) with k_3 padding.
Scores (1, 0, masked); weights (e / (e + 1), 1 / (e + 1), 0) =
(0.731059, 0.268941, 0); context c = (0.731059, 0.268941). With
W_c = [I I], h~ = tanh(c + h) = (tanh 1.731059, tanh 0.268941) =
(0.939181, 0.262640).

The golden fixture (course/fixtures/L4.3/luong_torch.npz) holds torch 2.14.1
float32 outputs and gradients of the three scores with copied weights, from
course/oracle/L4.3/luong_torch.py.
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
from tinyllm.seq2seq.additive import length_mask
from tinyllm.seq2seq.luong import LuongAttention

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L4.3", "luong_torch.npz")
SCORES = ("dot", "general", "concat")


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


def rand_inputs(seed: int, B=3, S=4, d=5, dtype=np.float32):
    g = PCG32(seed=seed)
    return g.normal_array((B, d)).astype(dtype), g.normal_array((B, S, d)).astype(dtype)


HAND_Q = [[1.0, 0.0]]
HAND_K = [[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]]
HAND_MASK = [[True, True, False]]


# --- the worked example ---------------------------------------------------------------


def test_hand_example_dot():
    # WHY: the chapter's worked example, number for number: dot scores 1 and
    #      0, the padded third key (score 2) weight exactly 0, weights 0.731059
    #      and 0.268941, and h~ = tanh(c + h) with W_c = [I I].
    # KIND: unit
    # CATCHES: s04, s06, s07
    # CHAPTER: L4.3 section 3, Worked example by hand
    att = LuongAttention(2, "dot", rng=Rng(0))
    att.load_state_dict(
        {"combine.weight": [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]}
    )
    q, k = Tensor(HAND_Q), Tensor(HAND_K)
    ctx, a = att(q, k, HAND_MASK)
    w = math.e / (math.e + 1.0)
    assert a.data[0, 2] == 0.0
    assert_close(a.data, [[w, 1.0 - w, 0.0]], dtype="float32")
    assert_close(ctx.data, [[w, 1.0 - w]], dtype="float32")
    assert_close(att.scores(q, k).data, [[1.0, 0.0, 2.0]], dtype="float32")
    ht = att.attentional(q, ctx)
    assert_close(ht.data, [[math.tanh(1.0 + w), math.tanh(1.0 - w)]], dtype="float32")


def test_hand_example_attentional_order():
    # WHY: h~ = tanh(W_c [c ; h]): the context comes FIRST. With
    #      W_c = [I 0] the result is tanh(c) = (tanh 0.731059, tanh 0.268941),
    #      not tanh(h) = (tanh 1, 0); swapping the halves loads Luong-trained
    #      weights wrong.
    # KIND: unit
    # CATCHES: s01, s04, s06, s07
    # CHAPTER: L4.3 section 3, Worked example by hand
    att = LuongAttention(2, "dot", rng=Rng(0))
    att.load_state_dict(
        {"combine.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]}
    )
    q = Tensor(HAND_Q)
    ctx, _ = att(q, Tensor(HAND_K), HAND_MASK)
    w = math.e / (math.e + 1.0)
    assert_close(
        att.attentional(q, ctx).data,
        [[math.tanh(w), math.tanh(1.0 - w)]],
        dtype="float32",
    )


# --- against torch ----------------------------------------------------------------------------


@pytest.mark.parametrize("score", SCORES)
def test_golden_torch(score):
    # WHY: each score in torch with the same weights must give the same
    #      weights, context, attentional state h~, and gradients (of
    #      sum(h~ * gh) + sum(weights * gw)) for the query, the keys, and every
    #      parameter, on lengths 4, 2, 3.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s06, s07, m01, m02, m03, m04, m06
    # CHAPTER: L4.3 section 2.2, Three scores
    f = np.load(FIX)
    att = LuongAttention(5, score, rng=Rng(1))
    pre = f"{score}.param."
    att.load_state_dict({n[len(pre) :]: f[n] for n in f.files if n.startswith(pre)})
    q = Tensor(f["query"], requires_grad=True)
    k = Tensor(f["keys"], requires_grad=True)
    ctx, a = att(q, k, length_mask(f["lengths"], f["keys"].shape[1]))
    ht = att.attentional(q, ctx)
    assert_close(a.data, f[f"{score}.weights"], rtol=1e-4, atol=1e-6)
    assert_close(ctx.data, f[f"{score}.context"], rtol=1e-4, atol=1e-6)
    assert_close(ht.data, f[f"{score}.attentional"], rtol=1e-4, atol=1e-6)
    (F.sum(ht * Tensor(f["gh"])) + F.sum(a * Tensor(f["gw"]))).backward()
    assert_close(q.grad, f[f"{score}.grad.query"], rtol=1e-4, atol=1e-6, msg="query")
    assert_close(k.grad, f[f"{score}.grad.keys"], rtol=1e-4, atol=1e-6, msg="keys")
    for name, p in att.named_parameters():
        assert_close(p.grad, f[f"{score}.grad.{name}"], rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("score", SCORES)
def test_gradcheck_every_input_and_parameter(score):
    # WHY: backward reaches the decoder state, the encoder outputs, and every
    #      parameter of each score and of W_c; checked against the frozen
    #      central differences in float64 with one row padded.
    # KIND: gradcheck
    # CATCHES: m03
    # CHAPTER: L4.3 section 2.2, Three scores
    att = LuongAttention(3, score, rng=Rng(2))
    params = [p for _, p in att.named_parameters()]
    for p in params:
        p.data = p.data.astype(np.float64)
    q0, k0 = rand_inputs(3, B=2, S=4, d=3, dtype=np.float64)
    mask = length_mask(np.array([4, 2]), 4)
    g = PCG32(seed=4)
    gh, gw = g.normal_array((2, 3)), g.normal_array((2, 4))

    def f(q, k, *ps):
        for p, a in zip(params, ps):
            p.data = a
        qt = Tensor(q, dtype=np.float64)
        ctx, w = att(qt, Tensor(k, dtype=np.float64), mask)
        return float(np.sum(att.attentional(qt, ctx).data * gh) + np.sum(w.data * gw))

    start = [p.data.copy() for p in params]
    q = Tensor(q0, requires_grad=True, dtype=np.float64)
    k = Tensor(k0, requires_grad=True, dtype=np.float64)
    ctx, w = att(q, k, mask)
    (F.sum(att.attentional(q, ctx) * gh) + F.sum(w * gw)).backward()
    analytic = [q.grad, k.grad] + [p.grad for p in params]
    gradcheck(
        f,
        [q0, k0] + start,
        analytic,
        names=["query", "keys"] + [n for n, _ in att.named_parameters()],
    )


# --- relations between the scores ------------------------------------------------------------


def test_general_with_identity_is_dot():
    # WHY: general is a learned bilinear form h . (W_a k); with W_a = I it is
    #      exactly the dot score. Dot is the special case that needs no
    #      parameters, and the one the transformer keeps (scaled, L5.1).
    # KIND: property
    # CATCHES: m06
    # CHAPTER: L4.3 section 2.2, Three scores
    q, k = rand_inputs(5)
    mask = length_mask(np.array([4, 1, 3]), 4)
    dot = LuongAttention(5, "dot", rng=Rng(3))
    gen = LuongAttention(5, "general", rng=Rng(3))
    gen.load_state_dict(
        {"score_proj.weight": np.eye(5), "combine.weight": dot.combine.weight.data}
    )
    c1, a1 = dot(Tensor(q), Tensor(k), mask)
    c2, a2 = gen(Tensor(q), Tensor(k), mask)
    assert_close(a2.data, a1.data, rtol=1e-6, atol=1e-7)
    assert_close(c2.data, c1.data, rtol=1e-6, atol=1e-7)


def test_general_is_not_symmetric():
    # WHY: h . (W_a k) uses W_a, not its transpose: with a W_a that has one
    #      off-diagonal entry the two give different scores, so loading Luong
    #      weights works only one way round. W_a = [[0, 1], [0, 0]] maps
    #      k = (0, 1) to (1, 0), so h = (1, 0) scores it 1; the transpose
    #      would score it 0.
    # KIND: unit
    # CATCHES: s02, s07, m06
    # CHAPTER: L4.3 section 2.2, Three scores
    gen = LuongAttention(2, "general", rng=Rng(4))
    gen.load_state_dict(
        {
            "score_proj.weight": [[0.0, 1.0], [0.0, 0.0]],
            "combine.weight": np.zeros((2, 4)),
        }
    )
    e = gen.scores(Tensor(HAND_Q), Tensor(HAND_K)).data
    assert_close(e, [[0.0, 1.0, 0.0]], dtype="float32")


def test_concat_projection_splits_the_weight():
    # WHY: W_a [h ; k] = W_h h + W_k k, with W_h the first d columns of W_a
    #      and W_k the last d. project_keys computes W_k k once per sentence;
    #      the score must equal v . tanh(W_a [h ; k]) computed on the
    #      concatenation directly.
    # KIND: property
    # CATCHES: s03, m01, m02
    # CHAPTER: L4.3 section 2.3, Precomputing the keys
    att = LuongAttention(5, "concat", rng=Rng(5))
    q, k = rand_inputs(6)
    W = att.score_proj.weight.data.astype(np.float64)
    v = att.v.weight.data.astype(np.float64)[0]
    hq = np.repeat(q[:, None, :], k.shape[1], axis=1).astype(np.float64)
    want = np.tanh(np.concatenate([hq, k], axis=-1) @ W.T) @ v
    proj = att.project_keys(Tensor(k))
    assert_close(proj.data, k.astype(np.float64) @ W[:, 5:].T, rtol=1e-5, atol=1e-6)
    assert_close(
        att.scores(Tensor(q), Tensor(k), proj).data, want, rtol=1e-5, atol=1e-6
    )
    assert_close(att.scores(Tensor(q), Tensor(k)).data, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("score", SCORES)
def test_masked_weights_are_exactly_zero(score):
    # WHY: as in L4.2, a padded key gets weight exactly 0 and no gradient, and
    #      the real ones sum to 1, for every score; a fully masked row reads
    #      nothing (weights and context 0, not NaN).
    # KIND: property
    # CATCHES: s04, s05
    # CHAPTER: L4.3 section 5, Pitfalls, item 3
    att = LuongAttention(5, score, rng=Rng(6))
    q, k = rand_inputs(7)
    lens = np.array([2, 0, 4])
    kt = Tensor(k, requires_grad=True)
    ctx, a = att(Tensor(q), kt, length_mask(lens, 4))
    w = a.data.astype(np.float64)
    assert (
        np.all(w[0, 2:] == 0.0) and np.all(w[1] == 0.0) and np.all(ctx.data[1] == 0.0)
    )
    assert np.isfinite(w).all() and np.isfinite(ctx.data).all()
    assert_close(w[[0, 2]].sum(axis=1), [1.0, 1.0], rtol=1e-6, atol=1e-6)
    F.sum(ctx).backward()
    assert np.all(kt.grad[0, 2:] == 0.0) and np.all(kt.grad[1] == 0.0)


def test_dot_is_not_scaled():
    # WHY: Luong's dot score is the plain inner product. Dividing by sqrt(d)
    #      is the transformer's fix for large d (L5.1, where the tests ask for
    #      it); here it would change every weight. The hand example's scores
    #      are exactly (1, 0).
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L4.3 section 5, Pitfalls, item 4
    att = LuongAttention(2, "dot", rng=Rng(0))
    assert_close(
        att.scores(Tensor(HAND_Q), Tensor(HAND_K)).data,
        [[1.0, 0.0, 2.0]],
        dtype="float32",
    )


# --- the interface ----------------------------------------------------------------------------------


def test_parameter_names_and_shapes():
    # WHY: the names are the safetensors keys of a Luong seq2seq checkpoint
    #      (L4.1, the zoo); each score registers exactly its own parameters,
    #      W_c last. query_from tells the decoder to attend AFTER its step.
    # KIND: unit
    # CATCHES: m04
    # CHAPTER: L4.3 section 4, The interface
    want = {
        "dot": [("combine.weight", (4, 8))],
        "general": [("score_proj.weight", (4, 4)), ("combine.weight", (4, 8))],
        "concat": [
            ("score_proj.weight", (4, 8)),
            ("v.weight", (1, 4)),
            ("combine.weight", (4, 8)),
        ],
    }
    for score, rows in want.items():
        att = LuongAttention(4, score, rng=Rng(7))
        assert [(n, v.shape) for n, v in att.state_dict().items()] == rows
        assert att.query_from == "current" and att.score == score


def test_validation():
    # WHY: Luong's scores compare h and k_s directly, so both must have width
    #      d; an unknown score name or a wrong mask is a wiring bug.
    # KIND: boundary
    # CATCHES: m05
    # CHAPTER: L4.3 section 4, The interface
    att = LuongAttention(5, "general", rng=Rng(8))
    q, k = rand_inputs(9)
    mask = length_mask(np.array([4, 4, 4]), 4)
    with pytest.raises(ValueError):
        LuongAttention(5, "scaled")
    with pytest.raises(ValueError):
        LuongAttention(0, "dot")
    with pytest.raises(ValueError):
        att(Tensor(q[:, :4]), Tensor(k), mask)
    with pytest.raises(ValueError):
        att(Tensor(q), Tensor(k[..., :4]), mask)
    with pytest.raises(ValueError):
        att(Tensor(q), Tensor(k), mask[:, :3])
    with pytest.raises(ValueError):
        att.attentional(Tensor(q), Tensor(q[:, :4]))
