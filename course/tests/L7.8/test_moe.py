"""Course tests for L7.8: mixture of experts (tinyllm/modern/moe.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L7.8), and the chapter section it comes
from.

The worked example of the chapter (section 3): 4 experts, top 2, two
tokens with router logits z0 = (ln 2, ln 4, 0, 0) and
z1 = (ln 3, 0, ln 3, ln 2). Softmax gives p0 = (1/4, 1/2, 1/8, 1/8) and
p1 = (1/3, 1/9, 1/3, 2/9). Token 0 goes to experts (1, 0) with weights
(2/3, 1/3); token 1 ties between experts 0 and 2 and takes both, lowest id
first: (0, 2) with (1/2, 1/2). Sorted dispatch: perm (1, 2, 0, 3), rows of
tokens (0, 1, 0, 1), offsets (0, 2, 3, 4, 4), inv_perm (2, 0, 1, 3).
Expert outputs (10, 20, 30, 40) in sorted order combine to 70/3 and 30.
The load-balancing loss is 4 (1 * 7/24 + 1/2 * 11/36 + 1/2 * 11/48) =
161/72.

The golden fixture (course/fixtures/L7.8/moe_hf.npz) holds transformers
5.19.0 MixtralSparseMoeBlock (and its load_balancing_loss_func),
Qwen3MoeSparseMoeBlock with norm_topk_prob False, and DeepseekV3MoE (sigmoid
router, correction bias, two shared experts, routed scaling 2.5), float32,
from course/oracle/L7.8/moe_hf.py.
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
from tinyllm.modern.moe import MoE, combine, dispatch, load_balance_loss, topk_ids

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L7.8", "moe_hf.npz")
GOLDEN = {
    "mixtral": dict(d_ff_expert=12, n_experts=4, top_k=2),
    "qwen3": dict(d_ff_expert=8, n_experts=4, top_k=2, norm_topk=False),
    "deepseek": dict(
        d_ff_expert=6,
        n_experts=8,
        top_k=2,
        n_shared=2,
        router="sigmoid",
        routed_scaling=2.5,
    ),
}


def random_moe(seed: int = 0, **kw) -> MoE:
    kw = dict(dict(d_ff_expert=5, n_experts=6, top_k=2), **kw)
    m = MoE(8, **kw)
    g = PCG32(seed=300 + seed)
    for _, p in m.named_parameters():
        p.data = 0.5 * g.normal_array(p.shape)
    return m


def oracle_weights(m: MoE, z: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The router's weights for the chosen experts, from the chapter's table."""
    if m.router == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-z))
        w = np.take_along_axis(s, idx, -1)
        w = w / (w.sum(-1, keepdims=True) + 1e-20) if m.norm_topk else w
    elif m.router == "softmax_topk":
        p = np.exp(z - z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        w = np.take_along_axis(p, idx, -1)
        w = w / w.sum(-1, keepdims=True) if m.norm_topk else w
    else:
        zz = np.take_along_axis(z, idx, -1)
        w = np.exp(zz - zz.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
    return w * m.routed_scaling


# --- the worked example ------------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example: softmax-then-top-k routing with a
    #      tie, the renormalized weights, the sorted dispatch arrays, the
    #      combine step, and the load-balancing loss, all on paper.
    # KIND: unit
    # CATCHES: s03, s04, s05, s07, s11
    # CHAPTER: L7.8 section 3, Worked example by hand
    m = MoE(2, 3, n_experts=4, top_k=2)
    z = np.array(
        [
            [math.log(2), math.log(4), 0.0, 0.0],
            [math.log(3), 0.0, math.log(3), math.log(2)],
        ]
    )
    m.gate.weight.data = z.T.astype(
        np.float32
    ).copy()  # x = identity rows, so the logits are z
    x = Tensor(np.eye(2))
    idx, w, aux = m.route(x)
    assert idx.tolist() == [[1, 0], [0, 2]]
    assert_close(w.data, [[2 / 3, 1 / 3], [0.5, 0.5]], rtol=1e-6, atol=1e-6)
    assert_close(aux.data / 0.01, 161 / 72, rtol=1e-6, atol=1e-6)
    xs, offsets, inv_perm = dispatch(Tensor(np.array([[100.0], [200.0]])), idx, 4)
    assert xs.data[:, 0].tolist() == [100.0, 200.0, 100.0, 200.0]
    assert offsets.tolist() == [0, 2, 3, 4, 4] and inv_perm.tolist() == [2, 0, 1, 3]
    y = combine(
        Tensor(np.array([[10.0], [20.0], [30.0], [40.0]])),
        Tensor(w.data, dtype=np.float64),
        inv_perm,
    )
    assert_close(y.data[:, 0], [70 / 3, 30.0], rtol=1e-6, atol=1e-6)


# --- against Hugging Face --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mixtral", "qwen3", "deepseek"])
def test_moe_golden(name):
    # WHY: three production routers: Mixtral (softmax, top 2, renormalized),
    #      Qwen3-MoE with norm_topk_prob False (raw softmax weights), and
    #      DeepSeek-V3 (sigmoid scores, a correction bias that chooses but
    #      never weighs, two shared experts, routed scaling 2.5). Same
    #      experts chosen, same weights, same output.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s08, s09, s11
    # CHAPTER: L7.8 section 2.1, Routers
    f = np.load(FIX)
    m = MoE(16, **GOLDEN[name])
    m.load_state_dict(
        {
            k[len(name) + 1 :]: f[k]
            for k in f.files
            if k.startswith(name + ".") and k.endswith(".weight")
        }
    )
    if name == "deepseek":
        m.e_score_correction_bias = f["deepseek.gate.e_score_correction_bias"].copy()
    x = f[f"{name}.x"]
    y = m(Tensor(x))
    assert y.shape == x.shape
    assert_close(y.data, f[f"{name}.out"], rtol=1e-4, atol=1e-5, msg="out")
    idx, w, aux = m.route(Tensor(x.reshape(-1, 16)))
    order = np.argsort(idx, axis=-1)
    assert np.array_equal(np.take_along_axis(idx, order, -1), f[f"{name}.topk_idx"])
    assert_close(
        np.take_along_axis(w.data, order, -1),
        f[f"{name}.topk_w"],
        rtol=1e-5,
        atol=1e-6,
        msg="weights",
    )
    if name == "mixtral":
        assert_close(
            aux.data / m.aux_loss_coef,
            f["mixtral.aux"],
            rtol=1e-5,
            atol=1e-6,
            msg="aux",
        )
        assert_close(m.aux_loss.data, aux.data, rtol=1e-6, atol=1e-7)


# --- dispatch ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(router="softmax_topk", norm_topk=False),
        dict(router="topk_softmax"),
        dict(router="sigmoid", n_shared=1, routed_scaling=1.5),
    ],
)
def test_sorted_dispatch_equals_dense_loop(kw):
    # WHY: the batched path (sort assignments by expert, run each expert once
    #      on its slice, scatter back) must equal the obvious loop: for each
    #      token, sum the chosen experts' outputs times the router weights
    #      (computed here from the logits by the chapter's table), plus the
    #      shared experts. float64, 3 routers.
    # KIND: differential
    # CATCHES: s01, s02, s03, s04, s08, s09, s11, s12
    # CHAPTER: L7.8 section 2.2, Sorted dispatch
    m = random_moe(seed=1, **kw)
    if m.router == "sigmoid":
        m.e_score_correction_bias = (0.4 * PCG32(seed=31).normal_array((6,))).astype(
            np.float32
        )
    x = PCG32(seed=32).normal_array((3, 5, 8))
    y = m(Tensor(x, dtype=np.float64)).data
    xf = x.reshape(-1, 8)
    z = xf @ m.gate.weight.data.T
    if m.router == "sigmoid":
        idx = topk_ids(1.0 / (1.0 + np.exp(-z)) + m.e_score_correction_bias, 2)
    elif m.router == "topk_softmax":
        idx = topk_ids(z, 2)
    else:
        idx = topk_ids(z, 2)  # softmax is monotone: the same experts as the logits
    w = oracle_weights(m, z, idx)
    want = np.zeros_like(xf)
    for n in range(xf.shape[0]):
        for j in range(2):
            want[n] += (
                w[n, j]
                * m.experts[int(idx[n, j])](
                    Tensor(xf[n : n + 1], dtype=np.float64)
                ).data[0]
            )
        if m.shared_experts is not None:
            want[n] += m.shared_experts(Tensor(xf[n : n + 1], dtype=np.float64)).data[0]
    assert_close(y.reshape(-1, 8), want, rtol=1e-9, atol=1e-11)
    assert m.expert_load.tolist() == np.bincount(idx.ravel(), minlength=6).tolist()


def test_dispatch_is_a_stable_grouping():
    # WHY: each expert reads one contiguous slice, offsets[e] .. offsets[e+1];
    #      inside a slice the tokens keep their order; inv_perm undoes the
    #      sort exactly, so combine(dispatch(x)) with weights summing to 1
    #      returns x. Gradients of the slices flow back to x.
    # KIND: property
    # CATCHES: s03, s04, s11
    # CHAPTER: L7.8 section 2.2, Sorted dispatch
    g = PCG32(seed=33)
    N, k, E = 9, 3, 5
    idx = np.array([[g.below(E) for _ in range(k)] for _ in range(N)])
    x = Tensor(g.normal_array((N, 4)), requires_grad=True, dtype=np.float64)
    xs, offsets, inv_perm = dispatch(x, idx, E)
    assert offsets[0] == 0 and offsets[-1] == N * k and np.all(np.diff(offsets) >= 0)
    assert sorted(inv_perm.tolist()) == list(range(N * k))
    for e in range(E):
        rows = [n for n in range(N) for j in range(k) if idx[n, j] == e]
        assert_close(
            xs.data[offsets[e] : offsets[e + 1]],
            x.data[rows],
            rtol=0,
            atol=0,
            msg=f"expert {e}",
        )
    w = Tensor(np.full((N, k), 1.0 / k), dtype=np.float64)
    assert_close(combine(xs, w, inv_perm).data, x.data, rtol=1e-12, atol=1e-12)
    F.sum(xs).backward()
    assert_close(x.grad, np.full((N, 4), float(k)), rtol=0, atol=0)


def test_topk_ties_go_to_lowest_id():
    # WHY: one tie rule across the course (greedy sampling, beam search,
    #      routing): equal scores go to the lowest id, so the same logits pick
    #      the same experts in Python, in C, and after a reload.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: L7.8 section 2.1, Routers
    s = np.array([[1.0, 3.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0], [0.0, 1.0, 0.0, 5.0]])
    assert topk_ids(s, 2).tolist() == [[1, 2], [0, 1], [3, 1]]
    assert topk_ids(s, 4)[1].tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        topk_ids(s, 5)


# --- the losses and the bias ---------------------------------------------------------------------------


def test_load_balance_loss_bounds_and_gradient():
    # WHY: L = E sum_e f_e P_e is k at perfect balance and grows when the
    #      router concentrates; f is a count (no gradient), so
    #      dL/dP_e = E f_e and dL/d probs[n, e] = E f_e / N exactly.
    # KIND: property
    # CATCHES: s05
    # CHAPTER: L7.8 section 2.3, The load-balancing loss
    E, k, N = 4, 2, 8
    uniform = Tensor(np.full((N, E), 1.0 / E))
    balanced = np.array([[n % E, (n + 1) % E] for n in range(N)])
    assert_close(
        load_balance_loss(uniform, balanced, E).data, float(k), rtol=1e-12, atol=1e-12
    )
    peaked = np.zeros((N, E))
    peaked[:, 0], peaked[:, 1] = 0.9, 0.1
    same = np.tile([0, 1], (N, 1))
    assert load_balance_loss(Tensor(peaked), same, E).data > float(k)
    p = Tensor(
        PCG32(seed=34).uniform_array((N, E)), requires_grad=True, dtype=np.float64
    )
    idx = np.array([[0, 1], [0, 2], [0, 1], [3, 0], [0, 1], [2, 1], [0, 3], [1, 0]])
    load_balance_loss(p, idx, E).backward()
    f = np.bincount(idx.ravel(), minlength=E) / N
    assert_close(p.grad, np.tile(E * f / N, (N, 1)), rtol=1e-12, atol=1e-14)


def test_gradcheck_router_and_experts():
    # WHY: MoE trains end to end (C1's ablation): the loss reaches x, the
    #      router through the top-k weights (the choice itself is piecewise
    #      constant), and the experts. Frozen central differences, float64,
    #      away from ties.
    # KIND: gradcheck
    # CATCHES: s05, s10, s11
    # CHAPTER: L7.8 section 2.4, Gradients through a discrete choice
    m = random_moe(seed=2, n_shared=1)
    g = PCG32(seed=35)
    x0, w0 = g.normal_array((4, 8)), m.gate.weight.data.astype(np.float64).copy()
    gy = g.normal_array((4, 8))

    def f(x, w):
        m.gate.weight.data = w
        return float(np.sum(m(Tensor(x, dtype=np.float64)).data * gy) + m.aux_loss.data)

    m.gate.weight.data = w0
    m.zero_grad()
    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    y = m(x)
    (F.sum(y * gy) + m.aux_loss).backward()
    gradcheck(f, [x0, w0], [x.grad, m.gate.weight.grad], names=["x", "gate.weight"])


def test_aux_free_bias():
    # WHY: DeepSeek-V3's rule by hand (load (10, 0, 5, 5), rate 0.1 gives
    #      bias (-0.1, +0.1, 0, 0)); the bias changes WHICH experts a token
    #      gets but never the weights, which come from the unbiased scores;
    #      it is not a trained parameter.
    # KIND: unit
    # CATCHES: s02, s06
    # CHAPTER: L7.8 section 2.5, Balancing without a loss
    m = random_moe(seed=3, n_experts=4, router="sigmoid", bias_update_rate=0.1)
    m.update_bias([10, 0, 5, 5])
    assert_close(m.e_score_correction_bias, [-0.1, 0.1, 0.0, 0.0], rtol=0, atol=1e-7)
    assert "e_score_correction_bias" not in " ".join(m.state_dict())
    x = Tensor(PCG32(seed=36).normal_array((6, 8)), dtype=np.float64)
    m.e_score_correction_bias = np.array(
        [0.0, 0.0, 0.0, 5.0], dtype=np.float32
    )  # expert 3 always chosen
    idx, w, _ = m.route(x)
    assert np.all(np.any(idx == 3, axis=-1))
    s = 1.0 / (1.0 + np.exp(-(x.data @ m.gate.weight.data.T)))
    ws = np.take_along_axis(s, idx, -1)
    assert_close(w.data, ws / ws.sum(-1, keepdims=True), rtol=1e-12, atol=1e-12)


def test_bias_updates_balance_the_load():
    # WHY: the point of the rule: a router that sends most tokens to one
    #      expert spreads them out after a few dozen updates, with no loss
    #      term and no gradient. 64 fixed tokens, 60 steps of rate 0.02.
    # KIND: property
    # CATCHES: s06, s11
    # CHAPTER: L7.8 section 2.5, Balancing without a loss
    m = random_moe(
        seed=4, n_experts=4, top_k=1, router="sigmoid", bias_update_rate=0.02
    )
    m.gate.weight.data[0] += (
        2.0 * np.abs(m.gate.weight.data[0]).max()
    )  # expert 0 favored
    x = Tensor(PCG32(seed=37).normal_array((64, 8)) + 1.0, dtype=np.float64)
    m(x)
    first = m.expert_load.copy()
    for _ in range(60):
        m(x)
        m.update_bias(m.expert_load)
    m(x)
    assert first.max() >= 40, first
    assert m.expert_load.max() <= 20, (
        m.expert_load
    )  # 64 tokens over 4 experts: 16 is perfect


def test_state_and_idle_experts():
    # WHY: state_dict keys are the Hub's (gate, experts.<e>.*, shared_experts)
    #      in that order; an expert that receives no token is skipped
    #      without breaking the forward; forward records aux_loss and the
    #      per-expert load without turning them into parameters.
    # KIND: unit
    # CATCHES: s11
    # CHAPTER: L7.8 section 4, The interface
    m = random_moe(seed=5, n_experts=3, n_shared=2)
    keys = list(m.state_dict())
    assert keys[0] == "gate.weight" and keys[1:4] == [
        f"experts.0.{p}_proj.weight" for p in ("gate", "up", "down")
    ]
    assert keys[-3:] == [
        f"shared_experts.{p}_proj.weight" for p in ("gate", "up", "down")
    ]
    assert m.shared_experts.up_proj.weight.shape == (10, 8)
    m.gate.weight.data[:] = 0.0
    m.gate.weight.data[0, 0] = m.gate.weight.data[1, 0] = (
        10.0  # every token: experts 0 and 1
    )
    y = m(Tensor(np.abs(PCG32(seed=38).normal_array((5, 8))) + 1.0, dtype=np.float64))
    assert np.isfinite(y.data).all() and m.expert_load.tolist() == [5, 5, 0]
    assert m.aux_loss is not None and m.aux_loss.data > 0
    assert list(m.state_dict()) == keys  # forward registered nothing new


def test_validation():
    # WHY: top_k beyond the experts, an unknown router, negative rates, and
    #      out-of-range expert ids are wiring bugs; fail loudly.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: L7.8 section 4, The interface
    for kw in (
        dict(top_k=0),
        dict(top_k=7),
        dict(router="hash"),
        dict(n_shared=-1),
        dict(bias_update_rate=-1.0),
        dict(routed_scaling=0.0),
    ):
        args = dict(d_ff_expert=4, n_experts=6, top_k=2)
        args.update(kw)
        with pytest.raises(ValueError):
            MoE(8, **args)
    x = Tensor(np.ones((2, 3)))
    for bad in (
        np.array([[0, 4]]).repeat(2, 0),
        np.array([[-1, 0]]).repeat(2, 0),
        np.array([[0.0, 1.0]]).repeat(2, 0),
    ):
        with pytest.raises(ValueError):
            dispatch(x, bad, 4)
    with pytest.raises(ValueError):
        load_balance_loss(Tensor(np.ones((2, 3))), np.zeros((2, 1), dtype=np.int64), 4)
    with pytest.raises(ValueError):
        MoE(8, 4, 6, 2).update_bias([1, 2, 3])
