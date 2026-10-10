"""Course tests for L8.6: speculative decoding (tinyllm/infer/spec.py).

Rung R0 for the course suite: read these before you write code. Each test
names why it exists (WHY), what kind of check it is (KIND), the planted bugs
it kills (CATCHES, mutants in course/mutants/L8.6), and the chapter section
it comes from.

The exactness claims are checked without sampling noise. Uniforms are not
drawn at random: a scripted source enumerates every outcome of each draw on
a grid of midpoints (j + 1/2) / N, and the toy distributions are multiples
of 1/N, so summing the grid weights is an exact integral and the output
distribution is compared with the target's as fractions. The greedy claim
uses _toy.BagLM, whose logits at a position cannot change by a bit whether
the target scores one token or five (see its docstring), so speculative
greedy must equal plain greedy token for token.
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from _toy import BagLM, Letters, greedy_reference

from tinyllm.infer.generate import generate
from tinyllm.infer.sample import SamplingParams, request_rng, sampling_distribution
from tinyllm.infer.spec import (
    ModelDraft,
    NGramDraft,
    PromptLookupDraft,
    speculative_generate,
    verify_draft,
)
from tinyllm.lm.ngram import NGramLM

GREEDY = SamplingParams(temperature=0.0, max_tokens=40)


class NeedMore(Exception):
    pass


class Script:
    """Returns fixed uniforms in order; asking past the end raises NeedMore,
    which the enumerator turns into a branch."""

    def __init__(self, values=()):
        self.values = list(values)
        self.calls = 0

    def uniform(self) -> float:
        if self.calls >= len(self.values):
            raise NeedMore
        v = self.values[self.calls]
        self.calls += 1
        return v


def enumerate_draws(fn, n_grid: int) -> dict:
    """{outcome: probability as a Fraction} of fn(source) over independent
    uniforms, each taking the values (j + 1/2) / n_grid with weight 1/n_grid."""
    out: dict = defaultdict(Fraction)
    todo = [()]
    while todo:
        prefix = todo.pop()
        src = Script([(j + 0.5) / n_grid for j in prefix])
        try:
            res = fn(src)
        except NeedMore:
            todo += [prefix + (j,) for j in range(n_grid)]
            continue
        assert src.calls == len(prefix)
        out[res] += Fraction(1, n_grid ** len(prefix))
    return dict(out)


def logits_of(p) -> np.ndarray:
    """Logits whose softmax is p (zeros become -inf, a token never drawn)."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.log(p)


T1 = SamplingParams(temperature=1.0)


# --- the worked example ----------------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked example, one draft token x = 1 with
    #      target P = [1/2, 1/4, 1/4] and draft Q = [1/4, 1/2, 1/4]. Keep
    #      x with probability min(1, P1 / Q1) = 1/2: u_accept = 0.7 rejects,
    #      and the residual max(0, P - Q) = [1/4, 0, 0] normalizes to
    #      [1, 0, 0], so the emitted token is 0 whatever u_resample is.
    #      u_accept = 0.3 keeps x; u_resample (0.9) is drawn but not used;
    #      the bonus token comes from the second row, which can only be 2,
    #      with the third uniform.
    # KIND: unit, smoke
    # CATCHES: s01, s04, s06, s07
    # CHAPTER: L8.6 section 3
    P = [0.5, 0.25, 0.25]
    rows = np.stack([logits_of(P), logits_of([0.0, 0.0, 1.0])])
    Q = np.array([[0.25, 0.5, 0.25]])
    assert verify_draft(rows, [1], Q, T1, [], Script([0.7, 0.4])) == ([0], 0)
    src = Script([0.3, 0.9, 0.1])
    assert verify_draft(rows, [1], Q, T1, [], src) == ([1, 2], 1)
    assert src.calls == 3  # two per draft position, one for the bonus


def test_hand_example_greedy():
    # WHY: greedy verification by hand: rows with argmax 2, 0, 1. Draft
    #      [2, 1]: position 0 matches (keep 2), position 1 does not (the
    #      target says 0), so emit [2, 0] with one acceptance and no draw.
    #      Draft [2, 0] matches twice and the bonus is row 2's argmax, 1.
    # KIND: unit, smoke
    # CATCHES: s05, s06
    # CHAPTER: L8.6 section 3
    rows = np.array([[0.0, 1.0, 3.0], [2.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    src = Script([])
    assert verify_draft(rows, [2, 1], None, GREEDY, [], src) == ([2, 0], 1)
    assert verify_draft(rows, [2, 0], None, GREEDY, [], src) == ([2, 0, 1], 2)
    assert verify_draft(rows[:1], [], None, GREEDY, [], src) == ([2], 0)
    assert src.calls == 0


# --- exactness by enumeration ------------------------------------------------------------


P5 = [Fraction(3, 8), Fraction(1, 4), Fraction(0), Fraction(1, 4), Fraction(1, 8)]
Q5 = [Fraction(1, 8), Fraction(1, 2), Fraction(1, 4), Fraction(0), Fraction(1, 8)]


def test_one_token_output_is_the_target_exactly():
    # WHY: Leviathan et al.'s theorem on the toy vocabulary of 5: a draft
    #      x ~ Q, kept with probability min(1, P(x)/Q(x)), else replaced by a
    #      residual draw, comes out distributed exactly as P. Enumerated over
    #      x (weight Q(x)) and every grid value of the two uniforms: the
    #      result equals P as fractions, not approximately.
    # KIND: statistical
    # CATCHES: s01, s03, s07
    # CHAPTER: L8.6 section 2.2
    rows = np.stack([logits_of([float(x) for x in P5])] * 2)
    Qf = np.array([[float(x) for x in Q5]])
    dist: dict = defaultdict(Fraction)
    for x, qx in enumerate(Q5):
        if qx == 0:
            continue
        for (first,), w in enumerate_draws(
            lambda s, x=x: tuple(verify_draft(rows, [x], Qf, T1, [], s)[0][:1]), 8
        ).items():
            dist[first] += qx * w
    assert [dist[i] for i in range(5)] == P5


TARGET2 = {  # P(y2 | y1) for the second position, multiples of 1/4
    0: [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2), Fraction(0), Fraction(0)],
    1: [Fraction(0), Fraction(3, 4), Fraction(0), Fraction(0), Fraction(1, 4)],
    3: [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4), Fraction(0), Fraction(0)],
}
P1 = [Fraction(1, 4), Fraction(1, 2), Fraction(0), Fraction(1, 4), Fraction(0)]
Q1 = [Fraction(1, 2), Fraction(1, 4), Fraction(0), Fraction(1, 4), Fraction(0)]
Q2 = [Fraction(0), Fraction(1, 2), Fraction(1, 4), Fraction(0), Fraction(1, 4)]
# Every acceptance ratio and every residual's cumulative sum above is a
# multiple of 1/4, so a grid of 4 midpoints per uniform integrates exactly.


def test_two_token_draft_is_exact_at_each_position():
    # WHY: with k = 2 the second position is checked against the target
    #      AFTER the first draft token, and only when the first was kept. So
    #      the first emitted token must follow P1, and, among runs whose
    #      first token y1 was a kept draft, the second must follow P(. | y1).
    #      Using the wrong row, or the wrong history, breaks one of the two.
    # KIND: statistical
    # CATCHES: s01, s03, s07
    # CHAPTER: L8.6 section 2.3
    first: dict = defaultdict(Fraction)
    second: dict = defaultdict(lambda: defaultdict(Fraction))
    for x1, q1 in enumerate(Q1):
        if q1 == 0 or x1 not in TARGET2:
            continue
        for x2, q2 in enumerate(Q2):
            if q2 == 0:
                continue
            rows = np.stack(
                [
                    logits_of([float(v) for v in P1]),
                    logits_of([float(v) for v in TARGET2[x1]]),
                    logits_of([1.0, 0, 0, 0, 0]),
                ]
            )
            Q = np.array([[float(v) for v in Q1], [float(v) for v in Q2]])
            res = enumerate_draws(
                lambda s, x1=x1, x2=x2: tuple(
                    verify_draft(rows, [x1, x2], Q, T1, [], s)[0]
                ),
                4,
            )
            for em, w in res.items():
                first[em[0]] += q1 * q2 * w
                if len(em) >= 2:
                    second[em[0]][em[1]] += q1 * q2 * w
    assert [first[i] for i in range(5)] == P1
    for y1, cond in second.items():
        total = sum(cond.values())
        assert [cond[i] / total for i in range(5)] == TARGET2[y1], f"after y1 = {y1}"


def test_deterministic_draft_is_exact():
    # WHY: n-gram and prompt-lookup drafts have no distribution: the draft
    #      is one fixed id, which is Q = one-hot. Then a kept draft has
    #      probability P(x) and the residual is P without x, renormalized:
    #      the output is still exactly P, for every choice of x (including
    #      one the target gives probability 0).
    # KIND: statistical
    # CATCHES: s02, s03, s09
    # CHAPTER: L8.6 section 2.2
    PD = [Fraction(1, 4)] * 4 + [Fraction(0)]  # residuals in thirds: grid of 24
    rows = np.stack([logits_of([float(x) for x in PD])] * 2)
    for x in range(5):
        res = enumerate_draws(
            lambda s, x=x: tuple(verify_draft(rows, [x], None, T1, [], s)[0][:1]), 24
        )
        assert [res.get((i,), Fraction(0)) for i in range(5)] == PD, f"draft {x}"


def test_penalties_see_the_accepted_drafts():
    # WHY: the target's sampler applies penalties to the ids generated so
    #      far, and at position i those include the drafts already kept in
    #      this call. A greedy target with a strong repetition penalty must
    #      reject a second copy of a kept token that the unpenalized row
    #      would have chosen.
    # KIND: unit
    # CATCHES: s05, s08
    # CHAPTER: L8.6 section 2.3
    rows = np.array([[3.0, 2.9, 0.0], [3.0, 2.9, 0.0]] + [[0.0, 0.0, 1.0]])
    p = SamplingParams(temperature=0.0, repetition_penalty=2.0)
    emitted, n = verify_draft(rows, [0, 0], None, p, [], Script([]))
    assert (emitted, n) == ([0, 1], 1)


def test_verify_checks_shapes():
    # WHY: m drafts need m + 1 target rows (one per draft plus the bonus)
    #      and, when given, m draft distributions; anything else is a caller
    #      bug that would silently verify against the wrong row.
    # KIND: boundary
    # CATCHES: m001, m002
    # CHAPTER: L8.6 section 4
    rows = np.zeros((2, 3))
    with pytest.raises(ValueError):
        verify_draft(rows, [0, 1], None, GREEDY, [], Script([]))
    with pytest.raises(ValueError):
        verify_draft(rows, [0], np.full((2, 3), 1 / 3), T1, [], Script([0.1, 0.1, 0.1]))


# --- drafts ----------------------------------------------------------------------------------


def test_prompt_lookup_hand_example():
    # WHY: prompt lookup copies from the context: for ctx = 1 2 3 9 1 2,
    #      the 3-gram 9 1 2 never occurred before, the 2-gram 1 2 did (at
    #      the start), so the draft is what followed it: 3 9 1. Among
    #      several earlier occurrences the most recent wins.
    # KIND: unit, smoke
    # CATCHES: s10, s11, s12
    # CHAPTER: L8.6 section 3
    d = PromptLookupDraft(max_ngram=3)
    assert d.propose([1, 2, 3, 9, 1, 2], 3, None) == ([3, 9, 1], None)
    assert d.propose([5, 1, 7, 1, 8, 1], 2, None) == ([8, 1], None)
    assert d.propose([4, 4, 4], 5, None) == (
        [4],
        None,
    )  # the continuation may reach the suffix itself
    assert d.propose([1, 2, 3], 2, None) == ([], None)
    assert d.propose([1, 2, 1, 2], 0, None) == ([], None)
    assert d.propose([1, 2, 3, 7, 2, 3, 1, 2, 3], 2, None) == (
        [7, 2],
        None,
    )  # the longest n-gram first
    with pytest.raises(ValueError):
        PromptLookupDraft(max_ngram=1, min_ngram=2)


def test_ngram_draft_follows_the_model():
    # WHY: the n-gram draft extends the context with the model's own
    #      choices: greedily, each id is the argmax of lm.logprobs(ctx +
    #      drafted so far); sampling, each row is the model's distribution
    #      (softmax of its logprobs), and the id was drawn from that row.
    # KIND: unit
    # CATCHES: s13, s14
    # CHAPTER: L8.6 section 2.4
    rng = PCG32(3)
    seqs = [[rng.below(6) for _ in range(30)] for _ in range(20)]
    lm = NGramLM(3, vocab_size=6)
    lm.fit(seqs)
    d = NGramDraft(lm)
    ctx = [1, 4, 2]
    ids, probs = d.propose(ctx, 4, None)
    cur = list(ctx)
    for t in ids:
        lp = lm.logprobs(cur)
        assert t == int(np.argmax(lp))
        cur.append(t)
    assert probs is None and len(ids) == 4
    ids, probs = d.propose(ctx, 3, request_rng(0))
    cur = list(ctx)
    for t, row in zip(ids, probs):
        assert_close(row, np.exp(lm.logprobs(cur)), rtol=0, atol=1e-12)
        assert row[t] > 0
        cur.append(t)


def test_model_draft_syncs_its_cache():
    # WHY: a model draft keeps a KV cache between calls. When the context
    #      changes behind it (a draft was rejected), it must drop the cached
    #      positions past the common prefix, and re-feed at least the last
    #      context id even when everything is cached (asked twice for the
    #      same context); its proposals must equal greedy decoding of the
    #      draft model from scratch.
    # KIND: unit
    # CATCHES: s15, s16
    # CHAPTER: L8.6 section 2.4
    m = BagLM(seed=5)
    d = ModelDraft(m)
    for ctx in ([1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 4], [1, 2, 5], [6], [6, 0, 2, 2, 7]):
        ids, probs = d.propose(ctx, 4, None)
        assert ids == greedy_reference(m, ctx, 4) and probs is None
    ids, probs = d.propose([3, 1], 2, request_rng(1))
    assert probs.shape == (2, 8)
    assert_close(probs.sum(axis=1), np.ones(2), rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        ModelDraft(m, temperature=0.0)


# --- the decoding loop ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "draft_kind", ["prompt-lookup", "ngram", "model-same", "model-other"]
)
def test_speculative_greedy_equals_greedy(draft_kind):
    # WHY: the guarantee the engine (L10.8) is sold on: with greedy decoding,
    #      speculation changes the speed, never the output. Four drafts that
    #      agree with the target to different degrees all give exactly the
    #      ids of L8.2's generate and of a plain argmax loop.
    # KIND: differential
    # CATCHES: s05, s06, s17, s18, s19
    # CHAPTER: L8.6 section 2.1
    target, tok = BagLM(seed=1), Letters()
    prompt = "abcabcabdabc"
    if draft_kind == "prompt-lookup":
        draft = PromptLookupDraft()
    elif draft_kind == "ngram":
        lm = NGramLM(2, vocab_size=8)
        lm.fit([tok.encode(prompt)])
        draft = NGramDraft(lm)
    else:
        draft = ModelDraft(target if draft_kind == "model-same" else BagLM(seed=9))
    g = speculative_generate(target, draft, tok, prompt, GREEDY, k=4)
    want = greedy_reference(BagLM(seed=1), tok.encode(prompt), 40)
    assert g.ids == want
    assert generate(BagLM(seed=1), tok, prompt, GREEDY).ids == want
    assert g.text == tok.decode(want) and g.finish_reason == "length"
    assert len(g.logprobs) == 40


def test_rejected_drafts_are_rolled_back():
    # WHY: the cache rollback. Every target pass must start exactly where
    #      the kept tokens end: rejected drafts' keys and values are dropped
    #      (KVCache.truncate), and the token that replaced them is fed first
    #      next round. BagLM itself asserts on every call that the cache
    #      holds exactly the positions before the chunk; here, the recorded
    #      calls start at 0 with the prompt, are contiguous, move forward,
    #      and always begin with the token the output has at that position.
    # KIND: property
    # CATCHES: s15, s17, s18, s19
    # CHAPTER: L8.6 section 2.1
    target, tok = BagLM(seed=1), Letters()
    g = speculative_generate(
        target, ModelDraft(BagLM(seed=9)), tok, "abcab", GREEDY, k=3
    )
    full = tok.encode("abcab") + g.ids
    assert (
        target.calls[0][0][0] == 0 and target.calls[0][1][:5] == full[:5]
    )  # the prompt first
    last = -1
    for pos, ids in target.calls:
        assert pos == list(range(pos[0], pos[0] + len(ids)))
        assert pos[0] > last
        assert ids[0] == full[pos[0]], (
            "a round must start by feeding the last kept token"
        )
        last = pos[0]
    assert 0 < g.stats["acceptance_rate"] < 1, "the test wants some rejections"


def test_stats_and_one_pass_per_round():
    # WHY: speculation's point is fewer target passes: with the target as
    #      its own draft every proposal is kept, so 40 tokens at k = 4 take
    #      ceil(40 / 5) = 8 passes and the acceptance rate is 1; with a
    #      useless draft (k = 0) it is one pass per token.
    # KIND: unit
    # CATCHES: s20, s21
    # CHAPTER: L8.6 section 4
    tok = Letters()
    t = BagLM(seed=1)
    g = speculative_generate(t, ModelDraft(BagLM(seed=1)), tok, "abc", GREEDY, k=4)
    assert g.stats["target_calls"] == 8 and len(t.calls) == 8
    assert (
        g.stats["acceptance_rate"] == 1.0
        and g.stats["drafted"] == g.stats["accepted"] == 32
    )
    assert g.stats["completion_tokens"] == 40 and g.stats["prompt_tokens"] == 3
    g0 = speculative_generate(
        BagLM(seed=1), PromptLookupDraft(), tok, "abc", GREEDY, k=0
    )
    assert (
        g0.stats["target_calls"] == 40
        and g0.stats["drafted"] == 0
        and g0.stats["acceptance_rate"] == 0.0
    )


def test_sampled_speculation_matches_the_target_distribution():
    # WHY: end to end at temperature 1: the first generated token over 400
    #      seeds, speculating with a different model, must follow the
    #      target's next-token distribution (chi-square, p > 1e-3 at the
    #      fixed seeds), like plain sampling does.
    # KIND: statistical
    # CATCHES: s01, s03, s16
    # CHAPTER: L8.6 section 2.2
    target, tok = BagLM(seed=1, scale=1.0), Letters()
    prompt = "abcab"
    P = sampling_distribution(
        target.forward(np.asarray([tok.encode(prompt)]))[0, -1], T1
    )
    counts = np.zeros(8)
    draft = ModelDraft(BagLM(seed=9, scale=1.0))
    for s in range(400):
        p = SamplingParams(temperature=1.0, max_tokens=2, seed=s)
        counts[
            speculative_generate(
                BagLM(seed=1, scale=1.0), draft, tok, prompt, p, k=2
            ).ids[0]
        ] += 1
    exp = 400 * P
    keep = exp >= 5
    chi2 = float((((counts - exp) ** 2) / exp)[keep].sum() + 0.0)
    dof = int(keep.sum()) - 1
    # Upper 1e-3 quantile of chi-square for dof <= 7 is at most 24.3.
    assert dof >= 2 and chi2 < 24.3, (
        f"chi2 {chi2:.2f} with {dof} dof, counts {counts}, expected {exp.round(1)}"
    )


def test_eos_stop_and_budget():
    # WHY: speculative_generate keeps generate's contract: an EOS id ends
    #      the output (and is not in it), a stop string cuts the text before
    #      it, and never more than max_tokens ids even when a round emits
    #      k + 1 tokens at once.
    # KIND: boundary
    # CATCHES: s22, s23, s24
    # CHAPTER: L8.6 section 4
    tok = Letters()
    want = greedy_reference(BagLM(seed=1), tok.encode("abc"), 40)
    eos = want[7]
    g = speculative_generate(
        BagLM(seed=1), ModelDraft(BagLM(seed=1)), tok, "abc", GREEDY, k=4, eos_ids=[eos]
    )
    assert g.ids == want[: want.index(eos)] and g.finish_reason == "stop"
    stop = tok.decode(want[5:7])
    p = SamplingParams(temperature=0.0, max_tokens=40, stop=[stop])
    g = speculative_generate(
        BagLM(seed=1), ModelDraft(BagLM(seed=1)), tok, "abc", p, k=4
    )
    full = tok.decode(want)
    assert g.text == full[: full.index(stop)] and g.finish_reason == "stop"
    for n in (1, 3, 7):
        g = speculative_generate(
            BagLM(seed=1),
            ModelDraft(BagLM(seed=1)),
            tok,
            "abc",
            SamplingParams(temperature=0.0, max_tokens=n),
            k=4,
        )
        assert g.ids == want[:n]


def test_bad_arguments():
    # WHY: a negative k, an empty prompt, and a prompt longer than the
    #      model's context are caller errors, reported before any pass.
    # KIND: boundary
    # CATCHES: m003
    # CHAPTER: L8.6 section 4
    tok = Letters()
    with pytest.raises(ValueError):
        speculative_generate(BagLM(), PromptLookupDraft(), tok, "abc", GREEDY, k=-1)
    with pytest.raises(ValueError):
        speculative_generate(BagLM(), PromptLookupDraft(), tok, "", GREEDY)
    with pytest.raises(ValueError):
        speculative_generate(
            BagLM(max_len=4), PromptLookupDraft(), tok, "abcde", GREEDY
        )


def test_logprobs_are_the_targets():
    # WHY: each reported logprob is L8.1's token_logprobs of the target row
    #      that produced the id (not the draft's): the same numbers plain
    #      generate reports.
    # KIND: unit
    # CATCHES: s25
    # CHAPTER: L8.6 section 4
    tok = Letters()
    g = speculative_generate(
        BagLM(seed=1), ModelDraft(BagLM(seed=9)), tok, "abcab", GREEDY, k=3
    )
    ref = generate(BagLM(seed=1), tok, "abcab", GREEDY)
    assert_close(np.array(g.logprobs), np.array(ref.logprobs), rtol=0, atol=1e-12)
    assert all(math.isfinite(x) and x <= 0 for x in g.logprobs)
