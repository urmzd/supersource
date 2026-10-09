"""Course tests for L4.4: beam search, generic over a step function
(tinyllm/infer/beam.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L4.4), and the chapter section it comes
from.

The oracle is exhaustive enumeration: on a vocabulary of 3 and 4 steps there
are only 31 possible outputs, so the test scores every one of them with its
own log-softmax and compares. The models are toy step functions whose
logits depend on the whole prefix (drawn from the frozen PCG32 keyed by a
hash of the prefix), so a search that mixes up whose state is whose gets
different numbers.

The worked example of the chapter (section 3), tokens eos = 0, a = 1, b = 2:
P(. | bos) = (0.1, 0.5, 0.4), P(. | a) = (0.2, 0.1, 0.7),
P(. | b) = (0.9, 0.05, 0.05). Greedy takes a, then b: P = 0.35. A beam of 2
keeps a and b, then finds b eos with P = 0.4 * 0.9 = 0.36.
"""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.infer.beam import Hypothesis, beam_search, greedy_decode, select_state

EOS, A, B = 0, 1, 2
BOS = 9  # outside the vocabulary: bos is only ever an input


def logsoftmax(z: np.ndarray) -> np.ndarray:
    """The test's own float64 log-softmax (not your M09.2)."""
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return z - m - np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))


class TableLM:
    """Next-token probabilities from a dict {prefix tuple: probs}; the prefix
    excludes bos. Missing prefixes are uniform."""

    def __init__(self, table: dict, V: int) -> None:
        self.table, self.V = table, V

    def logits(self, prefix: tuple) -> np.ndarray:
        p = self.table.get(prefix)
        if p is None:
            return np.zeros(self.V)
        with np.errstate(divide="ignore"):  # probability 0 is logit -inf
            return np.log(np.asarray(p, dtype=np.float64))


class ToyLM:
    """Logits that depend on the whole prefix: V normal draws from the frozen
    PCG32 seeded by a hash of (seed, prefix). bias[eos] shifts how soon
    sequences end; masked tokens are -inf everywhere."""

    def __init__(self, V: int, seed: int, eos_bias: float = 0.0, masked=()) -> None:
        self.V, self.seed, self.eos_bias, self.masked = V, seed, eos_bias, tuple(masked)
        self.cache: dict = {}

    def logits(self, prefix: tuple) -> np.ndarray:
        if prefix not in self.cache:
            h = self.seed * 7919 + 17
            for t in prefix:
                h = (h * 1000003 + t + 1) % ((1 << 61) - 1)
            z = 2.0 * PCG32(seed=h).normal_array((self.V,))
            z[EOS] += self.eos_bias
            for v in self.masked:
                z[v] = -np.inf
            self.cache[prefix] = z
        return self.cache[prefix].copy()


def make_step(lm, beam_size=None):
    """step_fn(state, y_prev) over lm with state {"prefix": int64 [k, t]}."""
    calls: list[int] = []

    def step(state, y_prev):
        y_prev = np.asarray(y_prev, dtype=np.int64)
        prefix = state["prefix"]
        assert prefix.shape[0] == y_prev.shape[0], "state rows and y_prev rows differ"
        if beam_size is not None:
            assert y_prev.shape[0] <= beam_size, "more live hypotheses than beam_size"
        calls.append(int(y_prev.shape[0]))
        if state["started"]:
            new = np.concatenate([prefix, y_prev[:, None]], axis=1)
        else:
            assert np.all(y_prev == BOS), "the first call must pass bos"
            new = prefix
        logits = np.stack([lm.logits(tuple(int(t) for t in r)) for r in new])
        return logits, {"prefix": new, "started": True}

    init = {"prefix": np.zeros((1, 0), dtype=np.int64), "started": False}
    return step, init, calls


def all_outputs(V: int, max_len: int, eos: int = EOS):
    """Every output a search can return: sequences that end at their only eos
    (length 1..max_len), and eos-free sequences of length exactly max_len."""
    others = [v for v in range(V) if v != eos]
    for n in range(1, max_len + 1):
        for body in itertools.product(others, repeat=n - 1):
            yield body + (eos,), True
    for body in itertools.product(others, repeat=max_len):
        yield body, False


def seq_logprob(lm, toks) -> float:
    return float(
        sum(logsoftmax(lm.logits(tuple(toks[:i])))[t] for i, t in enumerate(toks))
    )


def brute_force(lm, V, max_len, alpha):
    out = []
    for toks, fin in all_outputs(V, max_len):
        lp = seq_logprob(lm, toks)
        if np.isfinite(lp):
            out.append((lp / len(toks) ** alpha, lp, list(toks), fin))
    return sorted(out, key=lambda r: -r[0])


def hand_lm() -> TableLM:
    return TableLM(
        {(): [0.1, 0.5, 0.4], (A,): [0.2, 0.1, 0.7], (B,): [0.9, 0.05, 0.05]}, V=3
    )


# --- the worked example ---------------------------------------------------------------


def test_hand_example_beam_beats_greedy():
    # WHY: the chapter's worked example, number for number. Beam 2 keeps a
    #      (0.5) and b (0.4); step 2 ranks all six extensions together and
    #      keeps b eos (0.36, finished) and a b (0.35, cut at max_len = 2).
    # KIND: unit
    # CATCHES: s01, s03, s04, s05, s11, m01, m02, m03
    # CHAPTER: L4.4 section 3, Worked example by hand
    step, init, calls = make_step(hand_lm(), beam_size=2)
    hyps = beam_search(step, init, BOS, EOS, beam_size=2, max_len=2, length_penalty=0.0)
    assert [h.tokens for h in hyps] == [[B, EOS], [A, B]]
    assert [h.finished for h in hyps] == [True, False]
    assert_close(
        [h.logprob for h in hyps], [math.log(0.36), math.log(0.35)], dtype="float64"
    )
    assert_close(
        [h.score for h in hyps], [math.log(0.36), math.log(0.35)], dtype="float64"
    )
    assert calls == [1, 2]


def test_hand_example_greedy_commits_to_a():
    # WHY: greedy takes the best first token (a, 0.5) and then the best after
    #      it (b, 0.7): 0.35, below the 0.36 the beam found. That gap is why
    #      beam search exists.
    # KIND: unit
    # CATCHES: s05, s11, s15, m01, m02, m03
    # CHAPTER: L4.4 section 3, Worked example by hand
    step, init, _ = make_step(hand_lm())
    h = greedy_decode(step, init, BOS, EOS, max_len=2)
    assert isinstance(h, Hypothesis)
    assert h.tokens == [A, B] and h.finished is False
    assert_close(h.logprob, math.log(0.35), dtype="float64")
    assert_close(h.score, h.logprob, dtype="float64")


def test_hand_example_length_penalty():
    # WHY: with length_penalty = 1 the score is the mean log-probability per
    #      token: ln(0.36) / 2 and ln(0.35) / 2. The logprob field stays the
    #      plain sum.
    # KIND: unit
    # CATCHES: s01, s03, s05, s06, s11, m01, m02, m03
    # CHAPTER: L4.4 section 2.4, Length normalization
    step, init, _ = make_step(hand_lm())
    hyps = beam_search(step, init, BOS, EOS, beam_size=2, max_len=2, length_penalty=1.0)
    assert [h.tokens for h in hyps] == [[B, EOS], [A, B]]
    assert_close(
        [h.logprob for h in hyps], [math.log(0.36), math.log(0.35)], dtype="float64"
    )
    assert_close(
        [h.score for h in hyps],
        [math.log(0.36) / 2, math.log(0.35) / 2],
        dtype="float64",
    )


# --- equivalences -----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_beam_one_is_greedy(seed):
    # WHY: a beam of 1 keeps only the best extension at every step, which is
    #      greedy decoding. The test runs its own greedy loop (argmax, ties to
    #      the lowest id, stop at eos) over a model whose logits depend on the
    #      whole prefix.
    # KIND: differential
    # CATCHES: s02, s04, s05, s11, s16, m01, m02, m03, m04
    # CHAPTER: L4.4 section 2.2, Greedy as a beam of one
    V, max_len = 5, 7
    lm = ToyLM(V, seed, eos_bias=-0.5)
    step, init, _ = make_step(lm, beam_size=1)
    got = beam_search(
        step, init, BOS, EOS, beam_size=1, max_len=max_len, length_penalty=0.0
    )
    toks: list[int] = []
    lp = 0.0
    for _ in range(max_len):
        ls = logsoftmax(lm.logits(tuple(toks)))
        v = int(np.argmax(ls))
        toks.append(v)
        lp += ls[v]
        if v == EOS:
            break
    assert len(got) == 1
    assert got[0].tokens == toks
    assert got[0].finished == (toks[-1] == EOS)
    assert_close(got[0].logprob, lp, dtype="float64")


@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("seed", range(4))
def test_exhaustive_beam_equals_brute_force(seed, alpha):
    # WHY: with beam_size >= V ** max_len nothing is ever pruned, so beam
    #      search must return every possible output, ranked by score: the
    #      exact answer, computed here by enumerating all 31 sequences of a
    #      3-token vocabulary up to length 4.
    # KIND: differential
    # CATCHES: s02, s03, s04, s05, s06, s07, s11, m01, m03, m04
    # CHAPTER: L4.4 section 2.3, The algorithm
    V, max_len = 3, 4
    lm = ToyLM(V, 100 + seed)
    step, init, _ = make_step(lm, beam_size=V**max_len)
    got = beam_search(
        step,
        init,
        BOS,
        EOS,
        beam_size=V**max_len,
        max_len=max_len,
        length_penalty=alpha,
    )
    want = brute_force(lm, V, max_len, alpha)
    assert len(got) == len(want) == 31
    assert [h.tokens for h in got] == [w[2] for w in want]
    assert [h.finished for h in got] == [w[3] for w in want]
    assert_close([h.logprob for h in got], [w[1] for w in want], rtol=1e-12, atol=1e-12)
    assert_close([h.score for h in got], [w[0] for w in want], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_small_beam_is_a_prefix_of_the_exhaustive_ranking_at_step_one(seed):
    # WHY: after one step every live hypothesis is a single token, so the
    #      beam must hold exactly the beam_size most likely first tokens of
    #      ALL candidates: max_len = 1 makes that the whole answer.
    # KIND: property
    # CATCHES: s05, s11, m01, m03
    # CHAPTER: L4.4 section 2.3, The algorithm
    V = 6
    lm = ToyLM(V, 200 + seed)
    step, init, _ = make_step(lm, beam_size=3)
    got = beam_search(step, init, BOS, EOS, beam_size=3, max_len=1, length_penalty=0.0)
    ls = logsoftmax(lm.logits(()))
    order = sorted(range(V), key=lambda v: (-ls[v], v))[:3]
    assert [h.tokens for h in got] == [[v] for v in order]


# --- the beam -------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_beam_never_holds_more_than_beam_size(seed):
    # WHY: the cost of beam search is beam_size rows per step. Keeping the
    #      best beam_size tokens of EACH hypothesis instead of the best
    #      beam_size overall grows the beam to beam_size ** 2 and beyond; the
    #      step function here refuses more rows than beam_size, and the result
    #      is at most beam_size hypotheses.
    # KIND: property
    # CATCHES: s01, s03, m02, m03
    # CHAPTER: L4.4 section 5, Pitfalls, item 1
    lm = ToyLM(6, 300 + seed, eos_bias=-1.0)
    step, init, calls = make_step(lm, beam_size=3)
    hyps = beam_search(step, init, BOS, EOS, beam_size=3, max_len=6, length_penalty=1.0)
    assert 1 <= len(hyps) <= 3
    assert all(k <= 3 for k in calls)
    assert calls[0] == 1


@pytest.mark.parametrize("seed", range(5))
def test_eos_finishes_and_shrinks_the_beam(seed):
    # WHY: a hypothesis that emits eos is complete: it leaves the beam and is
    #      never extended, so eos is its last token and appears once. The beam
    #      then has fewer rows, and the search stops calling the model as soon
    #      as no hypothesis is live.
    # KIND: property
    # CATCHES: s01, s02, s03, s04, s16, m01, m03, m04
    # CHAPTER: L4.4 section 5, Pitfalls, item 3
    lm = ToyLM(4, 400 + seed, eos_bias=2.5)
    step, init, calls = make_step(lm, beam_size=4)
    hyps = beam_search(
        step, init, BOS, EOS, beam_size=4, max_len=30, length_penalty=1.0
    )
    for h in hyps:
        assert h.tokens.count(EOS) == (1 if h.finished else 0)
        if h.finished:
            assert h.tokens[-1] == EOS
        assert_close(h.logprob, seq_logprob(lm, h.tokens), rtol=1e-12, atol=1e-12)
    assert len(calls) < 30, (
        "the search kept calling the model after every hypothesis finished"
    )
    assert all(b <= a for a, b in zip(calls[1:], calls[2:])), (
        "the beam grew after it shrank"
    )


def test_search_stops_when_the_last_hypothesis_finishes():
    # WHY: once every live hypothesis has emitted eos there is nothing to
    #      extend; another call would pass the model zero rows. Here eos is
    #      certain at step 2, so the model is called exactly twice.
    # KIND: boundary
    # CATCHES: s03, s04, s16, m01, m03
    # CHAPTER: L4.4 section 2.3, The algorithm
    lm = TableLM(
        {(): [0.0, 0.6, 0.4], (A,): [1.0, 0.0, 0.0], (B,): [1.0, 0.0, 0.0]}, V=3
    )
    step, init, calls = make_step(lm)
    hyps = beam_search(
        step, init, BOS, EOS, beam_size=2, max_len=10, length_penalty=0.0
    )
    assert calls == [1, 2]
    assert [h.tokens for h in hyps] == [[A, EOS], [B, EOS]]


@pytest.mark.parametrize("max_len", [1, 2, 5])
def test_max_len_returns_unfinished(max_len):
    # WHY: a model that never says eos must still stop: after max_len tokens
    #      the live hypotheses are returned as they stand, unfinished, with
    #      exactly max_len tokens, and the model was called max_len times.
    # KIND: boundary
    # CATCHES: s01, s03, s05, s11, m01, m02, m03
    # CHAPTER: L4.4 section 5, Pitfalls, item 4
    lm = ToyLM(4, 7, masked=[EOS])
    step, init, calls = make_step(lm, beam_size=3)
    hyps = beam_search(
        step, init, BOS, EOS, beam_size=3, max_len=max_len, length_penalty=1.0
    )
    assert len(hyps) == 3
    assert all(len(h.tokens) == max_len and not h.finished for h in hyps)
    assert len(calls) == max_len


# --- scores, ties, masks ------------------------------------------------------------------------


def test_length_penalty_changes_the_winner():
    # WHY: summed log-probability favours short outputs, since every token
    #      costs probability. "a eos" has P = 0.3 and "b b b eos" has
    #      P = 0.7 * 0.9 * 0.9 * 0.5 = 0.2835: with length_penalty 0 the short
    #      one wins, with 1 (mean per token) the long one does. The returned
    #      list is ranked by score, not by logprob.
    # KIND: unit
    # CATCHES: s01, s03, s04, s06, s07, s11, m01, m02, m03
    # CHAPTER: L4.4 section 2.4, Length normalization
    lm = TableLM(
        {
            (): [0.0, 0.3, 0.7],
            (A,): [1.0, 0.0, 0.0],
            (B,): [0.0, 0.1, 0.9],
            (B, B): [0.0, 0.1, 0.9],
            (B, B, B): [0.5, 0.25, 0.25],
            (B, A): [1.0, 0.0, 0.0],
            (B, B, A): [1.0, 0.0, 0.0],
        },
        V=3,
    )
    for alpha, first in ((0.0, [A, EOS]), (1.0, [B, B, B, EOS])):
        step, init, _ = make_step(lm)
        hyps = beam_search(
            step, init, BOS, EOS, beam_size=2, max_len=4, length_penalty=alpha
        )
        assert hyps[0].tokens == first, f"length_penalty {alpha}"
        assert [h.score for h in hyps] == sorted((h.score for h in hyps), reverse=True)
        for h in hyps:
            assert_close(h.score, h.logprob / len(h.tokens) ** alpha, dtype="float64")


def test_ties_go_to_the_lowest_id():
    # WHY: with equal scores the order must not depend on the sort routine:
    #      the lower flat index (lower beam, then lower token id) wins, as in
    #      the Python and Rust samplers. Uniform logits make every candidate
    #      tie: beam 2 over V = 4 with eos = 3 keeps tokens 0 and 1 first.
    # KIND: boundary
    # CATCHES: s03, s05, s08, s11, m03
    # CHAPTER: L4.4 section 2.3, The algorithm
    lm = TableLM({}, V=4)
    step, init, _ = make_step(lm)
    hyps = beam_search(step, init, BOS, 3, beam_size=2, max_len=2, length_penalty=0.0)
    assert [h.tokens for h in hyps] == [[0, 0], [0, 1]]


def test_logits_need_not_be_normalized():
    # WHY: models return logits, not log-probabilities. Adding a different
    #      constant to every row must not change the search: each row is
    #      log-softmaxed before it is added to the running total.
    # KIND: property
    # CATCHES: s02, s03, m03
    # CHAPTER: L4.4 section 5, Pitfalls, item 2
    lm = ToyLM(4, 11, eos_bias=0.5)
    step, init, _ = make_step(lm)

    def shifted(state, y_prev):
        z, s = step(state, y_prev)
        return z + 10.0 * np.arange(1, z.shape[0] + 1)[:, None] + 3.0 * len(
            s["prefix"][0]
        ), s

    want = beam_search(step, init, BOS, EOS, beam_size=3, max_len=5)
    step, init, _ = make_step(lm)
    got = beam_search(shifted, init, BOS, EOS, beam_size=3, max_len=5)
    assert [h.tokens for h in got] == [h.tokens for h in want]
    assert_close(
        [h.logprob for h in got], [h.logprob for h in want], rtol=1e-12, atol=1e-12
    )


def test_masked_tokens_are_never_chosen():
    # WHY: -inf marks a token that may not follow (a grammar, a stop list).
    #      Such a candidate is never kept, even when fewer than beam_size
    #      candidates are finite: then the beam is narrower. Here only token 1
    #      is allowed at step 1 and only eos after it.
    # KIND: boundary
    # CATCHES: s04, s09, s16, m01, m03
    # CHAPTER: L4.4 section 5, Pitfalls, item 5
    lm = TableLM({(): [0.0, 1.0, 0.0, 0.0], (1,): [1.0, 0.0, 0.0, 0.0]}, V=4)
    step, init, calls = make_step(lm)
    hyps = beam_search(step, init, BOS, EOS, beam_size=3, max_len=4, length_penalty=1.0)
    assert [h.tokens for h in hyps] == [[1, EOS]]
    assert calls == [1, 1]
    assert np.isfinite(hyps[0].logprob)


def test_logprob_is_the_sum_along_the_path():
    # WHY: the logprob a hypothesis reports is exactly the sum of
    #      log p(token | its own prefix), so a caller can rescore or compare
    #      hypotheses from different searches.
    # KIND: property
    # CATCHES: s02, s03, s06, m01, m03, m04
    # CHAPTER: L4.4 section 4, The interface
    lm = ToyLM(5, 21, eos_bias=0.0)
    step, init, _ = make_step(lm)
    for h in beam_search(
        step, init, BOS, EOS, beam_size=4, max_len=6, length_penalty=0.7
    ):
        assert_close(h.logprob, seq_logprob(lm, h.tokens), rtol=1e-12, atol=1e-12)
        assert_close(h.score, h.logprob / len(h.tokens) ** 0.7, rtol=1e-12, atol=1e-12)


# --- select_state ----------------------------------------------------------------------------------


class Pair(NamedTuple):
    h: np.ndarray
    c: np.ndarray


def test_select_state_keeps_the_structure():
    # WHY: a decoder state is whatever the model needs (an LSTM's (h, c)
    #      NamedTuple, a dict with the encoder outputs, a list per layer).
    #      select_state gathers the parents' rows of every array, repeats
    #      included, keeps every container type, and leaves non-arrays alone.
    # KIND: unit
    # CATCHES: s13, s14
    # CHAPTER: L4.4 section 4, The interface
    h = np.arange(6.0).reshape(3, 2)
    state = {
        "rnn": Pair(h, 10 * h),
        "layers": [h + 1, (h + 2, None)],
        "memory": np.arange(3 * 4).reshape(3, 4),
        "step": 5,
        "scalar": np.float64(2.0),
    }
    got = select_state(state, [2, 2, 0])
    assert set(got) == set(state)
    assert type(got["rnn"]) is Pair
    assert np.array_equal(got["rnn"].h, h[[2, 2, 0]])
    assert np.array_equal(got["rnn"].c, 10 * h[[2, 2, 0]])
    assert isinstance(got["layers"], list) and isinstance(got["layers"][1], tuple)
    assert np.array_equal(got["layers"][0], (h + 1)[[2, 2, 0]])
    assert np.array_equal(got["layers"][1][0], (h + 2)[[2, 2, 0]])
    assert got["layers"][1][1] is None
    assert np.array_equal(got["memory"], state["memory"][[2, 2, 0]])
    assert got["step"] == 5 and got["scalar"] == 2.0
    with pytest.raises(TypeError):
        select_state({1, 2}, [0])


def test_state_follows_parents_in_a_real_search():
    # WHY: when two children of one parent survive and another parent dies,
    #      the state rows must move with them. The step function here checks,
    #      at every call, that row i's stored prefix plus y_prev[i] is a
    #      prefix the search really holds (its logits are what was scored).
    # KIND: property
    # CATCHES: s03, m01, m03
    # CHAPTER: L4.4 section 5, Pitfalls, item 6
    lm = ToyLM(4, 5, eos_bias=-1.0)
    seen: set = {()}
    step, init, _ = make_step(lm)

    def checked(state, y_prev):
        z, s = step(state, y_prev)
        for row in s["prefix"]:
            t = tuple(int(x) for x in row)
            assert t[:-1] in seen, (
                f"state row {t} does not extend a hypothesis the search kept"
            )
            seen.add(t)
        return z, s

    beam_search(checked, init, BOS, EOS, beam_size=3, max_len=6)
    assert len(seen) > 4


# --- validation ----------------------------------------------------------------------------------------


def test_validation():
    # WHY: a beam of 0 or a length of 0 is a caller bug, and logits that are
    #      not one row per live hypothesis mean the model and the search
    #      disagree about the state; all fail loudly. Greedy with nothing
    #      allowed at the first step has no answer.
    # KIND: boundary
    # CATCHES: s09, m05
    # CHAPTER: L4.4 section 4, The interface
    step, init, _ = make_step(hand_lm())
    for kw in ({"beam_size": 0, "max_len": 3}, {"beam_size": 2, "max_len": 0}):
        with pytest.raises(ValueError):
            beam_search(step, init, BOS, EOS, **kw)
    with pytest.raises(ValueError):
        beam_search(step, init, BOS, EOS, beam_size=2, max_len=3, length_penalty=-1.0)

    def bad(state, y_prev):
        return np.zeros((len(y_prev) + 1, 3)), state

    with pytest.raises(ValueError):
        beam_search(bad, None, BOS, EOS, beam_size=2, max_len=3)

    def nothing(state, y_prev):
        return np.full((len(y_prev), 3), -np.inf), state

    assert beam_search(nothing, None, BOS, EOS, beam_size=2, max_len=3) == []
    with pytest.raises(ValueError):
        greedy_decode(nothing, None, BOS, EOS, max_len=3)


# --- a real decoder --------------------------------------------------------------------------------------


def test_beam_decodes_a_seq2seq():
    # WHY: the call site. Beam search drives L4.1's Seq2Seq one decode_step
    #      at a time, gathering its DecoderState (a NamedTuple of Tensors and
    #      a mask) by parent with select_state. A beam of 1 must equal the
    #      model's own greedy decoding, and every hypothesis of a beam of 4
    #      must have the logprob that teacher-forced rescoring of its tokens
    #      gives: a state row that followed the wrong parent fails this.
    # KIND: differential
    # CATCHES: s02, s03, s05, s11, s13, m01, m03, m04
    # CHAPTER: L4.4 section 6, Where it's used next
    from tinyllm.seq2seq.luong import LuongAttention
    from tinyllm.seq2seq.model import Seq2Seq

    class R:
        def __init__(self, s):
            self.g = PCG32(seed=s)

        def uniform(self):
            return self.g.uniform()

        def next_u32(self):
            return self.g.next_u32()

    m = Seq2Seq(
        7,
        6,
        4,
        6,
        cell="lstm",
        attention=LuongAttention(6, "general", rng=R(1)),
        rng=R(2),
    )
    src, lens = np.array([[3, 4, 5, 6, 2]]), np.array([4])
    bos, eos = 1, 0

    def step(state, y):
        logits, new, _ = m.decode_step(y, state)
        return logits, new

    init = m.init_state(m.encode(src, lens))
    one = beam_search(step, init, bos, eos, beam_size=1, max_len=6, length_penalty=0.0)
    hyps = beam_search(step, init, bos, eos, beam_size=4, max_len=6, length_penalty=0.0)
    assert one[0].tokens == m.greedy(src, lens, bos, eos, max_len=6)[0]
    assert len(hyps) == 4
    for h in hyps:
        tgt_in = np.array([[bos] + h.tokens[:-1]])
        ls = logsoftmax(m(src, lens, tgt_in).data[0].astype(np.float64))
        assert_close(
            h.logprob,
            float(ls[np.arange(len(h.tokens)), h.tokens].sum()),
            rtol=1e-5,
            atol=1e-5,
        )
