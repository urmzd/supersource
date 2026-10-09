"""Course tests for L6.7: the evaluation harness and the model zoo
(tinyllm/eval/lm.py, tinyllm/eval/zoo.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L6.7), and the chapter section it comes
from.

The worked examples of the chapter (section 3):
  * strided windows: a model that gives the true next token probability
    1 / (j + 2) at window position j (it has read the j + 1 tokens up to there), 6
    tokens, ctx_len 4. Stride 2: windows [0, 4) and [2, 6); per-token NLLs
    ln 2, ln 3, ln 4 (first window), ln 3, ln 4 (second): mean
    (ln 2 + 2 ln 3 + 2 ln 4) / 5, perplexity 288^(1/5) = 3.1037. Stride 3:
    windows [0, 4) and [3, 6): ln 2, ln 3, ln 4, ln 2, ln 3.
  * multiple choice: context "a", choices "bb" and "c", with p(b | a) = 0.4,
    p(b | b) = 0.9, p(c | a) = 0.45. Sums: ln 0.4 + ln 0.9 = -1.0217 for "bb",
    ln 0.45 = -0.7985 for "c": acc picks "c". Per byte: -0.5108 against
    -0.7985: acc_norm picks "bb".
  * the zoo: a bigram that is uniform over 256 bytes pays ln 256 nats per
    byte, exactly 8 bits per byte, with a zero-width interval.

The golden fixture (course/fixtures/L6.7/strided_ppl_hf.npz) holds a tiny
random Hugging Face GPT-2 (transformers 5.19.0) and its per-token NLLs under
the strided definition (course/oracle/L6.7/strided_ppl_hf.py); L6.1's
load_hf_gpt2 loads it.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from schema_lite import errors
from tinyllm.eval.lm import (
    ByteTokenizer,
    Result,
    compare,
    eval_ppl,
    run_task,
    score_choices,
    token_nlls,
)
from tinyllm.eval.zoo import (
    encode_batch,
    load_model,
    run_zoo,
    save_word2vec,
    write_report,
)
from tinyllm.infer.beam import beam_search
from tinyllm.io.safetensors import save_safetensors
from tinyllm.lm.ngram import NGramLM
from tinyllm.lm.nplm import NPLM, save_nplm
from tinyllm.lm.word2vec import word_similarity
from tinyllm.obj.bert import BertConfig, BertEncoder, BertForMLM, save_bert
from tinyllm.obj.electra import ELECTRA, save_electra
from tinyllm.obj.gpt import GPT, GPTConfig, load_hf_gpt2, save_gpt
from tinyllm.obj.heads import SequenceClassifier, predict, save_classifier
from tinyllm.prob.stats import bootstrap_ci, mean_ci, wilson_interval
from tinyllm.prob.tests import paired_permutation_test
from tinyllm.rnn.rnnlm import RNNLM, save_rnnlm
from tinyllm.seq2seq.model import Seq2Seq, save_seq2seq
from tinyllm.xfmr.transformer import (
    Transformer,
    TransformerConfig,
    save_transformer,
    translate,
)

FIX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "formats"
LN2 = math.log(2.0)


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


class ContextCounter:
    """p(the true next token) = 1 / (j + 2) at window position j: the NLL
    reveals how much context the token was scored with."""

    def __init__(self, ids, V=8) -> None:
        self.ids, self.V, self.calls = np.asarray(ids), V, []

    def __call__(self, x):
        x = np.asarray(x)[0]
        self.calls.append(x.tolist())
        T = x.size
        z = np.full((T, self.V), -30.0)
        for j in range(T):
            p = 1.0 / (j + 2)
            nxt = (
                self.ids[self._start(x) + j + 1]
                if self._start(x) + j + 1 < len(self.ids)
                else 0
            )
            z[j, :] = math.log((1 - p) / (self.V - 1))
            z[j, nxt] = math.log(p)
        return z[None]

    def _start(self, x):
        n = x.size
        for s in range(len(self.ids) - n + 1):
            if np.array_equal(self.ids[s : s + n], x):
                return s
        raise AssertionError("window not found")


class TableLM:
    """A byte bigram from a table of next-byte probabilities."""

    def __init__(self, table) -> None:
        P = np.full((256, 256), 1.0)
        for a, row in table.items():
            rest = 1.0 - sum(row.values())
            P[ord(a)] = rest / (256 - len(row))
            for b, p in row.items():
                P[ord(a), ord(b)] = p
        self.L = np.log(P / P.sum(axis=1, keepdims=True))

    def __call__(self, x):
        return self.L[np.asarray(x)[0]][None]


TABLE = {"a": {"b": 0.4, "c": 0.45}, "b": {"b": 0.9}}


# --- the worked examples ---------------------------------------------------------------


def test_hand_example_strided_windows():
    # WHY: the chapter's worked example: which window scores which token,
    #      and with how much context. Each token is scored once, by the first
    #      window that reaches it.
    # KIND: unit
    # CATCHES: s01, m01
    # CHAPTER: L6.7 section 3, Worked example by hand
    ids = np.array([3, 1, 4, 1, 5, 2])
    m = ContextCounter(ids)
    nll = token_nlls(m, ids, ctx_len=4, stride=2)
    assert m.calls == [[3, 1, 4, 1], [4, 1, 5, 2]]
    assert_close(nll, np.log([2, 3, 4, 3, 4]), dtype="float64")
    r = eval_ppl(ContextCounter(ids), ids, 4, 2)
    assert r["n_tokens"] == 5
    assert_close(r["ppl"], 288**0.2, dtype="float64")
    assert_close(
        token_nlls(ContextCounter(ids), ids, 4, 3),
        np.log([2, 3, 4, 2, 3]),
        dtype="float64",
    )


def test_hand_example_choices():
    # WHY: the chapter's multiple-choice example: summed log-likelihood
    #      prefers the short choice, per-byte normalization the long one.
    # KIND: unit
    # CATCHES: s03, s04
    # CHAPTER: L6.7 section 3, Worked example by hand
    m, tok = TableLM(TABLE), ByteTokenizer()
    raw = score_choices(m, tok, "a", ["bb", "c"], normalize=False)
    assert_close(
        raw, [math.log(0.4) + math.log(0.9), math.log(0.45)], rtol=1e-12, atol=1e-12
    )
    norm = score_choices(m, tok, "a", ["bb", "c"], normalize=True)
    assert_close(
        norm,
        [(math.log(0.4) + math.log(0.9)) / 2, math.log(0.45)],
        rtol=1e-12,
        atol=1e-12,
    )
    assert int(np.argmax(raw)) == 1 and int(np.argmax(norm)) == 0


def test_hand_example_uniform_bigram_is_eight_bits(tmp_path):
    # WHY: the zoo's unit: a model that knows nothing about bytes pays
    #      exactly 8 bits per byte; bpb is the number every language model
    #      row is compared by.
    # KIND: unit
    # CATCHES: s05, m02
    # CHAPTER: L6.7 section 3, Worked example by hand
    d = tmp_path / "uni"
    d.mkdir()
    save_safetensors(
        str(d / "model.safetensors"),
        {"bigram.weight": np.zeros((256, 256), dtype=np.float32)},
        {"format": "tinyllm"},
    )
    (d / "config.json").write_text(
        json.dumps({"tl_arch": "bigram", "tl_tokenizer": "bytes", "vocab_size": 256})
    )
    write_tokens(tmp_path / "t.bin", list(b"hello"))
    (tmp_path / "zoo.json").write_text(
        json.dumps(
            {"models": [{"id": "u", "dir": "uni", "task": "t", "data": "t.bin"}]}
        )
    )
    row = run_zoo(str(tmp_path / "zoo.json"), Rng(0))["rows"][0]
    assert (
        row["status"] == "ok"
        and row["metric"] == "bpb"
        and row["higher_is_better"] is False
    )
    assert_close(row["value"], 8.0, dtype="float64")
    assert_close(row["ci95"], [8.0, 8.0], rtol=1e-12, atol=1e-9)
    assert row["n"] == 4 and row["tl_arch"] == "bigram"


# --- strided perplexity --------------------------------------------------------------------


@pytest.mark.parametrize("ctx,stride", [(8, 4), (8, 7), (16, 15), (16, 1)])
def test_strided_nll_matches_hf(ctx, stride):
    # WHY: the same strided definition computed with Hugging Face's GPT-2:
    #      per-token NLLs equal after L6.1 loads the weights. Strides 7 and
    #      15 are the cheapest valid ones; stride 1 gives full context.
    # KIND: golden
    # CATCHES: s01
    # CHAPTER: L6.7 section 2.1, Strided perplexity
    f = np.load(FIX / "L6.7" / "strided_ppl_hf.npz")
    model = GPT(
        GPTConfig(vocab=50, n_ctx=16, d_model=16, n_heads=2, n_layers=2, d_ff=64),
        Rng(0),
    )
    load_hf_gpt2(
        model, {k[len("param.") :]: f[k] for k in f.files if k.startswith("param.")}
    )
    model.eval()
    nll = token_nlls(model, f["ids"], ctx, stride)
    assert_close(nll, f[f"nll.{ctx}.{stride}"], rtol=1e-5, atol=1e-5)
    r = eval_ppl(model, f["ids"], ctx, stride)
    assert_close(r["nll_mean"], f[f"mean.{ctx}.{stride}"], rtol=1e-5, atol=1e-6)


def test_every_token_is_scored_once():
    # WHY: over random lengths, windows, and strides every token 1..n-1 is
    #      scored exactly once and sees at least ctx_len - stride tokens of
    #      context (fewer only near the start). Stride >= ctx_len is refused:
    #      it would skip each window's first token.
    # KIND: property
    # CATCHES: s01, s02
    # CHAPTER: L6.7 section 2.1, Strided perplexity
    g = PCG32(seed=5)
    for _ in range(25):
        n, ctx = 2 + g.below(40), 2 + g.below(10)
        stride = 1 + g.below(ctx - 1)
        ids = np.arange(
            n
        )  # distinct ids: the counter can tell where each window starts
        nll = token_nlls(ContextCounter(ids, V=n + 1), ids, ctx, stride)
        assert nll.shape == (n - 1,)
        c = np.exp(nll) - 1  # tokens of context each token was scored with (j + 1)
        t = np.arange(1, n)
        assert np.all(c >= np.minimum(t, ctx - stride) - 1e-9)
        assert np.all(c <= np.minimum(t, ctx - 1) + 1e-9)
    for bad in [(4, 4), (4, 0), (4, 5)]:
        with pytest.raises(ValueError):
            token_nlls(ContextCounter(np.arange(6)), np.arange(6), *bad)


def test_eval_ppl_bpb_and_interval():
    # WHY: the totals go through M11.2's accumulator (bpb from the byte
    #      count) and the interval is M07.4's Student t interval of the mean
    #      NLL, exponentiated for perplexity and rescaled for bits per byte.
    # KIND: differential
    # CATCHES: s06
    # CHAPTER: L6.7 section 2.3, Every metric with an interval
    ids = np.array([3, 1, 4, 1, 5, 2, 6, 5, 3, 5])
    r = eval_ppl(ContextCounter(ids), ids, 5, 2, n_bytes=20)
    nll = token_nlls(ContextCounter(ids), ids, 5, 2)
    m, lo, hi = mean_ci(nll)
    assert_close(r["nll_mean"], m, dtype="float64")
    assert_close(r["bpb"], nll.sum() / (20 * LN2), dtype="float64")
    assert_close(
        [r["ppl_lo"], r["ppl_hi"]], [math.exp(lo), math.exp(hi)], dtype="float64"
    )
    assert_close(
        [r["bpb_lo"], r["bpb_hi"]],
        [lo * 9 / (20 * LN2), hi * 9 / (20 * LN2)],
        dtype="float64",
    )
    assert "bpb" not in eval_ppl(ContextCounter(ids), ids, 5, 2)


# --- tasks and comparisons --------------------------------------------------------------------


def test_run_task_accuracy_with_bootstrap(tmp_path):
    # WHY: a task score is a mean over items with M07.4's bootstrap interval
    #      drawn from the given rng; acc and acc_norm differ exactly where
    #      the chapter says, and ties go to the first choice.
    # KIND: differential
    # CATCHES: s03, s04, s07
    # CHAPTER: L6.7 section 2.2, Multiple choice by log-likelihood
    items = [
        {"context": "a", "choices": ["bb", "c"], "label": 0},
        {"context": "a", "choices": ["c", "bb"], "label": 0},
        {"context": "b", "choices": ["b", "b"], "label": 0},
        {"context": "a", "choices": ["b", "c"], "label": 1},
    ]
    task = tmp_path / "t.jsonl"
    task.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    m, tok = TableLM(TABLE), ByteTokenizer()
    r = run_task(m, tok, str(task), "acc", Rng(3), n_boot=200)
    assert r.per_item.tolist() == [0.0, 1.0, 1.0, 1.0]
    want = bootstrap_ci(np.array([0.0, 1.0, 1.0, 1.0]), np.mean, 200, 0.05, Rng(3))
    assert_close([r.value, r.lo, r.hi], list(want), dtype="float64")
    assert r.n == 4
    rn = run_task(m, tok, str(task), "acc_norm", Rng(3), n_boot=50)
    assert rn.per_item.tolist() == [1.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        run_task(m, tok, str(task), "f1", Rng(3))


def test_compare_is_the_paired_permutation_test():
    # WHY: "is B better than A on THESE items" is a paired question; the
    #      comparison is M07.5's sign-flip test on the per-item scores, from
    #      the given rng, and refuses results over different items.
    # KIND: differential
    # CATCHES: s08
    # CHAPTER: L6.7 section 2.4, Paired comparison
    g = PCG32(seed=7)
    a = np.array([float(g.uniform() < 0.6) for _ in range(40)])
    b = np.array([float(g.uniform() < 0.8) for _ in range(40)])
    A, B = Result(a.mean(), 0, 1, 40, a), Result(b.mean(), 0, 1, 40, b)
    p = compare(A, B, 500, Rng(11))
    assert p == paired_permutation_test(a, b, 500, Rng(11))
    assert compare(A, A, 100, Rng(1)) == 1.0
    with pytest.raises(ValueError):
        compare(A, Result(0.5, 0, 1, 3, np.zeros(3)), 10, Rng(1))


# --- the zoo -----------------------------------------------------------------------------------


def write_tokens(path: Path, ids) -> None:
    header = [20240520, 1, len(ids), 256] + [0] * 252
    path.write_bytes(
        struct.pack("<256i", *header) + np.asarray(ids, dtype="<u2").tobytes()
    )


def sentences():
    return [
        ("val", "the film is good", 1),
        ("val", "the film is bad", 0),
        ("train", "x", 1),
        ("val", "i loved the cast", 1),
        ("val", "the plot is dull", 0),
    ]


def bert_cfg(V=256, d=8, layers=1) -> BertConfig:
    return BertConfig(
        vocab=V, max_len=24, d_model=d, n_heads=2, n_layers=layers, d_ff=16, dropout=0.0
    )


@pytest.fixture(scope="module")
def zoo(tmp_path_factory):
    """Every family, saved by its own module and listed in one manifest."""
    root = tmp_path_factory.mktemp("zoo")
    text = b"once upon a time a cat sat on a mat. the end. " * 3
    write_tokens(root / "val.bin", list(text))
    (root / "text.tsv").write_text(
        "split\tsentence\tlabel\n"
        + "".join(f"{s}\t{t}\t{l}\n" for s, t, l in sentences())
    )
    # The seq2seq model is untrained except for a huge eos bias: it answers
    # eos at once, so item 0 (an empty target) is an exact match.
    items = [{"src": [3, 4, 5], "tgt": []}, {"src": [6, 3], "tgt": [3, 6]}]
    (root / "rev.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (root / "sim.tsv").write_text(
        "cat\tdog\t3\ncat\tcake\t0\ndog\tfox\t3\ncake\tpie\t3\nfox\tpie\t0\n"
    )
    W = PCG32(seed=1).normal_array((256, 256)).astype(np.float32)
    (root / "bigram").mkdir()
    save_safetensors(
        str(root / "bigram" / "model.safetensors"),
        {"bigram.weight": W},
        {"format": "tinyllm"},
    )
    (root / "bigram" / "config.json").write_text(
        json.dumps({"tl_arch": "bigram", "tl_tokenizer": "bytes", "vocab_size": 256})
    )
    ng = NGramLM(3, vocab_size=256)
    ng.fit([list(text[:60])])
    ng.save(str(root / "kn3.safetensors"))
    save_nplm(NPLM(256, 3, 4, 8, rng=Rng(2)), str(root / "nplm"))
    save_rnnlm(RNNLM(256, 4, 8, "gru", rng=Rng(3)), str(root / "rnnlm"))
    save_gpt(
        GPT(
            GPTConfig(vocab=256, n_ctx=16, d_model=8, n_heads=2, n_layers=1, d_ff=16),
            Rng(4),
        ),
        str(root / "gpt"),
    )
    s2s = Seq2Seq(8, 8, 4, 6, rng=Rng(5))
    s2s.out.bias.data[2] = 50.0
    save_seq2seq(s2s, str(root / "s2s"))
    save_transformer(
        Transformer(
            TransformerConfig(
                src_vocab=8,
                tgt_vocab=8,
                d_model=8,
                n_heads=2,
                d_ff=16,
                n_enc=1,
                n_dec=1,
                dropout=0.0,
                max_len=16,
            ),
            Rng(6),
        ),
        str(root / "xfmr"),
    )
    save_bert(BertForMLM(bert_cfg(), Rng(7)), str(root / "bert"), tokenizer="bytes")
    save_electra(
        ELECTRA(bert_cfg(), bert_cfg(layers=2), Rng(8)),
        str(root / "electra"),
        mask_id=3,
        special_ids=[0, 1, 2, 3],
        tokenizer="bytes",
    )
    save_classifier(
        SequenceClassifier(BertEncoder(bert_cfg(), Rng(9)), 8, 2, rng=Rng(10)),
        str(root / "clf"),
        "bert",
        tokenizer="bytes",
    )
    words = ["cat", "dog", "fox", "cake", "pie"]
    E = np.array(
        [[1, 0.1, 0], [0.9, 0.2, 0], [0.8, 0.3, 0.1], [0, 0.2, 1], [0.1, 0.1, 0.9]]
    )
    save_word2vec(E, words, str(root / "w2v"))
    lm = lambda i, d: {"id": i, "dir": d, "task": "val", "data": "val.bin"}  # noqa: E731
    models = [
        lm("bigram", "bigram"),
        lm("kn3", "kn3.safetensors"),
        lm("nplm", "nplm"),
        lm("rnnlm", "rnnlm"),
        dict(lm("gpt", "gpt"), ctx_len=16, stride=5),
        {
            "id": "s2s",
            "dir": "s2s",
            "task": "rev",
            "data": "rev.jsonl",
            "bos": 1,
            "eos": 2,
            "beam": 3,
            "max_len": 5,
        },
        {
            "id": "xfmr",
            "dir": "xfmr",
            "task": "rev",
            "data": "rev.jsonl",
            "bos": 1,
            "eos": 2,
            "beam": 1,
            "max_len": 5,
        },
        {"id": "bert", "dir": "bert", "task": "text", "data": "text.tsv", "p": 0.5},
        {
            "id": "electra",
            "dir": "electra",
            "task": "text",
            "data": "text.tsv",
            "p": 0.5,
        },
        {"id": "clf", "dir": "clf", "task": "text", "data": "text.tsv"},
        {"id": "w2v", "dir": "w2v", "task": "sim", "data": "sim.tsv"},
    ]
    (root / "zoo.json").write_text(json.dumps({"suite": "zoo", "models": models}))
    report = run_zoo(str(root / "zoo.json"), Rng(12), seed=12)
    return root, report, text


def test_zoo_language_model_rows_are_each_familys_own_nll(zoo):
    # WHY: the zoo adds no modelling of its own: each language model row is
    #      the family's own per-token NLL in bits per byte (one byte per
    #      token), with M07.4's interval, so the rows compare across families.
    # KIND: differential
    # CATCHES: s05, s09, m02, m03
    # CHAPTER: L6.7 section 2.5, The model zoo
    root, report, text = zoo
    rows = {r["model"]: r for r in report["rows"]}
    ids = np.frombuffer(text, dtype=np.uint8).astype(np.int64)
    W = np.asarray(load_model(str(root / "bigram")).weight, dtype=np.float64)
    lsm = W - W.max(1, keepdims=True)
    lsm = lsm - np.log(np.exp(lsm).sum(1, keepdims=True))
    want = {
        "bigram": -lsm[ids[:-1], ids[1:]],
        "kn3": np.asarray(
            NGramLM.load(str(root / "kn3.safetensors")).nll(ids.tolist())
        ),
        "nplm": np.asarray(load_model(str(root / "nplm")).nll(ids)),
        "rnnlm": np.asarray(load_model(str(root / "rnnlm")).nll(ids)),
        "gpt": token_nlls(load_model(str(root / "gpt")), ids, 16, 5),
    }
    for name, nll in want.items():
        r = rows[name]
        assert r["status"] == "ok", r
        assert r["metric"] == "bpb" and r["n"] == nll.size
        assert_close(
            r["value"], nll.sum() / (nll.size * LN2), rtol=1e-6, atol=1e-9, msg=name
        )
        _, lo, hi = mean_ci(nll)
        assert_close(r["ci95"], [lo / LN2, hi / LN2], rtol=1e-6, atol=1e-9, msg=name)
        assert r["ci95"][0] <= r["value"] <= r["ci95"][1]
    assert "tl_arch" not in rows["kn3"] and rows["gpt"]["tl_arch"] == "gpt"
    assert rows["gpt"]["params"] == sum(
        p.data.size for p in load_model(str(root / "gpt")).parameters()
    )


def test_zoo_seq2seq_rows_decode_with_beam(zoo):
    # WHY: sequence-to-sequence rows are exact match of the best beam
    #      hypothesis without its eos, decoded through L4.4 (seq2seq) or
    #      L5.5's translate, with a Wilson interval.
    # KIND: differential
    # CATCHES: s10, m04
    # CHAPTER: L6.7 section 2.5, The model zoo
    root, report, _ = zoo
    rows = {r["model"]: r for r in report["rows"]}
    items = [json.loads(x) for x in (root / "rev.jsonl").read_text().splitlines()]
    m = load_model(str(root / "s2s"))
    k = 0
    for it in items:
        src = np.array([it["src"]])

        def step(state, y):
            logits, new, _ = m.decode_step(y, state)
            return logits.data, new

        best = beam_search(
            step, m.init_state(m.encode(src, np.array([src.shape[1]]))), 1, 2, 3, 5
        )[0]
        out = best.tokens[:-1] if best.finished else best.tokens
        k += out == it["tgt"]
    r = rows["s2s"]
    assert k == 1  # the eos-at-once answer matches the empty target
    assert (
        r["status"] == "ok"
        and r["metric"] == "em"
        and r["decode"] == "beam"
        and r["beam"] == 3
    )
    assert r["value"] == k / 2 and r["ci95"] == list(wilson_interval(k, 2))
    x = load_model(str(root / "xfmr"))
    kx = sum(
        translate(x, np.array([it["src"]]), 1, 2, 0, 5)[0] == it["tgt"] for it in items
    )
    assert rows["xfmr"]["value"] == kx / 2 and rows["xfmr"]["decode"] == "greedy"


def test_zoo_accuracy_rows(zoo):
    # WHY: classifiers (L6.5) report accuracy on the validation split, BERT
    #      its masked-token accuracy, ELECTRA its replaced-token detection
    #      accuracy (L6.3); each with a Wilson interval over its items.
    # KIND: unit
    # CATCHES: s11, m05
    # CHAPTER: L6.7 section 2.5, The model zoo
    root, report, _ = zoo
    rows = {r["model"]: r for r in report["rows"]}
    for name in ("clf", "bert", "electra"):
        r = rows[name]
        assert r["status"] == "ok", r
        assert r["metric"] == "accuracy" and r["higher_is_better"] is True
        k = round(r["value"] * r["n"])
        assert r["ci95"] == list(wilson_interval(k, r["n"]))
    assert rows["clf"]["n"] == 4  # the "val" rows only
    clf = load_model(str(root / "clf"))
    val = [(t, l) for s, t, l in sentences() if s == "val"]
    ids, real = encode_batch([t for t, _ in val], 24)
    want = np.mean(predict(clf, ids, real) == np.array([l for _, l in val]))
    assert rows["clf"]["value"] == want
    ids, real = encode_batch(["ab", "c"], 24)
    assert ids.tolist() == [[1, 97, 98, 2], [1, 99, 2, 0]] and real.tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 0],
    ]
    lens = [len(t.encode()) + 2 for s, t, _ in sentences() if s == "val"]
    assert rows["electra"]["n"] == sum(lens)


def test_zoo_word_vectors_row(zoo):
    # WHY: word2vec rows are L2.3's word_similarity Spearman with a Fisher z
    #      interval, loaded from the zoo's own word2vec directory format.
    # KIND: unit
    # CATCHES: s12
    # CHAPTER: L6.7 section 2.5, The model zoo
    root, report, _ = zoo
    r = {x["model"]: x for x in report["rows"]}["w2v"]
    emb, vocab = load_model(str(root / "w2v"))
    pairs = [
        (a, b, float(s))
        for a, b, s in (
            l.split("\t") for l in (root / "sim.tsv").read_text().splitlines()
        )
    ]
    rho = word_similarity(emb, vocab, pairs)
    assert r["metric"] == "spearman" and r["n"] == 5
    assert_close(r["value"], rho, dtype="float64")
    z, se = math.atanh(rho), 1 / math.sqrt(2)
    assert_close(
        r["ci95"],
        [math.tanh(z - 1.959963984540054 * se), math.tanh(z + 1.959963984540054 * se)],
        rtol=1e-9,
        atol=1e-12,
    )


def test_zoo_report_validates_against_the_schema(zoo, tmp_path):
    # WHY: EvalSuite (dur.11) and the capstone table read the report by its
    #      schema, formats/eval-results.schema.json: one row per manifest
    #      entry, in order, each with an interval around its value.
    # KIND: conformance
    # CATCHES: m06
    # CHAPTER: L6.7 section 4, The interface
    root, report, _ = zoo
    schema = json.loads((CONTRACTS / "eval-results.schema.json").read_text())
    assert errors(report, schema) == []
    assert (
        report["format"] == "tl.eval-results.v1"
        and report["seed"] == 12
        and report["suite"] == "zoo"
    )
    names = [m["id"] for m in json.loads((root / "zoo.json").read_text())["models"]]
    assert [r["model"] for r in report["rows"]] == names
    write_report(report, str(tmp_path / "out" / "zoo.json"))
    assert json.loads((tmp_path / "out" / "zoo.json").read_text()) == report


def test_zoo_reports_failures_as_rows(tmp_path):
    # WHY: one broken or unknown family must not hide the others: a missing
    #      directory is an "error" row with the reason, an unknown tl_arch a
    #      "skipped" row, and the healthy entry is still scored.
    # KIND: fault
    # CATCHES: m07
    # CHAPTER: L6.7 section 5, Pitfalls
    write_tokens(tmp_path / "t.bin", list(b"abcabc"))
    (tmp_path / "llama").mkdir()
    (tmp_path / "llama" / "config.json").write_text(json.dumps({"tl_arch": "llama"}))
    (tmp_path / "ok").mkdir()
    save_safetensors(
        str(tmp_path / "ok" / "model.safetensors"),
        {"bigram.weight": np.zeros((256, 256), dtype=np.float32)},
        {"format": "tinyllm"},
    )
    (tmp_path / "ok" / "config.json").write_text(
        json.dumps({"tl_arch": "bigram", "tl_tokenizer": "bytes"})
    )
    models = [
        {"id": "gone", "dir": "nope", "task": "t", "data": "t.bin"},
        {"id": "llama", "dir": "llama", "task": "t", "data": "t.bin"},
        {"id": "ok", "dir": "ok", "task": "t", "data": "t.bin"},
    ]
    (tmp_path / "zoo.json").write_text(json.dumps({"models": models}))
    rows = run_zoo(str(tmp_path / "zoo.json"), Rng(0))["rows"]
    assert [r["status"] for r in rows] == ["error", "skipped", "ok"]
    assert rows[0]["value"] is None and "nope" in rows[0]["reason"]
    assert "llama" in rows[1]["reason"]
    schema = json.loads((CONTRACTS / "eval-results.schema.json").read_text())
    assert (
        errors(
            {"format": "tl.eval-results.v1", "suite": "zoo", "seed": 0, "rows": rows},
            schema,
        )
        == []
    )
