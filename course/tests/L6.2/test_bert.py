"""Course tests for L6.2: BERT and masked-LM masking (tinyllm/obj/bert.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L6.2), and the chapter section it comes
from.

The worked example of the chapter (section 3): ids [CLS] the cat sat [SEP]
[PAD] = (2, 5, 6, 7, 3, 0), [MASK] = 4, p = 0.15, and a generator that
returns the uniforms 0.05, 0.50, 0.90, 0.10, 0.85 and then below(V) = 9.
[CLS] is special: no draw. "the": u = 0.05 < 0.15, picked; the action
uniform 0.50 < 0.8 shows [MASK]. "cat": u = 0.90, not picked. "sat":
u = 0.10, picked; 0.85 lies in [0.8, 0.9), a random token, below(V) = 9.
[SEP] and [PAD] are special. Inputs (2, 4, 6, 9, 3, 0), labels
(-100, 5, -100, 7, -100, -100): five uniforms and one below, nothing else.

The golden fixture (course/fixtures/L6.2/bert_tiny_hf.npz) holds a random
tiny Hugging Face BertForMaskedLM (transformers 5.19.0, torch 2.14.1): its
state dict, inputs with padding and token types, logits, loss, and
gradients, from course/oracle/L6.2/bert_hf.py.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.obj.bert import (
    BertConfig,
    BertEncoder,
    BertForMLM,
    load_bert,
    load_hf_bert,
    mlm_batch,
    mlm_mask,
    save_bert,
    special_ids,
)
from tinyllm.tok.wordpiece import WordPieceTokenizer

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L6.2", "bert_tiny_hf.npz")
CFG = BertConfig(
    vocab=40, max_len=16, d_model=16, n_heads=2, n_layers=2, d_ff=32, dropout=0.0
)
MASK_ID = 4


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


class Script:
    """A generator that returns given numbers, and fails on any extra draw."""

    def __init__(self, uniforms, belows) -> None:
        self.u, self.b = list(uniforms), list(belows)

    def uniform(self) -> float:
        assert self.u, "drew a uniform the op order does not have"
        return self.u.pop(0)

    def below(self, n: int) -> int:
        assert self.b, "drew below() where the op order does not"
        return self.b.pop(0)


def spec_mask(ids, special, mask_id, vocab, p, rng):
    """The contract's op order, written independently: C order, specials draw
    nothing, u < p picks, then the inverse CDF of (0.8, 0.1, 0.1)."""
    x = ids.astype(np.int64).copy().reshape(-1)
    lab = np.full(x.shape, -100, dtype=np.int64)
    for i, sp in enumerate(special.reshape(-1)):
        if sp or not rng.uniform() < p:
            continue
        lab[i] = x[i]
        a = rng.uniform()
        if a < 0.8:
            x[i] = mask_id
        elif a < 0.8 + 0.1:
            x[i] = rng.below(vocab)
    return x.reshape(ids.shape), lab.reshape(ids.shape)


def hf():
    f = np.load(FIX)
    return f, {k[3:]: f[k] for k in f.files if k.startswith("sd.")}


def rand_ids(seed, B, T, V=40, lo=5):
    g = PCG32(seed=seed)
    return np.array(
        [[lo + g.below(V - lo) for _ in range(T)] for _ in range(B)], dtype=np.int64
    )


# --- the worked example -------------------------------------------------------------------


def test_hand_example_mlm_mask():
    # WHY: the chapter's worked example, draw for draw: specials draw
    #      nothing, an unpicked position draws one uniform, a picked one draws
    #      a second uniform for the action and below(V) only when the action is
    #      a random token; the label is the ORIGINAL id, -100 elsewhere.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, m03, m04
    # CHAPTER: L6.2 section 3, Worked example by hand
    ids = np.array([[2, 5, 6, 7, 3, 0]])
    special = np.array([[True, False, False, False, True, True]])
    rng = Script([0.05, 0.50, 0.90, 0.10, 0.85], [9])
    inputs, labels = mlm_mask(ids, special, MASK_ID, 40, 0.15, rng)
    assert inputs.tolist() == [[2, 4, 6, 9, 3, 0]]
    assert labels.tolist() == [[-100, 5, -100, 7, -100, -100]]
    assert rng.u == [] and rng.b == [] and ids.tolist() == [[2, 5, 6, 7, 3, 0]]


def test_keep_action_still_predicts():
    # WHY: the third action shows the token itself, yet the position is
    #      still a prediction (label = the id): that is what keeps the model
    #      honest about tokens it can see.
    # KIND: unit
    # CATCHES: s03, s05
    # CHAPTER: L6.2 section 2.2, Masking 15%, and 80/10/10
    inputs, labels = mlm_mask(
        np.array([[8]]), np.array([[False]]), MASK_ID, 40, 0.15, Script([0.0, 0.95], [])
    )
    assert inputs.tolist() == [[8]] and labels.tolist() == [[8]]


# --- the masking spec ---------------------------------------------------------------------


def test_mask_follows_the_op_order():
    # WHY: one seed, one masking, in every implementation: on a [4, 32]
    #      batch with specials, the result equals the op order written out
    #      independently, using the same frozen PCG32 stream.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, m01, m02, m03, m04
    # CHAPTER: L6.2 section 2.2, Masking 15%, and 80/10/10
    ids = rand_ids(1, 4, 32, V=1000)
    special = np.zeros(ids.shape, dtype=bool)
    special[:, 0] = special[:, -1] = True
    special[2, 20:] = True
    got = mlm_mask(ids, special, MASK_ID, 1000, 0.3, Rng(2))
    want = spec_mask(ids, special, MASK_ID, 1000, 0.3, Rng(2))
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_mask_rates():
    # WHY: over 40 000 positions with V = 1000: the picked share is p within
    #      4 binomial standard deviations; among picked positions the input
    #      is [MASK], another token, or the token itself with the exact
    #      probabilities of 80/10/10 (a random draw can hit [MASK] or the
    #      token itself), chi-square p > 1e-3 at a fixed seed.
    # KIND: statistical
    # CATCHES: s02, s04, m01, m03
    # CHAPTER: L6.2 section 2.2, Masking 15%, and 80/10/10
    V, p = 1000, 0.15
    ids = rand_ids(3, 200, 200, V=V)
    inputs, labels = mlm_mask(
        ids, np.zeros(ids.shape, dtype=bool), MASK_ID, V, p, Rng(4)
    )
    picked = labels != -100
    n, k = ids.size, int(picked.sum())
    assert abs(k - n * p) < 4 * np.sqrt(n * p * (1 - p)), f"{k} picked of {n}"
    assert np.array_equal(labels[picked], ids[picked])
    shown = inputs[picked]
    orig = ids[picked]
    obs = np.array(
        [
            (shown == MASK_ID).sum(),
            ((shown != MASK_ID) & (shown != orig)).sum(),
            (shown == orig).sum(),
        ]
    )
    exp = k * np.array([0.8 + 0.1 / V, 0.1 * (V - 2) / V, 0.1 + 0.1 / V])
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    assert chi2 < 13.8, (
        f"chi-square {chi2:.2f} on 2 dof (p < 1e-3): observed {obs.tolist()}, expected {exp.round(1).tolist()}"
    )
    assert np.array_equal(inputs[~picked], ids[~picked])


# --- against Hugging Face -------------------------------------------------------------------------


def test_golden_hf_bert():
    # WHY: a random tiny BertForMaskedLM from Hugging Face, loaded through
    #      load_hf_bert, must give HF's logits (padded positions included),
    #      its MLM loss over the three labelled positions, and the gradient of
    #      that loss for every HF tensor.
    # KIND: golden
    # CATCHES: s07, s08, s09, s10, s11, m05, m06, m07, m08, m09, m10, m11, m15
    # CHAPTER: L6.2 section 2.1, The encoder
    f, sd = hf()
    m = BertForMLM(CFG, rng=Rng(0))
    load_hf_bert(m, sd)
    logits, loss = m(f["ids"], f["token_type_ids"], f["attention_mask"], f["labels"])
    assert_close(logits.data, f["logits"], rtol=1e-4, atol=1e-5)
    assert_close(float(loss.data), float(f["loss"]), rtol=1e-5, atol=1e-6)
    loss.backward()
    g = {n: p.grad for n, p in m.named_parameters()}
    pairs = {
        "bert.word_emb.weight": "bert.embeddings.word_embeddings.weight",
        "bert.pos_emb.weight": "bert.embeddings.position_embeddings.weight",
        "bert.type_emb.weight": "bert.embeddings.token_type_embeddings.weight",
        "bert.emb_norm.weight": "bert.embeddings.LayerNorm.weight",
        "decoder_bias": "cls.predictions.bias",
        "transform.weight": "cls.predictions.transform.dense.weight",
        "transform_norm.bias": "cls.predictions.transform.LayerNorm.bias",
    }
    for i in range(CFG.n_layers):
        e, h = f"bert.layers.{i}.", f"bert.encoder.layer.{i}."
        pairs.update(
            {
                e + "attn.q_proj.weight": h + "attention.self.query.weight",
                e + "attn.k_proj.bias": h + "attention.self.key.bias",
                e + "attn.v_proj.weight": h + "attention.self.value.weight",
                e + "attn.out_proj.weight": h + "attention.output.dense.weight",
                e + "attn_norm.weight": h + "attention.output.LayerNorm.weight",
                e + "ff1.weight": h + "intermediate.dense.weight",
                e + "ff2.bias": h + "output.dense.bias",
                e + "ff_norm.weight": h + "output.LayerNorm.weight",
            }
        )
    for ours, theirs in pairs.items():
        assert_close(g[ours], f["grad." + theirs], rtol=1e-4, atol=1e-6, msg=ours)


# --- the laws -------------------------------------------------------------------------------------------


def test_layer_matches_the_formula():
    # WHY: one BERT layer written out in float64 numpy from the layer's own
    #      weights: post-LN after each residual sum and the EXACT GELU
    #      x * Phi(x), with weights large enough that the tanh approximation
    #      would miss by far more than the tolerance.
    # KIND: differential
    # CATCHES: s06, s07, m07
    # CHAPTER: L6.2 section 2.1, The encoder
    import math

    from tinyllm.autograd.tensor import Tensor
    from tinyllm.obj.bert import BertLayer

    cfg = BertConfig(
        vocab=10, max_len=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0
    )
    layer = BertLayer(cfg, Rng(30))
    g = PCG32(seed=31)
    layer.load_state_dict(
        {
            n: g.normal_array(v.shape) * 0.7 + (1.0 if "norm.weight" in n else 0.0)
            for n, v in layer.state_dict().items()
        }
    )
    x = g.normal_array((2, 5, 8)) * 2.0
    keep = np.array([[True] * 5, [True] * 3 + [False] * 2])
    sd = {n: v.astype(np.float64) for n, v in layer.state_dict().items()}

    def ln(z, p):
        mu, var = z.mean(-1, keepdims=True), z.var(-1, keepdims=True)
        return (z - mu) / np.sqrt(var + 1e-12) * sd[p + ".weight"] + sd[p + ".bias"]

    def lin(z, p):
        return z @ sd[p + ".weight"].T + sd[p + ".bias"]

    q, k, v = (
        lin(x, f"attn.{n}_proj").reshape(2, 5, 2, 4).transpose(0, 2, 1, 3)
        for n in "qkv"
    )
    e = q @ k.transpose(0, 1, 3, 2) / 2.0 + np.where(
        keep[:, None, None, :], 0.0, -np.inf
    )
    a = np.exp(e - e.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    y = ln(
        x + lin((a @ v).transpose(0, 2, 1, 3).reshape(2, 5, 8), "attn.out_proj"),
        "attn_norm",
    )
    u = lin(y, "ff1")
    gelu = u * 0.5 * (1.0 + np.vectorize(math.erf)(u / math.sqrt(2.0)))
    want = ln(y + lin(gelu, "ff2"), "ff_norm")
    assert_close(
        layer(Tensor(x), keep[:, None, None, :]).data, want, rtol=1e-4, atol=1e-4
    )


def test_padding_is_never_read():
    # WHY: a padded position is masked as a key in every layer: changing
    #      the ids there changes no real position's output, bit for bit.
    # KIND: property
    # CATCHES: s08, m10
    # CHAPTER: L6.2 section 2.1, The encoder
    m = BertEncoder(CFG, rng=Rng(5))
    x = rand_ids(6, 2, 7)
    am = np.ones(x.shape, dtype=bool)
    am[1, 4:] = False
    a = m(x, None, am).data
    y = x.copy()
    y[1, 4:] = [9, 10, 11]
    b = m(y, None, am).data
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1, :4], b[1, :4])


def test_reads_both_directions():
    # WHY: unlike GPT (L6.1), every position reads the whole sentence:
    #      changing the LAST token changes the FIRST position's output. That
    #      is why BERT can fill a blank but cannot generate.
    # KIND: property
    # CATCHES: s11
    # CHAPTER: L6.2 section 2.1, The encoder
    m = BertEncoder(CFG, rng=Rng(7))
    x = rand_ids(8, 1, 6)
    y = x.copy()
    y[0, -1] = 5 if x[0, -1] != 5 else 6
    assert differs(m(x).data[0, 0], m(y).data[0, 0])


def test_token_types_are_added():
    # WHY: segment embeddings tell sentence A from sentence B; type 1
    #      changes the output, and the default is all type 0.
    # KIND: property
    # CATCHES: s09
    # CHAPTER: L6.2 section 2.1, The encoder
    m = BertEncoder(CFG, rng=Rng(9))
    x = rand_ids(10, 1, 5)
    assert np.array_equal(m(x).data, m(x, np.zeros_like(x)).data)
    assert differs(m(x, np.ones_like(x)).data, m(x).data)


def test_loss_only_at_labelled_positions():
    # WHY: the MLM loss is the mean cross-entropy over the labelled
    #      positions only; every other row of the logits gets no gradient.
    # KIND: unit
    # CATCHES: m15
    # CHAPTER: L6.2 section 2.3, The MLM head and loss
    m = BertForMLM(CFG, rng=Rng(11))
    x = rand_ids(12, 1, 6)
    labels = np.full(x.shape, -100)
    labels[0, 2] = 17
    logits, loss = m(x, labels=labels)
    z = logits.data[0, 2].astype(np.float64)
    want = np.log(np.exp(z - z.max()).sum()) + z.max() - z[17]
    assert_close(float(loss.data), want, rtol=1e-5, atol=1e-6)


def test_decoder_is_tied_and_has_a_bias():
    # WHY: the MLM decoder reuses the word embedding table as its weight
    #      (one parameter) and adds its own bias; the names are the keys the
    #      zoo and ELECTRA (L6.3) load.
    # KIND: unit
    # CATCHES: m11
    # CHAPTER: L6.2 section 2.3, The MLM head and loss
    m = BertForMLM(CFG, rng=Rng(13))
    names = [n for n, _ in m.named_parameters()]
    # Own parameters come before children (L0.4): decoder_bias first.
    assert names[:4] == [
        "decoder_bias",
        "bert.word_emb.weight",
        "bert.pos_emb.weight",
        "bert.type_emb.weight",
    ]
    assert not any("decoder.weight" in n for n in names)
    assert names.index("transform.weight") < names.index("transform_norm.weight")


# --- the interface -------------------------------------------------------------------------------------


def wordpiece():
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "cat",
        "sat",
        "##s",
        "on",
        "mat",
    ]
    return WordPieceTokenizer({t: i for i, t in enumerate(vocab)})


def test_mlm_batch_from_text():
    # WHY: a batch from raw text through L1.3's WordPiece: [CLS] ... [SEP],
    #      padding with [PAD] and attention_mask False there, and NO special
    #      or padded position is ever picked (p = 1 picks every other one,
    #      with the original id as its label).
    # KIND: unit
    # CATCHES: s01, s02, s04, s12, m12
    # CHAPTER: L6.2 section 4, The interface
    tok = wordpiece()
    assert special_ids(tok) == [0, 1, 2, 3, 4]
    b = mlm_batch(tok, ["the cats sat", "the cat"], 16, 1.0, Rng(14))
    want = np.array([[2, 5, 6, 8, 7, 3], [2, 5, 6, 3, 0, 0]])
    assert b["attention_mask"].tolist() == [[True] * 6, [True] * 4 + [False] * 2]
    assert np.array_equal(
        b["labels"], np.where(np.isin(want, [0, 1, 2, 3, 4]), -100, want)
    )
    assert np.all(b["token_type_ids"] == 0)
    sp = np.isin(want, [0, 2, 3])
    assert np.array_equal(b["input_ids"][sp], want[sp])
    none = mlm_batch(tok, ["the cats sat on the mat"], 5, 0.0, Rng(15))
    assert none["input_ids"].tolist() == [[2, 5, 6, 8, 3]] and np.all(
        none["labels"] == -100
    )


def test_load_encoder_alone_and_roundtrip(tmp_path):
    # WHY: a classifier (L6.5) loads only the encoder from an MLM checkpoint;
    #      it must equal the MLM model's own encoder. The model directory
    #      saves and loads exactly.
    # KIND: unit
    # CATCHES: m11, m13
    # CHAPTER: L6.2 section 4, The interface
    f, sd = hf()
    mlm, enc = BertForMLM(CFG, rng=Rng(16)), BertEncoder(CFG, rng=Rng(17))
    load_hf_bert(mlm, sd)
    load_hf_bert(enc, sd)
    x, tt, am = f["ids"], f["token_type_ids"], f["attention_mask"]
    assert np.array_equal(enc(x, tt, am).data, mlm.bert(x, tt, am).data)
    save_bert(mlm, str(tmp_path / "b"))
    back = load_bert(str(tmp_path / "b"))
    assert back.bert.cfg == CFG and np.array_equal(
        back(x, tt, am)[0].data, mlm(x, tt, am)[0].data
    )


def test_validation():
    # WHY: masking needs integer ids, a special mask of their shape, a rate
    #      in [0, 1], and a [MASK] id inside the vocabulary; loading needs a
    #      BERT module.
    # KIND: boundary
    # CATCHES: m10, m14
    # CHAPTER: L6.2 section 4, The interface
    x, sp = np.array([[5, 6]]), np.array([[False, False]])
    with pytest.raises(ValueError):
        mlm_mask(x.astype(float), sp, MASK_ID, 40, 0.15, Rng(18))
    with pytest.raises(ValueError):
        mlm_mask(x, sp[:, :1], MASK_ID, 40, 0.15, Rng(18))
    with pytest.raises(ValueError):
        mlm_mask(x, sp, MASK_ID, 40, 1.5, Rng(18))
    with pytest.raises(ValueError):
        mlm_mask(x, sp, 40, 40, 0.15, Rng(18))
    with pytest.raises(TypeError):
        load_hf_bert(object(), {})
    with pytest.raises(ValueError):
        BertEncoder(CFG)(x, None, np.ones((1, 3), dtype=bool))
