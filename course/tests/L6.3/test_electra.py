"""Course tests for L6.3: ELECTRA replaced-token detection
(tinyllm/obj/electra.py).

Rung R0 for these: read them before you write code. Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L6.3), and the chapter section it comes
from. The generator and the discriminator's body are L6.2's BertForMLM and
BertEncoder; masking is L6.2's mlm_mask; sampling is M07.1's
sample_categorical.

The worked example of the chapter (section 3): ids (5, 6, 7, 8), positions
1 and 3 masked (labels -100, 6, -100, 8). The generator's row at position 1
puts 0.75 on token 6 and 0.25 on token 9; at position 3, 0.5 each on tokens
2 and 8. The uniforms are 0.1 then 0.3: position 1 samples 6 (the original:
NOT replaced), position 3 samples 2 (replaced). corrupt = (5, 6, 7, 2),
is_replaced = (0, 0, 0, 1). With all discriminator logits 0 the loss is
ln 2 = 0.693147; with logits (-2, -2, -2, 2) every call is right and the
loss is ln(1 + e^-2) = 0.126928.

The golden fixture (course/fixtures/L6.3/electra_hf.npz) holds a tiny random
Hugging Face ElectraForPreTraining (transformers 5.19.0): its weights, logits,
loss, and gradients on a padded batch (course/oracle/L6.3/electra_hf.py).
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.tensor import Tensor
from tinyllm.obj.bert import BertConfig, hf_bert_encoder_sd, mlm_mask
from tinyllm.obj.electra import (
    ELECTRA,
    ElectraDiscriminator,
    electra_step,
    load_electra,
    replace_tokens,
    rtd_accuracy,
    rtd_loss,
    save_electra,
)

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L6.3", "electra_hf.npz")
MASK_ID = 1


class Rng:
    """The frozen PCG32 behind the generator API of M06.3; counts uniforms."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)
        self.uniforms_drawn = 0

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        self.uniforms_drawn += 1
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)], dtype=np.float64)

    def below(self, n: int) -> int:
        return self.g.below(n)


class Scripted:
    def __init__(self, us) -> None:
        self.us = list(us)

    def uniform(self) -> float:
        return self.us.pop(0)


def cfg(d=8, layers=1, ff=16, vocab=20) -> BertConfig:
    return BertConfig(
        vocab=vocab,
        max_len=12,
        d_model=d,
        n_heads=2,
        n_layers=layers,
        d_ff=ff,
        dropout=0.0,
    )


def batch(s=3, B=3, T=9):
    g = PCG32(seed=s)
    ids = np.array(
        [[2 + g.below(18) for _ in range(T)] for _ in range(B)], dtype=np.int64
    )
    real = np.ones((B, T), dtype=bool)
    real[1, 6:] = False
    real[2, 4:] = False
    special = np.zeros((B, T), dtype=bool)
    special[:, 0] = True  # a [CLS]-like first token
    return ids, real, special


def log_probs(table: dict[int, float], V: int = 10) -> np.ndarray:
    row = np.full(V, -np.inf)
    for k, p in table.items():
        row[k] = math.log(p)
    return row


# --- the worked example ---------------------------------------------------------------


def test_hand_example_replaced_token_labels():
    # WHY: the chapter's worked example: a sample equal to the original token
    #      is labelled ORIGINAL, one uniform per masked position in order,
    #      and the discriminator's loss is the mean BCE over the tokens.
    # KIND: unit
    # CATCHES: s01, s02, m01
    # CHAPTER: L6.3 section 3, Worked example by hand
    ids = np.array([[5, 6, 7, 8]])
    labels = np.array([[-100, 6, -100, 8]])
    logits = np.zeros((1, 4, 10))
    logits[0, 1] = log_probs({6: 0.75, 9: 0.25})
    logits[0, 3] = log_probs({2: 0.5, 8: 0.5})
    r = Scripted([0.1, 0.3])
    corrupt, rep = replace_tokens(ids, labels, logits, r)
    assert r.us == []
    assert corrupt.tolist() == [[5, 6, 7, 2]]
    assert rep.tolist() == [[0.0, 0.0, 0.0, 1.0]]
    assert corrupt.dtype == np.int64 and rep.dtype == np.float32
    assert_close(
        rtd_loss(Tensor(np.zeros((1, 4))), rep).data, math.log(2.0), dtype="float32"
    )
    good = Tensor(np.array([[-2.0, -2.0, -2.0, 2.0]]))
    assert_close(rtd_loss(good, rep).data, math.log1p(math.exp(-2.0)), dtype="float32")


def test_a_lucky_sample_is_original():
    # WHY: a generator that is certain about the original token "replaces"
    #      it with itself; labelling that position as replaced would teach the
    #      discriminator to flag tokens that were never changed.
    # KIND: boundary
    # CATCHES: s01, m01
    # CHAPTER: L6.3 section 2.2, The discriminator's labels
    ids, _, _ = batch(4)
    labels = np.where(PCG32(seed=1).uniform_array(ids.shape) < 0.5, ids, -100)
    logits = np.full(ids.shape + (20,), -30.0)
    np.put_along_axis(logits, ids[..., None], 30.0, axis=-1)
    corrupt, rep = replace_tokens(ids, labels, logits, Rng(2))
    assert np.array_equal(corrupt, ids)
    assert not rep.any()


def test_replace_draws_one_uniform_per_masked_position():
    # WHY: one uniform per picked position, in C order, and none for the
    #      others: the same seed must give the same corruption in every
    #      implementation, after mlm_mask's own draws.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L6.3 section 2.1, One step
    ids = np.array([[3, 3, 3], [3, 3, 3]])
    labels = np.array([[3, -100, 3], [-100, 3, -100]])
    logits = np.zeros((2, 3, 4))  # uniform over 4 tokens: u in [k/4, (k+1)/4) picks k
    r = Scripted([0.1, 0.6, 0.9])
    corrupt, rep = replace_tokens(ids, labels, logits, r)
    assert corrupt.tolist() == [[0, 3, 2], [3, 3, 3]]
    assert rep.tolist() == [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    assert r.us == []


# --- against Hugging Face ------------------------------------------------------------------


def test_discriminator_matches_hf():
    # WHY: the discriminator is BERT's encoder plus HF's head
    #      (dense, GELU, dense_prediction), and its loss is BCE averaged over
    #      the real tokens only: logits, loss, and gradients match
    #      ElectraForPreTraining with the same weights.
    # KIND: golden
    # CATCHES: s03, s04, m02
    # CHAPTER: L6.3 section 2.3, The discriminator
    f = np.load(FIX)
    c = BertConfig(
        vocab=40, max_len=16, d_model=16, n_heads=2, n_layers=2, d_ff=32, dropout=0.0
    )
    d = ElectraDiscriminator(c, rng=Rng(0))
    hf = {
        k[len("param.electra.") :]: f[k]
        for k in f.files
        if k.startswith("param.electra.")
    }
    sd = {"electra." + k: v for k, v in hf_bert_encoder_sd(hf, 2).items()}
    for k in f.files:
        if k.startswith("param.discriminator_predictions."):
            sd[k[len("param.") :]] = f[k]
    assert sorted(sd) == sorted(n for n, _ in d.named_parameters())
    d.load_state_dict(sd)
    logits = d(f["ids"], f["types"], f["mask"])
    assert_close(logits.data, f["logits"], rtol=1e-4, atol=1e-5)
    loss = rtd_loss(logits, f["labels"], f["mask"])
    assert_close(loss.data, f["loss"], rtol=1e-5, atol=1e-6)
    loss.backward()
    h = d.discriminator_predictions
    assert_close(h.dense.weight.grad, f["grad.dense.weight"], rtol=1e-3, atol=1e-6)
    assert_close(
        h.dense_prediction.weight.grad,
        f["grad.dense_prediction.weight"],
        rtol=1e-3,
        atol=1e-6,
    )
    assert_close(
        d.electra.word_emb.weight.grad, f["grad.word_emb.weight"], rtol=1e-3, atol=1e-6
    )


def test_rtd_loss_ignores_padding():
    # WHY: padded positions are not text; whatever their logits and labels,
    #      they must not move the loss or the mean's denominator.
    # KIND: property
    # CATCHES: s04
    # CHAPTER: L6.3 section 2.3, The discriminator
    g = PCG32(seed=8)
    z = g.uniform_array((3, 5), -3, 3)
    y = (g.uniform_array((3, 5)) < 0.3).astype(np.float32)
    real = np.ones((3, 5), dtype=bool)
    real[0, 3:] = False
    want = np.mean(
        [
            math.log1p(math.exp(-zz)) if yy else math.log1p(math.exp(zz))
            for zz, yy in zip(z[real], y[real])
        ]
    )
    assert_close(rtd_loss(Tensor(z), y, real).data, want, rtol=1e-5, atol=1e-6)
    z2, y2 = z.copy(), y.copy()
    z2[~real], y2[~real] = 50.0, 0.0
    assert_close(rtd_loss(Tensor(z2), y2, real).data, want, rtol=1e-5, atol=1e-6)


# --- one training step -----------------------------------------------------------------------


def test_electra_step_is_its_pieces():
    # WHY: one step is exactly mlm_mask (padding counts as special), the
    #      generator's MLM loss, one sample per picked position from the
    #      SAME generator, the discriminator on the corrupted ids, and
    #      L_G + lam L_D: recomputed here piece by piece from a copy of the rng.
    # KIND: differential
    # CATCHES: s05, s06, s07, m03
    # CHAPTER: L6.3 section 2.1, One step
    ids, real, special = batch(5)
    model = ELECTRA(cfg(layers=1), cfg(layers=2), rng=Rng(1))
    r = Rng(7)
    r2 = copy.deepcopy(r)
    out = electra_step(
        model.generator,
        model.discriminator,
        ids,
        r,
        50.0,
        mask_id=MASK_ID,
        special_mask=special,
        attn_mask=real,
        p=0.4,
    )
    inputs, labels = mlm_mask(ids, special | ~real, MASK_ID, 20, 0.4, r2)
    assert np.array_equal(out["inputs"], inputs) and np.array_equal(
        out["labels"], labels
    )
    gl, gloss = model.generator(inputs, None, real, labels)
    corrupt, rep = replace_tokens(ids, labels, gl.data, r2)
    assert np.array_equal(out["corrupt"], corrupt) and np.array_equal(
        out["is_replaced"], rep
    )
    dl = model.discriminator(corrupt, None, real)
    dloss = rtd_loss(dl, rep, real)
    assert_close(out["gen_loss"].data, gloss.data, dtype="float32")
    assert_close(out["disc_loss"].data, dloss.data, dtype="float32")
    assert_close(out["loss"].data, gloss.data + 50.0 * dloss.data, rtol=1e-5, atol=1e-5)
    assert r.uniforms_drawn == r2.uniforms_drawn


def test_padding_is_never_masked_or_scored():
    # WHY: padding is not text: it is never picked for masking, never
    #      replaced, and never in the discriminator's loss; specials (the
    #      [CLS]-like first token) are never picked either.
    # KIND: property
    # CATCHES: s05, m03
    # CHAPTER: L6.3 section 5, Pitfalls
    ids, real, special = batch(6)
    model = ELECTRA(cfg(), cfg(), rng=Rng(2))
    out = electra_step(
        model.generator,
        model.discriminator,
        ids,
        Rng(3),
        mask_id=MASK_ID,
        special_mask=special,
        attn_mask=real,
        p=0.9,
    )
    off = ~real | special
    assert (out["labels"][off] == -100).all()
    assert np.array_equal(out["corrupt"][off], ids[off])
    assert (out["labels"][~off] != -100).sum() >= 5


def test_no_gradient_through_the_sample():
    # WHY: sampling is not differentiable; the generator learns from its MLM
    #      loss alone (the paper's choice), so its gradients from the full
    #      loss equal its gradients from L_G alone.
    # KIND: property
    # CATCHES: s07
    # CHAPTER: L6.3 section 2.1, One step
    ids, real, special = batch(7)
    a = ELECTRA(cfg(), cfg(), rng=Rng(4), tie_embeddings=False)
    b = ELECTRA(cfg(), cfg(), rng=Rng(4), tie_embeddings=False)
    kw = dict(mask_id=MASK_ID, special_mask=special, attn_mask=real, p=0.5)
    electra_step(a.generator, a.discriminator, ids, Rng(9), **kw)["loss"].backward()
    electra_step(b.generator, b.discriminator, ids, Rng(9), **kw)["gen_loss"].backward()
    for (n, pa), (_, pb) in zip(
        a.generator.named_parameters(), b.generator.named_parameters()
    ):
        assert pb.grad is not None or pa.grad is None, n
        if pb.grad is not None:
            assert_close(pa.grad, pb.grad, rtol=1e-5, atol=1e-6, msg=n)
    assert any(
        p.grad is not None and np.abs(p.grad).sum() > 0
        for p in a.discriminator.parameters()
    )


def test_lambda_weights_the_discriminator():
    # WHY: the discriminator's loss is per token and much smaller than the
    #      MLM loss, so the paper weights it by lam = 50; the default must be
    #      50 and the total must be L_G + lam L_D for any lam.
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L6.3 section 2.1, One step
    ids, real, special = batch(8)
    m = ELECTRA(cfg(), cfg(), rng=Rng(5))
    kw = dict(mask_id=MASK_ID, special_mask=special, attn_mask=real, p=0.5)
    for lam in (None, 1.0, 7.5):
        o = m(ids, Rng(11), **kw) if lam is None else m(ids, Rng(11), lam=lam, **kw)
        want = o["gen_loss"].data + (50.0 if lam is None else lam) * o["disc_loss"].data
        assert_close(o["loss"].data, want, rtol=1e-5, atol=1e-5)


# --- the model ------------------------------------------------------------------------------


def test_tied_embeddings_are_one_tensor():
    # WHY: ELECTRA shares the token and position embeddings between the two
    #      networks: one Tensor, saved once, trained by both losses.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: L6.3 section 2.4, Sharing embeddings
    m = ELECTRA(cfg(), cfg(layers=2), rng=Rng(6))
    assert m.generator.bert.word_emb.weight is m.discriminator.electra.word_emb.weight
    assert m.generator.bert.pos_emb.weight is m.discriminator.electra.pos_emb.weight
    names = [n for n, _ in m.named_parameters()]
    assert "discriminator.electra.word_emb.weight" in names
    assert not any(n.startswith("generator.bert.word_emb") for n in names)
    assert names[0].startswith("discriminator.")
    with pytest.raises(ValueError):
        ELECTRA(cfg(d=8), cfg(d=12), rng=Rng(0))
    with pytest.raises(ValueError):
        ELECTRA(cfg(vocab=20), cfg(vocab=21), rng=Rng(0), tie_embeddings=False)
    ELECTRA(cfg(d=8), cfg(d=12), rng=Rng(0), tie_embeddings=False)


def test_save_load_roundtrip(tmp_path):
    # WHY: the zoo loads ELECTRA checkpoints by tl_arch: the directory holds
    #      both networks, the tie, the mask id, and the specials, and the
    #      loaded model scores exactly like the saved one.
    # KIND: property
    # CATCHES: s08, m04
    # CHAPTER: L6.3 section 4, The interface
    ids, real, special = batch(9)
    m = ELECTRA(cfg(), cfg(layers=2, d=8), rng=Rng(7))
    save_electra(m, str(tmp_path / "e"), mask_id=MASK_ID, special_ids=[3, 0])
    back, conf = load_electra(str(tmp_path / "e"))
    assert (
        conf["tl_arch"] == "electra"
        and conf["tl_mask_id"] == MASK_ID
        and conf["tl_special_ids"] == [0, 3]
    )
    assert (
        conf["num_hidden_layers"] == 2
        and conf["tl_generator"]["num_hidden_layers"] == 1
    )
    assert (
        back.generator.bert.word_emb.weight
        is back.discriminator.electra.word_emb.weight
    )
    for (n, a), (_, b) in zip(m.named_parameters(), back.named_parameters()):
        assert np.array_equal(a.data, b.data), n
    m.eval()
    kw = dict(mask_id=MASK_ID, special_mask=special, attn_mask=real, p=0.5)
    assert np.array_equal(
        m(ids, Rng(1), **kw)["disc_logits"].data,
        back(ids, Rng(1), **kw)["disc_logits"].data,
    )
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "config.json").write_text('{"tl_arch": "bert"}')
    with pytest.raises(ValueError):
        load_electra(str(tmp_path / "x"))


def test_rtd_accuracy_definition():
    # WHY: the zoo's ELECTRA row is replaced-token detection accuracy over
    #      the real tokens; a discriminator that always says "original"
    #      scores exactly the fraction of tokens that were not replaced.
    # KIND: unit
    # CATCHES: m05
    # CHAPTER: L6.3 section 4, The interface
    ids, real, special = batch(10)
    m = ELECTRA(cfg(), cfg(), rng=Rng(8))
    head = m.discriminator.discriminator_predictions.dense_prediction
    head.weight.data[...] = 0.0
    head.bias.data[...] = -5.0  # every logit -5: "original"
    kw = dict(mask_id=MASK_ID, special_mask=special, attn_mask=real, p=0.6)
    acc = rtd_accuracy(m, ids, Rng(4), **kw)
    out = m(ids, Rng(4), **kw)
    assert acc.shape == (int(real.sum()),)
    assert_close(
        acc.mean(),
        1.0 - out["is_replaced"][real].astype(np.float64).mean(),
        dtype="float64",
    )
    head.bias.data[...] = 5.0  # "replaced" everywhere
    assert_close(
        rtd_accuracy(m, ids, Rng(4), **kw).mean(),
        out["is_replaced"][real].astype(np.float64).mean(),
        dtype="float64",
    )
