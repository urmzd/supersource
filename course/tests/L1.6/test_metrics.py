"""Course tests for L1.6: tokenizer metrics (tinyllm/tok/metrics.py).

Rung R0 for these course tests (your own tests for this module are rung R2:
section 4 of the chapter lists their names). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L1.6), and the chapter section it comes from.

Oracle (course/oracle/tok/golden.py): fertility, bytes per token, and the
fallback rate of GPT-2, SmolLM2, bert-base-uncased, and a Hugging Face
Unigram on the 300 shared test strings, computed with Hugging Face
`tokenizers` (course/fixtures/L1.6/metrics.json). The tokenizers are loaded
here through your L1.2, L1.3, and L1.4 loaders.

The chapter's worked example (section 3) measures the text "naïve café"
(12 UTF-8 bytes, 10 code points, words "naïve" and "café") with two
tokenizers:

    tokenizer                        tokens  bytes/token  fertility  fallback
    ByteTokenizer                        12       1.0         5.5      4/12
    CharTokenizer.train(["naïve"])       10       1.2         4.5      4/10
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Sequence

import pytest
from _lib.close import assert_close
from tinyllm.info.ppl import NLLAccumulator
from tinyllm.tok.base import ByteTokenizer
from tinyllm.tok.bpe import BPETokenizer
from tinyllm.tok.char import CharTokenizer
from tinyllm.tok.metrics import (
    byte_fallback_rate,
    bytes_per_token,
    fertility,
    is_fallback,
)
from tinyllm.tok.unigram import UnigramTokenizer
from tinyllm.tok.wordpiece import WordPieceTokenizer

FX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
TEXT = "naïve café"
WORDS = ["naïve", "café"]


def char_tok() -> CharTokenizer:
    return CharTokenizer.train(["naïve"])


def golden() -> dict:
    with open(FX / "L1.6" / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def texts() -> list[str]:
    # The 300 shared strings: the "text" field of every oracle case file.
    with open(FX / "tok-gpt2" / "cases.jsonl", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


class Bare:
    """The smallest object the Tokenizer protocol allows: two ids, "a" and
    "b"; nothing else (no vocab attribute, no merges), and it is not a
    subclass of anything in tinyllm."""

    vocab_size = 3
    special_ids = {"<s>": 2}
    unk_id: Optional[int] = None

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = [0 if ch == "a" else 1 for ch in text]
        return [2] + ids if add_special else ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return "".join(
            {0: "a", 1: "b", 2: "" if skip_special else "<s>"}[i] for i in ids
        )

    def token_to_id(self, s: str) -> Optional[int]:
        return {"a": 0, "b": 1, "<s>": 2}.get(s)

    def id_to_token(self, i: int) -> str:
        return ["a", "b", "<s>"][i]

    def save(self, dir: str) -> None:
        pass

    @classmethod
    def load(cls, dir: str) -> "Bare":
        return cls()


# --- the worked example ---------------------------------------------------------


def test_hand_example_metrics():
    # WHY: section 3 by hand. "naïve café" is 12 bytes ("ï" and "é" take two
    #      each) but 10 code points. Bytes: 12 tokens, 1 byte each; the four
    #      halves of "ï" and "é" decode alone to U+FFFD, so 4 of 12 are
    #      fallbacks. Char tokenizer trained on "naïve": the space, "c", "f",
    #      and "é" are <unk>, so 4 of 10; "café" costs c a f é = 4 tokens.
    # KIND: unit
    # CATCHES: s01, s02, s05, s07, s09
    # CHAPTER: L1.6 section 3, Worked example by hand
    b, c = ByteTokenizer(), char_tok()
    assert bytes_per_token(b, [TEXT]) == 1.0
    assert fertility(b, WORDS) == 5.5
    assert byte_fallback_rate(b, [TEXT]) == 4 / 12
    assert bytes_per_token(c, [TEXT]) == 1.2
    assert fertility(c, WORDS) == 4.5
    assert byte_fallback_rate(c, [TEXT]) == 0.4


def test_golden_metrics():
    # WHY: the three metrics of four real tokenizers on 300 strings (emoji,
    #      CJK, control characters, special-token texts, white-space runs)
    #      equal what Hugging Face tokenizers gives. Each is a ratio of
    #      integer counts, so only rounding of the last division can differ.
    #      GPT-2 matches "<|endoftext|>" in the text as a special id, which
    #      is never a fallback; BERT and the Unigram have an unknown id,
    #      which always is.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, s07
    # CHAPTER: L1.6 section 4, What the tests check
    g = golden()
    t = texts()
    words = sorted({w for x in t for w in x.split()})
    assert len(words) == g["words"]
    toks = {
        "gpt2": BPETokenizer.from_hf_json(str(FX / "tok-gpt2" / "tokenizer.json")),
        "smollm2": BPETokenizer.from_hf_json(
            str(FX / "tok-smollm2" / "tokenizer.json")
        ),
        "bert": WordPieceTokenizer.from_hf_json(
            str(FX / "tok-bert" / "tokenizer.json")
        ),
        "hf-unigram": UnigramTokenizer.from_hf_json(
            str(FX / "L1.4" / "hf-unigram" / "tokenizer.json")
        ),
    }
    for name, tok in toks.items():
        want = g[name]
        got = {
            "fertility": fertility(tok, words),
            "bytes_per_token": bytes_per_token(tok, t),
            "byte_fallback_rate": byte_fallback_rate(tok, t),
        }
        for k, v in got.items():
            assert_close(v, want[k], rtol=1e-12, atol=0.0, msg=f"{name} {k}")


# --- definitions ------------------------------------------------------------------


def test_bytes_are_utf8_not_code_points():
    # WHY: compression is measured in UTF-8 bytes, the unit bits per byte
    #      uses. len(text) counts code points: an emoji is 1 code point but 4
    #      bytes, so a code-point count understates every non-ASCII text.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L1.6 section 5, Pitfalls, item 1
    b = ByteTokenizer()
    assert bytes_per_token(b, ["\U0001f642"]) == 1.0
    c = CharTokenizer.train(["\U0001f642日"])
    assert bytes_per_token(c, ["\U0001f642日"]) == 3.5  # 4 + 3 bytes, 2 tokens


def test_ratio_of_totals_not_mean_of_ratios():
    # WHY: bytes_per_token is total bytes over total tokens. Averaging the
    #      per-text ratios weights a 1-byte text like a 1000-byte one, and
    #      bits per byte (M11.2) is a ratio of totals too.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: L1.6 section 2, Principles
    c = CharTokenizer.train(["ab"])
    # "ab" -> 2 tokens, 2 bytes; "é" -> 1 token (<unk>), 2 bytes.
    # totals: 4 / 3; the mean of ratios would be (1 + 2) / 2 = 1.5.
    assert bytes_per_token(c, ["ab", "é"]) == 4 / 3
    assert byte_fallback_rate(c, ["ab", "é"]) == 1 / 3


def test_words_are_encoded_one_at_a_time():
    # WHY: fertility asks what one word costs on its own. Encoding the words
    #      joined by spaces adds the space tokens (bytes) or lets BPE merge
    #      across the join, and that is no longer tokens per word.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L1.6 section 5, Pitfalls, item 2
    b = ByteTokenizer()
    assert fertility(b, ["ab", "c"]) == 1.5
    assert fertility(b, ["a"]) == 1.0


def test_repeated_words_count_each_time():
    # WHY: fertility is a mean over the word list as given. A word that occurs
    #      twice counts twice; deduplicating changes the weighting.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: L1.6 section 2, Principles
    b = ByteTokenizer()
    assert fertility(b, ["a", "a", "bb"]) == 4 / 3


def test_no_special_tokens_are_added():
    # WHY: a [CLS] ... [SEP] or <bos> ... <eos> template is not text. Counting
    #      it would charge every word two extra tokens and make a tokenizer
    #      with a template look worse than one without.
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: L1.6 section 5, Pitfalls, item 3
    c = CharTokenizer.train(["hi"], specials=("<bos>", "<eos>"))
    assert fertility(c, ["hi"]) == 2.0
    assert bytes_per_token(c, ["hi"]) == 1.0
    assert byte_fallback_rate(c, ["hi"]) == 0.0
    assert fertility(Bare(), ["ab", "a"]) == 1.5


def test_is_fallback_rules():
    # WHY: a fallback is a token that is not whole characters: the unknown
    #      id, or a piece of a multi-byte character (it decodes alone to
    #      U+FFFD). A whole multi-byte character ("é" in GPT-2 is one token)
    #      is not, and a special token never is.
    # KIND: unit
    # CATCHES: s04, s05
    # CHAPTER: L1.6 section 2, Principles
    b = ByteTokenizer()
    assert is_fallback(b, 0xC3) and is_fallback(b, 0xA9)
    assert not is_fallback(b, ord("a"))
    c = CharTokenizer.train(["a�"], specials=("<bos>",))
    assert is_fallback(c, c.unk_id)
    assert not is_fallback(c, c.special_ids["<bos>"])
    assert is_fallback(c, c.token_to_id("�"))  # text already lost upstream
    gpt2 = BPETokenizer.from_hf_json(str(FX / "tok-gpt2" / "tokenizer.json"))
    (e_acute,) = gpt2.encode("é")
    assert not is_fallback(gpt2, e_acute)
    assert not is_fallback(gpt2, gpt2.special_ids["<|endoftext|>"])


def test_fallback_counts_every_occurrence():
    # WHY: the rate is over token occurrences, not over distinct ids: "éé" in
    #      bytes is four fallback tokens out of four.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L1.6 section 4, The interface
    b = ByteTokenizer()
    assert byte_fallback_rate(b, ["éé"]) == 1.0
    assert byte_fallback_rate(b, ["aé", "é"]) == 4 / 5


def test_empty_inputs_raise_value_error():
    # WHY: a metric over nothing has no value; returning 0 or NaN would let an
    #      empty eval split pass a threshold. Each case is ValueError, not a
    #      ZeroDivisionError the caller cannot tell from a bug.
    # KIND: boundary
    # CATCHES: s11, m01
    # CHAPTER: L1.6 section 4, The interface
    b = ByteTokenizer()
    with pytest.raises(ValueError):
        fertility(b, [])
    with pytest.raises(ValueError):
        bytes_per_token(b, [])
    with pytest.raises(ValueError):
        bytes_per_token(b, ["", ""])
    with pytest.raises(ValueError):
        byte_fallback_rate(b, [""])
    assert fertility(b, [""]) == 0.0  # one word that encodes to nothing is a mean
    assert bytes_per_token(b, ["a"]) == 1.0  # one token is enough to measure


def test_bits_per_byte_is_tokenizer_independent():
    # WHY: per-token loss depends on the tokenizer; bits per byte does not.
    #      The same model cost S nats on "naïve café" gives the same bpb from
    #      M11.2's accumulator whichever tokenizer split it, and
    #      bits_per_token / bytes_per_token is that bpb: the exchange rate.
    # KIND: property
    # CATCHES: s07
    # CHAPTER: L1.6 section 2, Principles
    total = 12 * math.log(2.0)  # 12 bits over 12 bytes: 1 bit per byte
    n_bytes = len(TEXT.encode("utf-8"))
    for tok in (ByteTokenizer(), char_tok()):
        n = len(tok.encode(TEXT))
        acc = NLLAccumulator()
        acc.add([total / n] * n, n_bytes=n_bytes)
        r = acc.result()
        assert_close(r["bpb"], 1.0, rtol=1e-12, atol=0.0)
        assert_close(
            r["bits_per_token"] / bytes_per_token(tok, [TEXT]),
            r["bpb"],
            rtol=1e-12,
            atol=0.0,
        )


def test_any_protocol_object_is_measured():
    # WHY: the metrics see only the L1.1 protocol (encode, decode, unk_id,
    #      special_ids), so the Rust tokenizer of L1.5 and any later one is
    #      measured by the same code; a special id is never a fallback,
    #      whatever its text.
    # KIND: unit
    # CATCHES: s04
    # CHAPTER: L1.6 section 4, The interface
    t = Bare()
    assert bytes_per_token(t, ["abba"]) == 1.0
    assert byte_fallback_rate(t, ["abba"]) == 0.0
    assert not is_fallback(t, 2)
