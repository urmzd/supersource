"""Course tests for L1.3: WordPiece and the BERT basic tokenizer
(tinyllm/tok/wordpiece.py).

Rung R0 for these course tests (your own tests for this module are rung R2:
section 4 of the chapter lists their names). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L1.3), and the chapter section it comes from.

Oracle (course/oracle/tok/golden.py): bert-base-uncased's tokenizer.json and
the ids, template ids, and decodes Hugging Face tokenizers gives for 300
strings. The chapter's worked example (section 3) is the 15-entry vocab
HAND below and the text "The UNAFFABLE!", which encodes to [5, 6, 7, 8, 10].
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tinyllm.tok.wordpiece import WordPieceTokenizer

FX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
HAND = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "the",
    "un",
    "##aff",
    "##able",
    "##a",
    "!",
    "u",
    "##n",
    "cafe",
    "##s",
]
_cache: dict[str, WordPieceTokenizer] = {}


def hand(**kw) -> WordPieceTokenizer:
    return WordPieceTokenizer({t: i for i, t in enumerate(HAND)}, **kw)


def bert() -> WordPieceTokenizer:
    if "bert" not in _cache:
        _cache["bert"] = WordPieceTokenizer.from_hf_json(
            str(FX / "tok-bert" / "tokenizer.json")
        )
    return _cache["bert"]


def cases() -> list[dict]:
    with open(FX / "tok-bert" / "cases.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_hand_example_basic_tokenize():
    # WHY: section 3: lowercase, split on white space, punctuation alone, and
    #      every CJK ideograph its own word; accents go with lowercasing.
    # KIND: unit
    # CATCHES: s16
    # CHAPTER: L1.3 section 3, Worked example by hand
    tok = hand()
    assert tok.basic_tokenize("The UNAFFABLE!") == ["the", "unaffable", "!"]
    assert tok.basic_tokenize("日本x y") == ["日", "本", "x", "y"]
    assert tok.basic_tokenize("Café") == ["cafe"]
    assert tok.basic_tokenize("  ") == []


def test_hand_example_encode_decode():
    # WHY: section 3: "unaffable" is un + ##aff + ##able; add_special wraps
    #      [CLS] ... [SEP]; decode glues ## pieces and removes the space before "!".
    # KIND: unit
    # CATCHES: s02, s03, s11, s12, s13
    # CHAPTER: L1.3 section 3, Worked example by hand
    tok = hand()
    assert tok.wordpiece("unaffable") == [6, 7, 8]
    assert tok.encode("The UNAFFABLE!") == [5, 6, 7, 8, 10]
    assert tok.encode("The UNAFFABLE!", add_special=True) == [2, 5, 6, 7, 8, 10, 3]
    assert tok.decode([5, 6, 7, 8, 10]) == "the unaffable!"
    assert tok.decode([2, 5, 3], skip_special=True) == "the"
    assert tok.encode("cafés") == [13, 14]


def test_whole_word_becomes_unk():
    # WHY: when any position of a word has no match, the WHOLE word is one
    #      [UNK]: "unaffablex" is [UNK], not un ##aff ##able [UNK].
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: L1.3 section 5, Pitfalls, item 1
    tok = hand()
    assert tok.wordpiece("unaffablex") == [1]
    assert tok.encode("the unaffablex") == [5, 1]


def test_greedy_is_longest_first():
    # WHY: after "un" both ##a and ##aff match; WordPiece takes the longest,
    #      and with the shortest the rest ("ffable") has no segmentation.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L1.3 section 5, Pitfalls, item 2
    assert hand().wordpiece("unaffable") == [6, 7, 8]
    assert hand().wordpiece("una") == [6, 9]


def test_continuation_needs_prefix():
    # WHY: inside a word only "##" pieces match: "unn" is un ##n, and "aff"
    #      alone is [UNK] because "##aff" cannot start a word.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: L1.3 section 2, Principles
    assert hand().wordpiece("unn") == [6, 12]
    assert hand().wordpiece("aff") == [1]


def test_max_input_chars_per_word():
    # WHY: a word longer than max_input_chars_per_word code points is [UNK]
    #      without any search; exactly the limit is still segmented.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: L1.3 section 4, The interface
    tok = hand(max_input_chars_per_word=5)
    assert tok.wordpiece("unaff") == [6, 7]
    assert tok.wordpiece("unaffa") == [1]


def test_clean_text_rules():
    # WHY: U+0000, U+FFFD, and control characters (C*) vanish, even U+000B,
    #      which is also white space; tab and other white space become U+0020.
    # KIND: boundary
    # CATCHES: s05
    # CHAPTER: L1.3 section 2, Principles
    tok = hand()
    assert tok.normalize("a\x00b�c") == "abc"
    assert tok.normalize("a\x0bb") == "ab"
    assert tok.normalize("a​b") == "ab"
    assert tok.normalize("a\tb c　d") == "a b c d"


def test_strip_accents_follows_lowercase():
    # WHY: strip_accents=None means "strip when lowercasing" (uncased BERT);
    #      a cased model keeps its accents; an explicit value wins.
    # KIND: unit
    # CATCHES: s06
    # CHAPTER: L1.3 section 5, Pitfalls, item 3
    v = {t: i for i, t in enumerate(HAND)}
    assert WordPieceTokenizer(v).normalize("Café") == "cafe"
    assert WordPieceTokenizer(v, lowercase=False).normalize("Café") == "Café"
    assert WordPieceTokenizer(v, strip_accents=False).normalize("Café") == "café"
    assert (
        WordPieceTokenizer(v, lowercase=False, strip_accents=True).normalize("Café")
        == "Cafe"
    )


def test_only_nonspacing_marks_are_stripped():
    # WHY: stripping removes category Mn after NFD (the acute accent), not
    #      spacing marks (Mc, the Devanagari vowel sign U+093E), which carry
    #      a vowel, not an accent.
    # KIND: boundary
    # CATCHES: s07
    # CHAPTER: L1.3 section 2, Principles
    tok = hand()
    assert tok.normalize("é") == "e"
    assert tok.normalize("ना") == "ना"


def test_punctuation_is_split():
    # WHY: BERT's punctuation is Unicode P* plus every ASCII punctuation
    #      character, so $ + ^ (symbols, not P*) split too.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: L1.3 section 2, Principles
    assert hand().basic_tokenize("$5+x^2«a»") == [
        "$",
        "5",
        "+",
        "x",
        "^",
        "2",
        "«",
        "a",
        "»",
    ]


def test_lowercase_per_code_point():
    # WHY: Hugging Face lowercases one code point at a time; Python's
    #      str.lower() applies the final-sigma rule and gives a different
    #      letter (and so different ids) for "ΣΑΣ".
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: L1.3 section 5, Pitfalls, item 4
    assert hand().normalize("ΣΑΣ") == "σασ"


def test_specials_match_as_written():
    # WHY: "[MASK]" in raw text is the mask id (the masked-LM objective of
    #      L6.2 writes it there); "[mask]" is plain text, because specials are
    #      matched before normalization.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: L1.3 section 4, The interface
    tok = hand()
    assert tok.encode("the [MASK]!") == [5, 4, 10]
    assert 4 not in tok.encode("the [mask]!")


def test_bert_ids_match_oracle():
    # WHY: bert-base-uncased, 300 strings: ids equal Hugging Face's exactly,
    #      through every normalizer and pre-tokenizer rule.
    # KIND: golden
    # CATCHES: s01, s07, s08, s09, s16
    # CHAPTER: L1.3 section 4, The interface
    tok = bert()
    assert tok.vocab_size == 30522 and tok.unk_id == 100
    for c in cases():
        assert tok.encode(c["text"]) == c["ids"], repr(c["text"])


def test_bert_template_matches_oracle():
    # WHY: add_special is BERT's TemplateProcessing: [CLS] ids [SEP].
    # KIND: golden
    # CATCHES: s11
    # CHAPTER: L1.3 section 4, The interface
    tok = bert()
    for c in cases()[:100]:
        assert tok.encode(c["text"], add_special=True) == c["ids_special"], repr(
            c["text"]
        )


def test_bert_decode_matches_oracle():
    # WHY: the WordPiece decoder with cleanup, and skip_special, exactly as
    #      Hugging Face decodes (the L6.2 masked-LM demo prints with it).
    # KIND: golden
    # CATCHES: s12, s13
    # CHAPTER: L1.3 section 4, The interface
    tok = bert()
    for c in cases():
        assert tok.decode(c["ids"]) == c["decoded"], repr(c["text"])
        assert tok.decode(c["ids"], skip_special=True) == c["decoded_skip"], repr(
            c["text"]
        )


def test_from_vocab_equals_tokenizer_json(tmp_path):
    # WHY: vocab.txt is the original BERT release: line n is id n, from 0.
    # KIND: golden
    # CATCHES: s14
    # CHAPTER: L1.3 section 4, The interface
    doc = json.loads((FX / "tok-bert" / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = sorted(doc["model"]["vocab"].items(), key=lambda kv: kv[1])
    (tmp_path / "vocab.txt").write_text(
        "".join(t + "\n" for t, _ in vocab), encoding="utf-8"
    )
    tok = WordPieceTokenizer.from_vocab(str(tmp_path / "vocab.txt"))
    assert tok.special_ids == bert().special_ids
    for c in cases()[:120]:
        assert tok.encode(c["text"]) == c["ids"], repr(c["text"])


def test_save_load_roundtrip(tmp_path):
    # WHY: save writes a BERT-style tokenizer.json that load (and Hugging
    #      Face) reads back to the same ids, casing and accent rules included.
    # KIND: property
    # CATCHES: s15
    # CHAPTER: L1.3 section 4, The interface
    for kw in ({}, {"lowercase": False}, {"strip_accents": True, "lowercase": False}):
        tok = hand(**kw)
        tok.save(str(tmp_path))
        back = WordPieceTokenizer.load(str(tmp_path))
        for t in ["The UNAFFABLE!", "Cafés", "the [MASK]"]:
            assert back.encode(t) == tok.encode(t), (kw, t)
            assert back.encode(t, add_special=True) == tok.encode(t, add_special=True)


def test_vocab_needs_unk():
    # WHY: [UNK] is where every unsegmentable word goes; a vocab without it
    #      cannot encode arbitrary text, so the constructor refuses it.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: L1.3 section 4, The interface
    with pytest.raises(ValueError):
        WordPieceTokenizer({"a": 0, "b": 1})
    with pytest.raises(ValueError):
        WordPieceTokenizer({"[UNK]": 0, "a": 2})
