"""data.02 course tests: corpus.stage and corpus.filter (generator stages).

Annotated exemplars (DESIGN 5.12). The hand-labelled tables below are the
rules of the chapter written as cases; the golden test runs your whole
`filters` pipeline on 200 synthetic documents whose fate was fixed when they
were generated (course/oracle/data.02/golden_corpus.py), not by any
implementation. The laziness and memory tests feed an endless generator, so
a stage that collects its input never returns.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import tracemalloc
from pathlib import Path

import pytest
from _lib.pcg32 import PCG32

from corpus.filter import (
    GOPHER_RULES,
    STOPWORDS,
    build_filters,
    dup_ngram_char_frac,
    gopher_reasons,
    gopher_rules,
    identify_lang,
    lang_filter,
    length_filter,
    normalize_text,
    normalize_unicode,
    ppl_filter,
    repetition_filter,
)
from corpus.stage import Doc, compose

FIX = (
    Path(
        os.environ.get(
            "TINYLLM_FIXTURES", Path(__file__).resolve().parents[2] / "fixtures"
        )
    )
    / "data.02"
)
GOLDEN_FILTERS = {
    "min_chars": 20,
    "langs": ["en"],
    "lang_min_conf": 0.65,
    "repetition": {"n": 3, "max_frac": 0.3},
}

STORY = (
    "Once upon a time, Lily had a red kite that she loved very much. "
    "Every morning she walked to the park with a little dog. "
    "The dog wanted to play with the kite, but it was too high.\n\n"
    "One day the wind took the kite away, so Lily ran after it as fast as possible. "
    "A kind old man gave her a map of the park with a big cross on it. "
    "In the end everyone went home tired and very happy."
)


def docs(*texts: str) -> list[Doc]:
    return [Doc(f"t:{i}", "t", t, {}) for i, t in enumerate(texts)]


def run(stage, texts) -> list[str]:
    return [d.text for d in stage(iter(docs(*texts)))]


# --- normalization -----------------------------------------------------------------


def test_hand_example_normalize_and_identify():
    # WHY: the section 3 worked example. A BOM, a decomposed ë (e + U+0308),
    #      CRLF line ends, three blank lines, a trailing tab and space, and a
    #      BEL control normalize to "Zoë and the cat\n\nsat on a mat."; then
    #      "The cat and the dog sat in a box." has 5 English stop-word hits
    #      out of 7 (in is also Dutch, a is also Portuguese): en at 5/7.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: data.02 section 3, Worked example by hand
    raw = "\ufeffZoe\u0308 and the cat\r\n\r\n\r\n\r\nsat on a mat.\t \x07"
    assert normalize_text(raw) == "Zoë and the cat\n\nsat on a mat."
    assert identify_lang("The cat and the dog sat in a box.") == ("en", 5 / 7)


@pytest.mark.parametrize(
    "raw, want",
    [
        ("a\r\nb\rc", "a\nb\nc"),  # CRLF and lone CR
        ("Cafe\u0301", "Café"),  # NFC composes
        ("ﬁne", "ﬁne"),  # NFC, not NFKC: the fi ligature stays
        ("a\x00b\x1bc\x7fd", "abcd"),  # Cc controls removed
        ("a\tb", "a\tb"),  # a tab inside a line stays
        ("\ufeffhi", "hi"),  # BOM removed
        ("a \t\nb  ", "a\nb"),  # trailing blanks of each line
        ("a\n\n\n\n\nb", "a\n\nb"),  # 3+ line feeds become 2
        ("a\n \n \nb", "a\n\nb"),  # blank-looking lines collapse after stripping
        ("  \n hi there \n\n", "hi there"),  # outer white space
        ("x\u200dy", "x\u200dy"),  # a format character (ZWJ) is kept
    ],
)
def test_normalize_rules_one_by_one(raw, want):
    # WHY: each rule of normalize_text as a hand-labelled case (section 2.2),
    #      in particular NFC and not NFKC (NFKC rewrites ligatures, full-width
    #      letters, and superscripts, changing what the model reads), and only
    #      category Cc removed (a zero-width joiner holds emoji together).
    # KIND: unit
    # CATCHES: s01, s03, s04, m01
    assert normalize_text(raw) == want


def test_normalize_is_idempotent():
    # WHY: law: normalizing twice changes nothing. A pipeline rerun after a
    #      crash, or a stage run on already clean text, must not drift; the
    #      step order (strip trailing blanks, then collapse line feeds) is
    #      what makes this hold. 500 seeded random strings over an alphabet
    #      built from every rule's special characters.
    # KIND: property
    # CATCHES: s05
    alphabet = list("ab \t\n\r.") + [
        "\r\n",
        "\ufeff",
        "\x00",
        "\x07",
        "e\u0301",
        "\u0301",
        "é",
        "\u200d",
        "  \n",
    ]
    rng = PCG32(seed=7)
    for _ in range(500):
        s = "".join(alphabet[rng.below(len(alphabet))] for _ in range(rng.below(40)))
        once = normalize_text(s)
        assert normalize_text(once) == once, repr(s)


def test_normalize_unicode_drops_empty_and_keeps_identity():
    # WHY: a document that is only white space or controls carries nothing;
    #      every other document keeps its id, source, and meta.
    # KIND: unit
    # CATCHES: s06
    src = [
        Doc("a:0", "a", " \r\n\ufeff\x00 ", {}),
        Doc("a:1", "a", "Hi\r\n", {"url": "u#2"}),
    ]
    out = list(normalize_unicode(iter(src)))
    assert out == [Doc("a:1", "a", "Hi", {"url": "u#2"})]


def test_length_filter_bounds_are_inclusive():
    # WHY: min_chars and max_chars are both allowed lengths (code points,
    #      so "é" is 1 whether it arrived composed or not, after NFC).
    # KIND: boundary
    # CATCHES: m02
    stage = length_filter(3, 5)
    assert run(stage, ["ab", "abc", "abcde", "abcdef", "ééé"]) == [
        "abc",
        "abcde",
        "ééé",
    ]


# --- language ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, lang",
    [
        (STORY, "en"),
        ("Der kleine Hund spielt im Garten und die Katze schläft auf dem Sofa.", "de"),
        (
            "La grand-mère raconte une histoire sur un renard qui était très malin et le chat dort.",
            "fr",
        ),
        (
            "Oma vertelt een verhaal over een vos die heel slim was en het is mooi weer.",
            "nl",
        ),
        ("12345 67890 !!!", "und"),
        ("the", "und"),  # one hit is too little evidence
    ],
)
def test_identify_lang_hand_labelled(text, lang):
    # WHY: hand-labelled sentences, one per language the course cares about,
    #      plus the "und" (undetermined) rule: fewer than two stop-word hits.
    # KIND: unit
    # CATCHES: s07
    assert identify_lang(text)[0] == lang


def test_identify_lang_ties_go_to_the_first_code():
    # WHY: "de" is a stop word in es, fr, nl, and pt. Two of them give 4
    #      languages 2 hits each; the alphabetically first (es) wins, with
    #      confidence 2/8, so the answer never depends on dict order.
    # KIND: boundary
    # CATCHES: s08
    assert identify_lang("de de") == ("es", 0.25)


def test_identify_lang_uses_the_stopword_table():
    # WHY: confidence is a share of evidence: with the STOPWORDS of your
    #      module, a text of k English-only stop words and j words listed
    #      only by German scores k / (k + j). Computed from your own table,
    #      so this checks the formula, not the word lists.
    # KIND: property
    # CATCHES: s08
    en_only = sorted(
        w
        for w in STOPWORDS["en"]
        if all(w not in s for c, s in STOPWORDS.items() if c != "en")
    )
    de_only = sorted(
        w
        for w in STOPWORDS["de"]
        if all(w not in s for c, s in STOPWORDS.items() if c != "de")
    )
    for k, j in ((3, 1), (2, 2), (5, 4)):
        text = " ".join(en_only[:k] + de_only[:j])
        lang, conf = identify_lang(text)
        assert conf == pytest.approx(max(k, j) / (k + j), abs=1e-12)
        assert lang == ("de" if j >= k else "en")


def test_lang_filter_threshold_is_inclusive_and_records_meta():
    # WHY: confidence exactly at min_conf is kept (>=), the kept document
    #      carries meta["lang"] and meta["lang_conf"] for the shard's `lang`
    #      column (data.06), and the input document is not mutated.
    # KIND: boundary
    # CATCHES: s09, s10
    d = Doc("x:0", "x", "The cat and the dog sat in a box.", {"url": "u"})
    (out,) = list(lang_filter(5 / 7, ("en",))(iter([d])))
    assert out.meta == {"url": "u", "lang": "en", "lang_conf": 5 / 7}
    assert d.meta == {"url": "u"}
    assert list(lang_filter(5 / 7 + 1e-9, ("en",))(iter([d]))) == []
    assert list(lang_filter(0.0, ("de",))(iter([d]))) == []


# --- Gopher rules ---------------------------------------------------------------------

_W = "the cat sat with joy and that was fine for her "  # 11 words, 2+ stop words


def _words(n: int) -> str:
    ws = (_W * (n // 11 + 1)).split()[:n]
    return " ".join(ws)


@pytest.mark.parametrize(
    "text, reasons",
    [
        (STORY, []),
        (", ".join(_words(60).split()) + ".", []),  # "cat," holds letters: alphabetic
        (_words(49), ["words"]),
        (" ".join(["extraordinarily"] * 40 + ["the", "and"] * 10), ["mean_word_len"]),
        (_words(60) + " #a #b #c #d #e #f #g", ["symbol_ratio"]),
        (_words(30) + " ... ... ... ... ... ... ... " + _words(30), ["symbol_ratio"]),
        ("\n".join("- " + _words(11) for _ in range(10)), ["bullet_lines"]),
        (
            "\n".join(_words(11) + (" ..." if i % 2 else "") for i in range(10)),
            ["ellipsis_lines"],
        ),
        (_words(40) + " " + " ".join(["123"] * 15), ["alpha_words"]),
        (" ".join(["cat", "sat", "on", "mat", "bird"] * 12), ["stopwords"]),
        ("", ["words", "mean_word_len", "alpha_words", "stopwords"]),
    ],
)
def test_gopher_rules_hand_labelled(text, reasons):
    # WHY: one hand-labelled text per Gopher rule (section 2.4), each failing
    #      exactly that rule, plus a clean story and the empty text; the
    #      reasons come back in GOPHER_RULES order.
    # KIND: unit
    # CATCHES: s11, s12, s13, s14, s15
    assert gopher_reasons(text) == reasons


def test_gopher_boundaries_are_the_papers():
    # WHY: Rae et al. keep 50 words (not 51), a ratio of exactly 0.1, and
    #      exactly 30% ellipsis lines; off-by-one comparisons drop real text
    #      at every boundary.
    # KIND: boundary
    # CATCHES: s16, s17, s18, m03
    assert gopher_reasons(_words(50)) == []
    assert gopher_reasons(_words(50) + " #x #y #z #w #v") == []  # 5 / 55 < 0.1
    assert gopher_reasons(_words(45) + " #a #b #c #d #e") == []  # 5 / 50 == 0.1
    lines = [_words(11)] * 7 + [_words(11) + " ..."] * 3
    assert gopher_reasons("\n".join(lines)) == []  # 3 / 10 == 0.3
    bullets = ["- " + _words(11)] * 9 + [_words(11)]
    assert gopher_reasons("\n".join(bullets)) == []  # 9 / 10 == 0.9
    assert gopher_reasons(_words(60), min_words=61) == ["words"]


def test_gopher_stopwords_count_occurrences_after_stripping_punctuation():
    # WHY: "The," and "the" are the same stop word; the rule counts
    #      occurrences, so "the ... the" is two even with no other stop word.
    # KIND: boundary
    # CATCHES: s19
    base = " ".join(["cat", "sat", "on", "mat", "bird"] * 12)
    assert gopher_reasons(base + " The, (the)") == []
    assert gopher_reasons(base + " the") == ["stopwords"]


def test_gopher_rules_reject_unknown_thresholds_when_built():
    # WHY: a typo in the corpus config (`min_word` for `min_words`) must
    #      fail when the pipeline is built, not silently use the default.
    # KIND: boundary
    # CATCHES: s20
    with pytest.raises(ValueError):
        gopher_rules(min_word=10)
    assert run(gopher_rules(min_words=3), ["the cat and the dog"]) == [
        "the cat and the dog"
    ]


# --- repetition and perplexity -------------------------------------------------------


def test_dup_ngram_hand_example():
    # WHY: section 3: in "the cat sat the cat sat down" the trigram
    #      (the, cat, sat) occurs twice, so the first six words are covered:
    #      18 of 22 characters, 0.818...
    # KIND: unit
    # CATCHES: s21, s22
    assert dup_ngram_char_frac("the cat sat the cat sat down", 3) == pytest.approx(
        18 / 22, abs=1e-12
    )
    assert dup_ngram_char_frac("a b", 3) == 0.0
    assert dup_ngram_char_frac("x y z", 3) == 0.0
    assert dup_ngram_char_frac("aa b aa b aa", 2) == pytest.approx(1.0)


def test_repetition_filter_keeps_the_boundary():
    # WHY: max_frac is allowed (<=). The text below has exactly half its
    #      characters inside duplicated bigrams.
    # KIND: boundary
    # CATCHES: s23
    text = "ab cd ab cd ef gh ij kl"  # ab cd twice: 8 of 16 characters
    assert dup_ngram_char_frac(text, 2) == 0.5
    assert run(repetition_filter(2, 0.5), [text]) == [text]
    assert run(repetition_filter(2, 0.49), [text]) == []
    with pytest.raises(ValueError):
        repetition_filter(0, 0.3)


def test_ppl_filter_takes_any_perplexity_object():
    # WHY: in C1 the L2.1 n-gram model plugs in here; until then anything
    #      with perplexity(str) works. Equal to max_ppl is kept, and the
    #      score is recorded in meta for the datasheet.
    # KIND: unit
    # CATCHES: s24

    class LenLM:
        def perplexity(self, text: str) -> float:
            return float(len(text))

    out = list(ppl_filter(LenLM(), 3.0)(iter(docs("ab", "abc", "abcd"))))
    assert [(d.text, d.meta["ppl"]) for d in out] == [("ab", 2.0), ("abc", 3.0)]


# --- composition, laziness, memory ----------------------------------------------------


def test_compose_runs_left_to_right():
    # WHY: compose(a, b) is b after a; compose() passes everything through.
    #      Order matters: normalizing before the length filter counts the
    #      characters the model will see.
    # KIND: unit
    # CATCHES: s25

    def tag(c):
        def stage(ds):
            for d in ds:
                yield dataclasses.replace(d, text=d.text + c)

        return stage

    assert run(compose(tag("a"), tag("b")), ["x"]) == ["xab"]
    assert run(compose(), ["x", "y"]) == ["x", "y"]
    assert run(
        compose(normalize_unicode, length_filter(3, 9)), ["ab  \r\n", "abc\r\n"]
    ) == ["abc"]


class Endless:
    """An endless stream of good stories that counts what was pulled."""

    def __init__(self) -> None:
        self.pulled = 0

    def __iter__(self):
        for i in itertools.count():
            self.pulled += 1
            yield Doc(f"e:{i}", "e", STORY.replace("Lily", f"Lily{i % 97}"), {})


def test_stages_are_lazy():
    # WHY: a stage that calls list() or sorted() on its input never returns
    #      on an endless stream (and holds the whole corpus on a finite one).
    #      Taking 50 documents from the full filter pipeline may pull only
    #      those 50 from the source, one at a time.
    # KIND: property
    # CATCHES: s26
    src = Endless()
    pipeline = build_filters({"min_chars": 20})
    out = list(itertools.islice(pipeline(iter(src)), 50))
    assert len(out) == 50
    assert src.pulled == 50


def test_pipeline_memory_stays_flat():
    # WHY: DESIGN 4.4: the stages stream. 2,000 documents through the whole
    #      filter pipeline must peak under 0.5 MB of Python allocations (a
    #      streaming pipeline peaks near 0.03 MB); a stage that buffers its
    #      input needs about 1.25 MB here. (The design states 200k documents
    #      under 50 MB; 2k keeps this test under 1 s, and the laziness test
    #      above covers the unbounded case.)
    # KIND: property
    # CATCHES: s32
    src = Endless()
    pipeline = build_filters({"min_chars": 20})
    tracemalloc.start()
    try:
        n = sum(1 for _ in itertools.islice(pipeline(iter(src)), 2_000))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert n == 2_000
    assert peak < 512 * 1024, f"peak {peak / 1e6:.2f} MB"


# --- the configured pipeline -----------------------------------------------------------


def test_golden_200_documents():
    # WHY: conformance on 200 synthetic documents with known fates: 125 kept
    #      (15 of them only after normalization repairs CRLF, NFD, a BOM,
    #      controls, and trailing blanks) and 75 dropped, each by one named
    #      rule. The output ids, their order, their lang, and the sha256 of
    #      every output text must match, so normalization is byte-exact.
    # KIND: conformance
    # CATCHES: s27
    rows = [
        json.loads(x)
        for x in (FIX / "golden_in.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    want = [
        json.loads(x)
        for x in (FIX / "golden_out.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    out = list(
        build_filters(GOLDEN_FILTERS)(
            Doc(r["id"], r["source_id"], r["text"], {}) for r in rows
        )
    )
    why = {r["id"]: r["why"] for r in rows}
    got_ids = [d.id for d in out]
    missing = [f"{i} ({why[i]})" for i in [w["id"] for w in want] if i not in got_ids]
    extra = [f"{i} ({why[i]})" for i in got_ids if i not in {w["id"] for w in want}]
    assert not missing and not extra, (
        f"dropped but should keep: {missing[:5]}; kept but should drop: {extra[:5]}"
    )
    assert got_ids == [w["id"] for w in want]
    for d, w in zip(out, want):
        assert d.meta["lang"] == w["lang"]
        assert hashlib.sha256(d.text.encode("utf-8")).hexdigest() == w["sha256"], (
            f"{d.id}: {d.text[:80]!r}"
        )


def test_build_filters_validates_the_table():
    # WHY: the corpus config is reviewed input; an unknown key, an unknown
    #      Gopher name, or a perplexity threshold without a model is a
    #      mistake to report before any document flows.
    # KIND: boundary
    # CATCHES: s28
    with pytest.raises(ValueError):
        build_filters({"min_char": 20})
    with pytest.raises(ValueError):
        build_filters({"gopher": {"max_word": 9}})
    with pytest.raises(ValueError):
        build_filters({"ppl": {"max_ppl": 50.0}})

    class Flat:
        def perplexity(self, text):
            return 10.0

    assert (
        len(
            list(
                build_filters({"min_chars": 20, "ppl": {"max_ppl": 10.0}}, lm=Flat())(
                    iter(docs(STORY))
                )
            )
        )
        == 1
    )
    assert (
        list(
            build_filters({"min_chars": 20, "ppl": {"max_ppl": 9.0}}, lm=Flat())(
                iter(docs(STORY))
            )
        )
        == []
    )


def test_gopher_rule_names_are_the_contract():
    # WHY: the corpus config's `gopher` table and the datasheet's filter
    #      report use these names; renaming one breaks every config.
    # KIND: unit
    # CATCHES: m04
    assert GOPHER_RULES == (
        "words",
        "mean_word_len",
        "symbol_ratio",
        "bullet_lines",
        "ellipsis_lines",
        "alpha_words",
        "stopwords",
    )
