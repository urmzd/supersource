"""data.03 course tests: corpus.dedup (exact paragraph dedup).

Annotated exemplars (DESIGN 5.12). The oracle for "which paragraphs are
repeats" is a Python set of exact paragraph texts, the thing the Bloom
filter replaces; the fixture corpus (course/fixtures/data.03) was built with
planted copies whose counts are known from construction. Several tests use
a deliberately tiny filter (a quarter of a byte per paragraph, false
positives near 38%) so the confirm step has real work to do.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import pytest
from _lib.pcg32 import PCG32

from corpus.dedup import STATS_KEYS, bloom_rate, exact_dedup, paragraph_hash, paragraphs
from corpus.stage import Doc, compose

FIX = (
    Path(
        os.environ.get(
            "TINYLLM_FIXTURES", Path(__file__).resolve().parents[2] / "fixtures"
        )
    )
    / "data.03"
)


def naive(docs: list[Doc]) -> list[tuple[str, str]]:
    """The obvious oracle: a set of every paragraph text seen so far."""
    seen: set[str] = set()
    out = []
    for d in docs:
        keep = []
        for p in paragraphs(d.text):
            if p not in seen:
                seen.add(p)
                keep.append(p)
        if keep:
            out.append((d.id, "\n\n".join(keep)))
    return out


def dedup(docs, b=10, **kw):
    stats: dict = {}
    out = list(exact_dedup(iter(docs), b, stats=stats, **kw))
    return out, stats


def fixture() -> list[Doc]:
    rows = [
        json.loads(x)
        for x in (FIX / "dupes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [Doc(r["id"], r["source_id"], r["text"], {"why": r["why"]}) for r in rows]


def test_hand_example_three_documents(tmp_path):
    # WHY: the section 3 worked example. A = "P1 / P2", B = "P2 / P3",
    #      C = "P1": B's P2 and all of C repeat earlier paragraphs. A comes out
    #      unchanged, B as "P3", C is dropped: 5 paragraphs in, 2 repeats
    #      removed, 1 document dropped.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: data.03 section 3, Worked example by hand
    p1, p2, p3 = "The cat sat.", "The dog ran.", "A bird sang."
    docs = [
        Doc("s:0", "s", f"{p1}\n\n{p2}", {"url": "u#1"}),
        Doc("s:1", "s", f"{p2}\n\n{p3}", {"url": "u#4"}),
        Doc("s:2", "s", p1, {"url": "u#7"}),
    ]
    out, stats = dedup(docs, spool_dir=tmp_path)
    assert out == [docs[0], Doc("s:1", "s", p3, {"url": "u#4"})]
    assert (stats["docs_in"], stats["paragraphs"], stats["duplicate_paragraphs"]) == (
        3,
        5,
        2,
    )
    assert (stats["docs_dropped"], stats["docs_out"]) == (1, 2)


def test_fixture_duplicate_counts():
    # WHY: conformance on 200 documents with planted whole-document copies
    #      and reused paragraphs; the counts and the kept texts were fixed
    #      when the corpus was generated, not by any implementation.
    # KIND: conformance
    # CATCHES: s03, s05, m01, m03
    exp = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
    out, stats = dedup(fixture())
    assert stats["docs_in"] == exp["docs"]
    assert stats["paragraphs"] == exp["paragraphs"]
    assert stats["duplicate_paragraphs"] == exp["duplicate_paragraphs"]
    assert stats["docs_dropped"] == exp["dropped_docs"]
    got = [(d.id, hashlib.sha256(d.text.encode("utf-8")).hexdigest()) for d in out]
    assert got == [(k["id"], k["sha256"]) for k in exp["kept"]]


def test_zero_false_drops_with_a_tiny_filter():
    # WHY: the invariant of the module: a Bloom filter answers "maybe", so a
    #      positive is only a candidate. With a quarter byte per paragraph
    #      the filter is wrong about a third of the time, and the output must
    #      still equal the exact set-based oracle: every false positive was
    #      confirmed away by the sort-merge, none dropped a paragraph.
    # KIND: property
    # CATCHES: s04
    docs = fixture()
    out, stats = dedup(docs, 0.25)
    assert [(d.id, d.text) for d in out] == naive(docs)
    assert stats["false_positives"] > 50, (
        "the tiny filter should have produced false positives"
    )
    # Every paragraph that occurs twice is a candidate (no false negatives);
    # a paragraph that occurs once is one only by a false positive.
    counts = Counter(p for d in docs for p in paragraphs(d.text))
    repeated = sum(1 for c in counts.values() if c >= 2)
    assert stats["candidates"] == repeated + stats["false_positives"]


def test_random_corpora_match_the_set_oracle():
    # WHY: law over 40 seeded corpora drawn from a small paragraph pool (so
    #      repeats are common), at three filter sizes: the output always
    #      equals the set-based oracle, documents and order included.
    # KIND: property
    # CATCHES: s06
    pool = [
        f"Paragraph {i} about the {w}."
        for i, w in enumerate("cat dog fox owl bee ant eel yak".split())
    ]
    for seed in range(40):
        rng = PCG32(seed=seed)
        docs = []
        for i in range(rng.below(12) + 1):
            k = rng.below(4)
            docs.append(
                Doc(
                    f"r:{i}",
                    "r",
                    "\n\n".join(pool[rng.below(len(pool))] for _ in range(k)),
                    {},
                )
            )
        for b in (0.1, 1, 10):
            out, _ = dedup(docs, b)
            assert [(d.id, d.text) for d in out] == naive(docs), (seed, b)


def test_paragraph_split_rules():
    # WHY: what counts as one paragraph decides what counts as a repeat: a
    #      blank line may hold spaces or tabs, a single line break stays
    #      inside a paragraph, outer white space is not part of it, and empty
    #      pieces are not paragraphs.
    # KIND: unit
    # CATCHES: s07
    text = "  One line\nsame paragraph.  \n \t \nTwo.\n\n\n\nThree\n"
    assert paragraphs(text) == ["One line\nsame paragraph.", "Two.", "Three"]
    assert paragraphs("") == [] and paragraphs("\n\n  \n") == []
    assert paragraph_hash("Two.") == hashlib.sha256(b"Two.").digest()


def test_a_repeat_inside_one_document_is_removed():
    # WHY: a page that repeats its own footer is a repeat too: the second
    #      occurrence goes, the first stays, and the document survives.
    # KIND: boundary
    # CATCHES: s08
    out, stats = dedup([Doc("d:0", "d", "Hello.\n\nFooter.\n\nBody.\n\nFooter.", {})])
    assert [d.text for d in out] == ["Hello.\n\nFooter.\n\nBody."]
    assert stats["duplicate_paragraphs"] == 1


def test_untouched_documents_keep_their_exact_text():
    # WHY: dedup must not rewrite text it does not dedup: a document that
    #      loses nothing comes back equal (text, line breaks, meta), so the
    #      shard's sha256 of the text matches the filtered stage's output.
    # KIND: unit
    # CATCHES: s09
    d = Doc(
        "d:0",
        "d",
        "First line\nsecond line\n\n\nNext paragraph.",
        {"lang": "en", "lang_conf": 0.9},
    )
    out, _ = dedup([d])
    assert out == [d]


def test_bloom_is_sized_from_bytes_per_item():
    # WHY: bloom_bytes_per_item is the memory knob of the corpus config: B
    #      bytes per paragraph means p = exp(-8 B (ln 2)^2) and, by the
    #      formats/bloom.md sizing, m = ceil(-N ln p / (ln 2)^2) bits, about
    #      8 B N. Sizing for documents instead of paragraphs, or for a fixed
    #      count, breaks the false-positive budget on a large corpus.
    # KIND: unit
    # CATCHES: s10, s11
    assert bloom_rate(1) == pytest.approx(math.exp(-8 * math.log(2) ** 2), rel=1e-12)
    assert bloom_rate(10) == pytest.approx(2.0e-17, rel=0.05)
    docs = fixture()
    for b in (1, 10):
        _, stats = dedup(docs, b)
        n = stats["paragraphs"]
        m = math.ceil(-n * math.log(bloom_rate(b)) / math.log(2) ** 2)
        assert stats["bloom_bits"] == m
        assert abs(m - 8 * b * n) <= 1
    for bad in (0, -1):
        with pytest.raises(ValueError):
            bloom_rate(bad)


def test_stats_name_every_key():
    # WHY: the stats feed the corpus manifest (dedup.exact_dropped) and the
    #      datasheet; every key of STATS_KEYS is filled once the output is
    #      exhausted, also for an empty input.
    # KIND: unit
    # CATCHES: s12, m02
    _, stats = dedup([])
    assert set(stats) == set(STATS_KEYS)
    assert stats["docs_in"] == stats["paragraphs"] == stats["docs_out"] == 0


def test_the_spool_is_removed(tmp_path):
    # WHY: the input is spooled to disk so memory stays flat; the spool is
    #      a copy of the corpus and must not outlive the stage, whether the
    #      output is read to the end or abandoned after one document.
    # KIND: unit
    # CATCHES: s13
    docs = fixture()[:20]
    list(exact_dedup(iter(docs), spool_dir=tmp_path))
    assert list(tmp_path.iterdir()) == []
    gen = exact_dedup(iter(docs), spool_dir=tmp_path)
    next(gen)
    assert list(tmp_path.iterdir()) != []
    gen.close()
    assert list(tmp_path.iterdir()) == []


def test_exact_dedup_is_a_stage():
    # WHY: with its options bound it composes with the data.02 stages; the
    #      pipeline order is filter, then dedup.
    # KIND: unit
    # CATCHES: s14
    stage = compose(functools.partial(exact_dedup, bloom_bytes_per_item=2))
    docs = [
        Doc("a:0", "a", "x\n\ny", {}),
        Doc("a:1", "a", "y\n\nz\n\nv", {}),
        Doc("a:2", "a", "w", {}),
    ]
    assert [(d.id, d.text) for d in stage(iter(docs))] == [
        ("a:0", "x\n\ny"),
        ("a:1", "z\n\nv"),
        ("a:2", "w"),
    ]
