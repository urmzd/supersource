"""Course tests for data.05: PII scrub with typed placeholders and audit
spans (corpus/pii.py).

Rung R0 for these course tests (your own tests for this module are rung R3:
red then green, section 4 of the chapter). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/data.05), and the chapter section it comes from.

Fixture (course/oracle/data.05/make_labelled.py): labelled.jsonl, 396
sentences with labelled spans of every kind (example domains, 555-01xx
numbers, documentation IP blocks, random Luhn-valid cards, random keys) and
lookalike sentences with no spans.

The chapter's worked example (section 3):

    "Mail ana@example.com or call (212) 555-0123; card 4111 1111 1111 1111."
    email [5, 20), phone [29, 43), card [50, 69)   (offsets in this text)
    Luhn of 4111111111111111: doubled 4 -> 8 and seven 1 -> 2 (22), plus
    eight undoubled 1s (8): 30, a multiple of 10, so valid
    -> "Mail <EMAIL> or call <PHONE>; card <CARD>."
"""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter
from pathlib import Path

import pytest
from corpus.minhash import near_dedup
from corpus.pii import (
    KINDS,
    MANIFEST_KEYS,
    PLACEHOLDERS,
    PiiSpan,
    audit_line,
    detect,
    luhn_ok,
    redact,
    scrub,
    scrub_stage,
)
from corpus.stage import Doc

FX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "data.05"
HAND = "Mail ana@example.com or call (212) 555-0123; card 4111 1111 1111 1111."


def doc(i: str, text: str, source: str = "s", **meta) -> Doc:
    return Doc(id=i, source_id=source, text=text, meta=meta)


def kinds(text: str) -> list[tuple[str, str]]:
    return [(s.kind, text[s.start : s.end]) for s in detect(text)]


# --- the worked example ------------------------------------------------------------


def test_hand_example():
    # WHY: section 3 by hand: three spans with their offsets in the
    #      original text, the Luhn sum 30, and the redacted sentence with
    #      typed placeholders. The offsets index the ORIGINAL text, so an
    #      audit can point at what was removed without keeping it.
    # KIND: unit
    # CATCHES: s01, s10, m09
    # CHAPTER: data.05 section 3, Worked example by hand
    spans = detect(HAND)
    assert spans == [
        PiiSpan("email", 5, 20),
        PiiSpan("phone", 29, 43),
        PiiSpan("card", 50, 69),
    ]
    assert luhn_ok("4111111111111111")
    assert redact(HAND, spans) == "Mail <EMAIL> or call <PHONE>; card <CARD>."
    out, got = scrub(doc("d", HAND))
    assert got == spans and out.text == "Mail <EMAIL> or call <PHONE>; card <CARD>."
    assert out.meta["pii_redactions"] == 3
    assert out.meta["pii"] == {"email": 1, "phone": 1, "card": 1, "ip": 0, "key": 0}


def test_luhn():
    # WHY: the checksum is what separates a card number from any 16-digit
    #      id: the doubling starts at the second digit from the RIGHT, and
    #      a doubled digit above 9 loses 9. 79927398713 is the textbook
    #      valid number; changing its last digit breaks it.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: data.05 section 2, Principles
    assert luhn_ok("79927398713") and not luhn_ok("79927398714")
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert luhn_ok("378282246310005")  # 15 digits: odd length
    assert (
        luhn_ok("0")
        and not luhn_ok("")
        and not luhn_ok("4111-1111")
        and not luhn_ok("４１")
    )


# --- the labelled fixture ----------------------------------------------------------


def test_labelled_fixture():
    # WHY: the design's bar: recall >= 0.98 on email and card, precision
    #      >= 0.95 over all kinds, scored span by span (kind and exact
    #      offsets) on sentences whose labels come from how they were built.
    # KIND: conformance
    # CATCHES: s01, s02, m02, m03
    # CHAPTER: data.05 section 4, What the tests check
    with open(FX / "labelled.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    tp, fn, fp = Counter(), Counter(), 0
    for r in rows:
        want = {(s["kind"], s["start"], s["end"]) for s in r["spans"]}
        got = {(s.kind, s.start, s.end) for s in detect(r["text"])}
        for k, *_ in want & got:
            tp[k] += 1
        for k, *_ in want - got:
            fn[k] += 1
        fp += len(got - want)
    for k in ("email", "card"):
        recall = tp[k] / (tp[k] + fn[k])
        assert recall >= 0.98, f"{k} recall {recall:.3f}"
    precision = sum(tp.values()) / (sum(tp.values()) + fp)
    assert precision >= 0.95, f"precision {precision:.3f}"


def test_lookalikes_are_not_pii():
    # WHY: a scrubber that fires on version strings, ISBNs, dates, commit
    #      hashes, C++ scopes, Python slices, or our own C symbols damages
    #      exactly the technical text a code-aware model needs. Each line
    #      here looks like one kind and is none.
    # KIND: boundary
    # CATCHES: s04, s05, s07, m06
    # CHAPTER: data.05 section 5, Pitfalls, item 1
    for text in [
        "ISBN 978-0-306-40615-7 and ISBN 0-306-40615-2",
        "version 1.2.3, v1.2.3.4, 1.2.3.4.5, react@18.3.1",
        "on 2024-05-20 at 12:30:45, ratio 3:2:1",
        "id 123e4567-e89b-12d3-a456-426614174000, commit 9fceb02d0ae598e95dc970b74767f19372d61af8",
        "a[1::2], ::1, std::vector, Foo::Bar",
        "root@localhost, dial 555-0123",
        "order 4111111111111112, millis 1696867200123, 9780306406157",
        "tl_matmul_f32, tl_kv_pool_create, sk-learn, sk-abc",
    ]:
        assert detect(text) == [], text


def test_card_needs_a_network_prefix():
    # WHY: about one random number in ten passes Luhn. The first digit
    #      (2 to 6 for card networks) rules out ISBN-13s (978, 979) and
    #      millisecond timestamps (1...); the same Luhn-valid body behind a
    #      4 is a card.
    # KIND: boundary
    # CATCHES: s02, s03, m03
    # CHAPTER: data.05 section 5, Pitfalls, item 2
    assert luhn_ok("9780306406156") and detect("isbn 9780306406156") == []
    assert luhn_ok("4780306406151") and kinds("card 4780306406151") == [
        ("card", "4780306406151")
    ]
    assert kinds("amex 3782-822463-10005") == [("card", "3782-822463-10005")]


# --- detectors ---------------------------------------------------------------------


def test_phones():
    # WHY: North American numbers in four layouts and international ones
    #      with a +country code are phones; a 7-digit local number is too
    #      ambiguous to call one, and a digit run longer than 15 is not.
    # KIND: unit
    # CATCHES: s06, m04
    # CHAPTER: data.05 section 2, Principles
    for p in [
        "(212) 555-0123",
        "212-555-0123",
        "212.555.0123",
        "+1 212 555 0123",
        "+44 20 7946 0958",
        "+442079460958",
    ]:
        assert kinds(f"call {p} now") == [("phone", p)], p
    assert detect("call 555-0123 or +1234567890123456789") == []


def test_ips():
    # WHY: IPv4 octets stop at 255 and the address cannot be part of a
    #      longer dotted run; IPv6 candidates must parse and carry real
    #      groups, so ::1 (loopback, not personal) and slices like 1::2
    #      stay.
    # KIND: unit
    # CATCHES: s05, s07, m05
    # CHAPTER: data.05 section 2, Principles
    assert kinds("from 203.0.113.9.") == [("ip", "203.0.113.9")]
    assert detect("from 256.1.1.1 and 1.2.3.4.5") == []
    for v6 in [
        "2001:db8::8a2e:370:7334",
        "fe80::1ff:fe23:4567:890a",
        "2001:db8:0:0:1:0:0:1",
    ]:
        assert kinds(f"host {v6} up") == [("ip", v6)], v6
    assert detect("x[1::2] and ::1 and 12:30:45") == []


def test_keys():
    # WHY: leaked credentials are the costliest PII in a code corpus. Each
    #      provider shape is caught whole (prefix included), and a short or
    #      embedded lookalike is not.
    # KIND: unit
    # CATCHES: s08, m06
    # CHAPTER: data.05 section 2, Principles
    keys = [
        "sk-" + "a1B2" * 6,
        "sk-proj-" + "Zz9_" * 8,
        "AKIA" + "ABCDEFGHIJ123456",
        "ghp_" + "x" * 36,
        "xoxb-123456789012-abcdefABCDEF",
        "AIza" + "S" * 35,
        "tl_k3y9_" + "Q" * 24,
    ]
    for k in keys:
        assert kinds(f"key={k};") == [("key", k)], k
    assert detect("key=sk-short; AKIA123; xghp_" + "x" * 36) == []


def test_overlaps_resolve_left_to_right():
    # WHY: candidates overlap ("1.2.3.4@example.com" is an IPv4 and an
    #      email; "+4222222222222" is a phone and a card). The one that
    #      starts first wins, then the longer one, and spans never overlap,
    #      or redact would splice two placeholders into one region.
    # KIND: boundary
    # CATCHES: s09, m07
    # CHAPTER: data.05 section 5, Pitfalls, item 3
    assert kinds("to 1.2.3.4@example.com!") == [("email", "1.2.3.4@example.com")]
    assert kinds("dial +4222222222222 now") == [("phone", "+4222222222222")]
    spans = detect("a@example.com b@example.com 203.0.113.1")
    assert [s.start for s in spans] == sorted(s.start for s in spans)
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:]))


def test_offsets_are_in_the_original_text():
    # WHY: a placeholder has a different length than what it replaces. The
    #      audit offsets must index the original text (so they can be
    #      checked against it), not the partly redacted one.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: data.05 section 5, Pitfalls, item 4
    text = "x a.long.address@example.com y 203.0.113.77 z sk-" + "q" * 30
    out, spans = scrub(doc("d", text))
    assert [text[s.start : s.end] for s in spans] == [
        "a.long.address@example.com",
        "203.0.113.77",
        "sk-" + "q" * 30,
    ]
    assert out.text == "x <EMAIL> y <IP> z <KEY>"


def test_redact_rejects_bad_spans():
    # WHY: redact trusts nothing it is given: unsorted, overlapping, empty,
    #      or out-of-range spans would corrupt the text silently.
    # KIND: boundary
    # CATCHES: m08
    # CHAPTER: data.05 section 4, The interface
    t = "0123456789"
    with pytest.raises(ValueError):
        redact(t, [PiiSpan("ip", 5, 7), PiiSpan("ip", 1, 3)])
    with pytest.raises(ValueError):
        redact(t, [PiiSpan("ip", 1, 5), PiiSpan("ip", 4, 6)])
    with pytest.raises(ValueError):
        redact(t, [PiiSpan("ip", 8, 11)])
    with pytest.raises(ValueError):
        redact(t, [PiiSpan("ip", 3, 3)])
    assert redact(t, []) == t
    assert PLACEHOLDERS == {k: "<" + k.upper() + ">" for k in KINDS}
    assert MANIFEST_KEYS == {
        "email": "emails",
        "phone": "phones",
        "card": "cards",
        "ip": "ips",
        "key": "keys",
    }


def test_scrub_keeps_the_document_and_upstream_meta():
    # WHY: scrub runs after near dedup (data.04): the id, the source, and
    #      every meta key already set (minhash_cluster, url, license) must
    #      survive, and the two pii keys are added with a count for every
    #      kind (zeros included) for the manifest (data.06).
    # KIND: unit
    # CATCHES: s11, m01, m09
    # CHAPTER: data.05 section 6, Where it's used next
    docs = [
        doc(
            "a",
            "write to ann@example.org today please",
            url="u1",
            license_spdx="CC0-1.0",
        )
    ]
    tagged = list(near_dedup(docs, drop=False))
    out, _ = scrub(tagged[0])
    assert (out.id, out.source_id) == ("a", "s")
    assert out.meta["minhash_cluster"] == "a" and out.meta["url"] == "u1"
    assert out.meta["pii"] == {"email": 1, "phone": 0, "card": 0, "ip": 0, "key": 0}
    two, _ = scrub(doc("c", "ann@example.org and bo@example.org"))
    assert two.meta["pii"]["email"] == 2 and two.meta["pii_redactions"] == 2
    clean, spans = scrub(doc("b", "nothing here"))
    assert (
        spans == []
        and clean.text == "nothing here"
        and clean.meta["pii_redactions"] == 0
    )


def test_scrub_is_idempotent():
    # WHY: placeholders never match a detector, so a second scrub (a
    #      retried activity, a re-run stage) changes nothing, and its count
    #      is this pass's count (0), not a running total.
    # KIND: property
    # CATCHES: m10
    # CHAPTER: data.05 section 2, Principles
    with open(FX / "labelled.jsonl", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f][:120]
    for i, t in enumerate(texts):
        once, _ = scrub(doc(str(i), t))
        twice, spans = scrub(once)
        assert twice.text == once.text and spans == []
        assert twice.meta["pii_redactions"] == 0


def test_policies():
    # WHY: the ledger's pii_policy is per source (ethics.02): "scrub"
    #      redacts, "drop" removes a document with any PII and keeps clean
    #      ones, "none" (a source known to be PII-free) skips detection.
    #      Every yielded document carries the counts; an unknown policy is
    #      an error, not a silent scrub.
    # KIND: unit
    # CATCHES: s12, s13, s14, m11
    # CHAPTER: data.05 section 5, Pitfalls, item 5
    docs = [
        doc("1", "mail a@example.com", "web"),
        doc("2", "clean text", "web"),
        doc("3", "mail b@example.com", "forum"),
        doc("4", "mail c@example.com", "books"),
    ]
    seen = []
    out = list(
        scrub_stage(
            docs,
            policy={"forum": "drop", "books": "none"},
            audit=lambda i, sp: seen.append((i, len(sp))),
        )
    )
    assert [(d.id, d.text, d.meta["pii_redactions"]) for d in out] == [
        ("1", "mail <EMAIL>", 1),
        ("2", "clean text", 0),
        ("4", "mail c@example.com", 0),
    ]
    assert seen == [("1", 1), ("3", 1)]
    with pytest.raises(ValueError):
        list(scrub_stage([docs[0]], policy={"web": "mask"}))


def test_audit_holds_spans_not_text():
    # WHY: an audit log that copies the email it redacted is a second copy
    #      of the PII. audit_line records the document id, kinds, and
    #      offsets only.
    # KIND: unit
    # CATCHES: s01, m12
    # CHAPTER: data.05 section 5, Pitfalls, item 6
    line = audit_line("web:7", detect(HAND))
    assert "\n" not in line
    assert json.loads(line) == {
        "id": "web:7",
        "spans": [
            {"kind": "email", "start": 5, "end": 20},
            {"kind": "phone", "start": 29, "end": 43},
            {"kind": "card", "start": 50, "end": 69},
        ],
    }
    assert "example.com" not in line and "4111" not in line


def test_stage_streams():
    # WHY: the scrub runs on the whole corpus and must pull one document
    #      at a time: islice takes 3 from a stream that fails past 30.
    # KIND: property
    # CATCHES: s15, m11
    # CHAPTER: data.05 section 4, The interface
    def endless():
        for i in itertools.count():
            assert i < 30, "scrub_stage read far ahead"
            yield doc(str(i), f"row {i} at r{i}@example.com")

    out = list(itertools.islice(scrub_stage(endless()), 3))
    assert [d.text for d in out] == [
        "row 0 at <EMAIL>",
        "row 1 at <EMAIL>",
        "row 2 at <EMAIL>",
    ]
