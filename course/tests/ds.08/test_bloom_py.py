"""Course tests for ds.08, Python half: tinyllm_rs.Bloom (rust/crates/tl-py),
the filter data.03 builds its exact-dedup screen with.

Rung R0 for these course tests (your own tests for this module are rung R4,
proptest properties: section 4 of the chapter). Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/ds.08), and the chapter section.

The oracle is course/fixtures/parity/bloom.json, an independent transcription
of contracts/formats/bloom.md (course/oracle/ds.08/bloom_golden.py). The Rust
half of the tests is course/tests/rust/ds_08.rs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import tinyllm_rs

FX = Path(os.environ.get("TINYLLM_FIXTURES", ""))


def cases() -> list[dict]:
    return json.loads((FX / "parity" / "bloom.json").read_text())["cases"]


def test_py_hand_example():
    # WHY: section 3 by hand, through Python: with_rate(4, 0.1) is 20 bits
    #      and 3 hashes; after cat and dog the 35 serialized bytes are the
    #      formats/bloom.md worked example, and bird is absent.
    # KIND: unit
    # CATCHES: s01, s02, s03
    # CHAPTER: ds.08 section 3, Worked example by hand
    b = tinyllm_rs.Bloom.with_rate(4, 0.1)
    b.insert(b"cat")
    b.insert(b"dog")
    assert b.to_bytes() == bytes.fromhex(
        "544c424601000000140000000000000003000000000000000200000000000000096802"
    )
    assert b.contains(b"cat") and b.contains(b"dog") and not b.contains(b"bird")


def test_py_matches_rust_golden():
    # WHY: the binding is the same filter, byte for byte: every oracle case
    #      built through Python serializes to the oracle's bytes and answers
    #      its probes identically, so a screen built in one process is read
    #      in another (parity suite `bloom`).
    # KIND: golden
    # CATCHES: s02, s03, s10
    # CHAPTER: ds.08 section 4
    for c in cases():
        inp, out = c["input"], c["output"]
        b = tinyllm_rs.Bloom.with_rate(inp["n"], inp["p"])
        for it in inp["insert"]:
            b.insert(bytes.fromhex(it))
        assert b.to_bytes().hex() == out["bytes"], c["name"]
        got = [b.contains(bytes.fromhex(p)) for p in inp["probe"]]
        assert got == out["contains"], c["name"]
        again = tinyllm_rs.Bloom.from_bytes(b.to_bytes())
        assert again.to_bytes() == b.to_bytes()


def test_py_union_including_itself():
    # WHY: union ORs another filter in place, and `b.union(b)` is legal
    #      Python: the binding must read the other filter's bits before it
    #      borrows this one mutably, or PyO3 raises a borrow error on a
    #      perfectly valid call. The bits are unchanged; the counts add.
    # KIND: boundary
    # CATCHES: s11
    # CHAPTER: ds.08 section 5, Pitfalls
    a = tinyllm_rs.Bloom.with_rate(100, 0.01)
    b = tinyllm_rs.Bloom.with_rate(100, 0.01)
    a.insert(b"one")
    b.insert(b"two")
    a.union(b)
    assert a.contains(b"one") and a.contains(b"two")
    before = a.to_bytes()
    a.union(a)
    after = a.to_bytes()
    assert after[32:] == before[32:], "OR with itself changes no bit"
    assert int.from_bytes(after[24:32], "little") == 2 * int.from_bytes(
        before[24:32], "little"
    )
    with pytest.raises(ValueError):
        a.union(tinyllm_rs.Bloom.with_rate(1000, 0.01))


def test_py_errors_are_value_errors():
    # WHY: bad input crosses the boundary as ValueError (the contract): a
    #      negative or zero n (a Python int, never cast to an unsigned one),
    #      p outside (0, 1), and bytes that are not a version-1 filter.
    # KIND: boundary
    # CATCHES: s12, m01
    # CHAPTER: ds.08 section 4
    for n, p in [(0, 0.1), (-5, 0.1), (10, 0.0), (10, 1.0), (10, float("nan"))]:
        with pytest.raises(ValueError):
            tinyllm_rs.Bloom.with_rate(n, p)
    good = tinyllm_rs.Bloom.with_rate(4, 0.1).to_bytes()
    for bad in (
        b"",
        good[:-1],
        good + b"\x00",
        b"XLBF" + good[4:],
        good[:20] + b"\x01" + good[21:],
    ):
        with pytest.raises(ValueError):
            tinyllm_rs.Bloom.from_bytes(bad)


def test_py_no_false_negatives_on_paragraph_hashes():
    # WHY: how data.03 uses it: 2,000 paragraph texts hashed to bytes are
    #      all reported present after insertion, and a screen of 10 bits per
    #      item (p about 0.008) reports few of 2,000 fresh paragraphs.
    # KIND: property
    # CATCHES: s06
    # CHAPTER: ds.08 section 2.1
    b = tinyllm_rs.Bloom.with_rate(2000, 0.008)
    seen = [f"paragraph {i}: once upon a time".encode() for i in range(2000)]
    for x in seen:
        b.insert(x)
    assert all(b.contains(x) for x in seen)
    fresh = sum(b.contains(f"another paragraph {i}".encode()) for i in range(2000))
    assert fresh < 60, f"{fresh} of 2000 fresh paragraphs reported present"
