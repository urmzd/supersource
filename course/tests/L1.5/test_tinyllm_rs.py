"""Course tests for L1.5, Python half: tinyllm_rs.Bpe (rust/crates/tl-py),
diffed against your Python BPE (L1.2), the specification (P6).

Rung R0 for these course tests (your own tests for this module are rung R4,
proptest properties: section 4 of the chapter). Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/L1.5), and the chapter section.

A live differential also checks one side against the oracle (DESIGN 4.0, E),
so two equally wrong implementations still fail: the golden ids in
course/fixtures/L1.5/golden.jsonl come from Hugging Face tokenizers
(course/oracle/L1.5/golden.py). The Rust half of the tests is
course/tests/rust/l1_5.rs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import tinyllm_rs
from _lib.pcg32 import PCG32
from tinyllm.tok.bpe import BPETokenizer

FX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
ALPHABET = list("ab hgpsu'.,!?0129\t\nTS") + [
    "  ",
    "é",
    "́",
    "日本",
    "\U0001f642",
    " ",
    "　",
    "½",
    "'s",
    "'T",
    "<|endoftext|>",
]
_cache: dict[str, object] = {}


def seed() -> int:
    return int(os.environ.get("SS_SEED", "0"))


def golden(n: int) -> list[dict]:
    if "golden" not in _cache:
        with open(FX / "L1.5" / "golden.jsonl", encoding="utf-8") as f:
            _cache["golden"] = [json.loads(line) for line in f]
    return _cache["golden"][:n]  # type: ignore[index]


def rust(name: str) -> "tinyllm_rs.Bpe":
    key = "rs-" + name
    if key not in _cache:
        _cache[key] = tinyllm_rs.Bpe.from_hf_json(
            str(FX / f"tok-{name}" / "tokenizer.json")
        )
    return _cache[key]  # type: ignore[return-value]


def python(name: str) -> BPETokenizer:
    key = "py-" + name
    if key not in _cache:
        _cache[key] = BPETokenizer.from_hf_json(
            str(FX / f"tok-{name}" / "tokenizer.json")
        )
    return _cache[key]  # type: ignore[return-value]


def random_text(rng: PCG32, n: int) -> str:
    return "".join(ALPHABET[rng.below(len(ALPHABET))] for _ in range(n))


# --- the binding against your Python -------------------------------------------


@pytest.mark.parametrize("name", ["gpt2", "smollm2"])
def test_py_bpe_matches_python_l12(name: str):
    # WHY: the port is proven against YOUR Python BPE (P6), and one side is
    #      also held to the oracle, so two equally wrong implementations fail:
    #      on 1,500 golden strings Rust ids == Python ids == Hugging Face ids.
    # KIND: differential, golden
    # CATCHES: s01, s05, s06, s08
    # CHAPTER: L1.5 section 4
    rs, py = rust(name), python(name)
    for c in golden(1500):
        want = c[name]
        got = rs.encode(c["text"])
        assert got == want, f"Rust vs oracle on {c['text']!r}"
        assert py.encode(c["text"]) == got, f"your Python vs Rust on {c['text']!r}"


def test_py_random_text_matches_python():
    # WHY: seeded random text over contractions in both cases, white-space
    #      runs, marks, CJK, emoji, Unicode digits, and an added-token text:
    #      Rust and Python agree on every string, for both files.
    # KIND: differential
    # CATCHES: s05, s06, m04
    # CHAPTER: L1.5 section 4
    rng = PCG32(seed(), 5)
    for name in ("gpt2", "smollm2"):
        rs, py = rust(name), python(name)
        for _ in range(400):
            t = random_text(rng, rng.below(24))
            assert rs.encode(t) == py.encode(t), f"{name}: {t!r}"


def test_py_trained_tokenizer_loads_in_rust():
    # WHY: C1 trains its tokenizer in Python (L1.2) and serves it from Rust:
    #      a BPETokenizer trained on course/fixtures/L1.2/train.txt and saved
    #      as tokenizer.json loads in tinyllm_rs and encodes every string
    #      exactly as the Python that trained it, specials included.
    # KIND: differential
    # CATCHES: s02, s05
    # CHAPTER: L1.5 section 2.4
    import tempfile

    texts = (FX / "L1.2" / "train.txt").read_text(encoding="utf-8").splitlines()
    tok = BPETokenizer.train(
        texts, vocab_size=420, specials=["<|endoftext|>"], min_freq=2
    )
    with tempfile.TemporaryDirectory() as d:
        tok.save(d)
        rs = tinyllm_rs.Bpe.from_hf_json(str(Path(d) / "tokenizer.json"))
    assert rs.vocab_size() == tok.vocab_size
    rng = PCG32(seed(), 5)
    for t in texts[:50] + [random_text(rng, 30) for _ in range(200)]:
        assert rs.encode(t) == tok.encode(t), repr(t)
        assert rs.decode(rs.encode(t)) == t


# --- the binding's contract ----------------------------------------------------


def test_py_encode_batch_matches_encode():
    # WHY: encode_batch releases the GIL and runs on OS threads; its result
    #      is [encode(t) for t in texts] in order for any thread count, and a
    #      negative count is a ValueError, never a panic.
    # KIND: property, boundary
    # CATCHES: s12, s14
    # CHAPTER: L1.5 section 2.6
    rs = rust("gpt2")
    texts = [c["text"] for c in golden(500)]
    serial = [rs.encode(t) for t in texts]
    for threads in (0, 1, 3):
        assert rs.encode_batch(texts, threads) == serial
    assert rs.encode_batch([], 2) == []
    with pytest.raises(ValueError):
        rs.encode_batch(texts, -1)


def test_py_decode_contract():
    # WHY: decode concatenates token bytes and decodes UTF-8 with
    #      replacement: GPT-2's token 47249 alone is three bytes of an emoji,
    #      which decode to one U+FFFD. Ids outside the vocabulary, negative
    #      ones included, are ValueError (not OverflowError, not a panic).
    # KIND: boundary
    # CATCHES: s16
    # CHAPTER: L1.5 section 4
    rs = rust("gpt2")
    assert rs.decode([47249, 222]) == "\U0001f600"
    assert rs.decode([47249]) == "�"
    assert rs.decode([]) == ""
    assert rs.vocab_size() == 50257 and rust("smollm2").vocab_size() == 49152
    for bad in ([50257], [-1], [0, 2**40]):
        with pytest.raises(ValueError):
            rs.decode(bad)


def test_py_load_errors(tmp_path: Path):
    # WHY: errors cross the boundary as the exceptions the contract names:
    #      a missing file is OSError, malformed JSON and fields outside the
    #      formats/tokenizer.md subset are ValueError naming the field.
    # KIND: boundary
    # CATCHES: s13, s15
    # CHAPTER: L1.5 section 5, Pitfalls
    with pytest.raises(OSError):
        tinyllm_rs.Bpe.from_hf_json(str(tmp_path / "missing.json"))
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        tinyllm_rs.Bpe.from_hf_json(str(tmp_path / "bad.json"))
    src = (FX / "tok-smollm2" / "tokenizer.json").read_text(encoding="utf-8")
    (tmp_path / "norm.json").write_text(
        src.replace('"normalizer": null', '"normalizer": {"type": "NFC"}', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="normalizer"):
        tinyllm_rs.Bpe.from_hf_json(str(tmp_path / "norm.json"))
