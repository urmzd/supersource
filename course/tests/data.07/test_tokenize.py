"""Course tests for data.07: tokenize the shards and pack llm.c .bin token
streams (corpus/tokenize.py).

Rung R0 for these course tests (your own tests for this module are rung R4,
properties: section 4 of the chapter). Each test names why it exists (WHY),
what kind of check it is (KIND), the planted bugs it kills (CATCHES, mutants
in course/mutants/data.07), and the chapter section it comes from.

The corpora are written with data.06's write_shards. The GPT-2 oracle is
course/fixtures/tok-gpt2 (tokenizer.json and the ids Hugging Face
`tokenizers` gives for 300 strings, L1.2's fixture), read here through the
Python BPE. Headers are parsed with struct, not with your read_bin.

The chapter's worked example (section 3), byte tokenizer, val_permille 100:

    a:0 "Hi"  sha256 mod 1000 = 147 -> train   ids 72 105
    a:1 "é"   sha256 mod 1000 =  51 -> val     ids 195 169
    a:2 "ok"  sha256 mod 1000 = 814 -> train   ids 111 107
    train-00000.bin: 1032 bytes; header 88 d8 34 01 | 01 00 00 00 |
    04 00 00 00 | 00 01 00 00 | 1008 zero bytes; ids 48 00 69 00 6f 00 6b 00
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

import pytest
from corpus.shard import read_shards, write_shards
from corpus.stage import Doc
from corpus.tokenize import MAGIC, read_bin, tokenize_shards, write_bin
from schema_lite import errors
from tinyllm.tok.metrics import bytes_per_token

FX = Path(os.environ.get("TINYLLM_FIXTURES", ""))
CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "formats"
GPT2 = FX / "tok-gpt2" / "tokenizer.json"


def doc(i: str, text: str) -> Doc:
    return Doc(
        id=i, source_id="a", text=text, meta={"url": "u", "license_spdx": "CC0-1.0"}
    )


def corpus(
    tmp: Path, texts: list[str], val_permille: int = 100, name: str = "c"
) -> Path:
    write_shards(
        [doc(f"a:{i:04d}", t) for i, t in enumerate(texts)],
        tmp / name,
        dataset="d",
        version="v1",
        shard_rows=64,
        val_permille=val_permille,
    )
    return tmp / name / "_MANIFEST.json"


def header(path: Path) -> tuple[int, ...]:
    return struct.unpack("<256i", path.read_bytes()[:1024])[:4]


def ids_of(path: Path) -> list[int]:
    raw = path.read_bytes()
    width = 2 if header(path)[1] == 1 else 4
    return list(
        struct.unpack(
            f"<{(len(raw) - 1024) // width}{'H' if width == 2 else 'I'}", raw[1024:]
        )
    )


def gpt2_cases() -> list[dict]:
    with open(FX / "tok-gpt2" / "cases.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- the worked example ------------------------------------------------------------


def test_hand_example(tmp_path):
    # WHY: section 3 by hand: the byte tokenizer's ids are the UTF-8 bytes,
    #      train and val come from the shard's split column, and the file
    #      is the llm.c layout byte for byte.
    # KIND: unit
    # CATCHES: s01, s03, s06, s08, m01, m08
    # CHAPTER: data.07 section 3, Worked example by hand
    write_shards(
        [doc("a:0", "Hi"), doc("a:1", "é"), doc("a:2", "ok")],
        tmp_path / "c",
        dataset="d",
        version="v1",
        shard_rows=4,
        val_permille=100,
    )
    tm = tokenize_shards(
        tmp_path / "c" / "_MANIFEST.json", None, tmp_path / "t", tokenizer_id="bytes"
    )
    train = (tmp_path / "t" / "train-00000.bin").read_bytes()
    assert len(train) == 1032
    assert train[:16] == bytes.fromhex("88d83401010000000400000000010000")
    assert train[16:1024] == b"\0" * 1008
    assert train[1024:] == bytes.fromhex("480069006f006b00")
    assert ids_of(tmp_path / "t" / "val-00000.bin") == [195, 169]
    assert (tm.vocab_size, tm.doc_sep_id, tm.val_bytes_per_token) == (256, None, 1.0)
    assert [(f["name"], f["n_tokens"], f["n_docs"]) for f in tm.files] == [
        ("train-00000.bin", 4, 2),
        ("val-00000.bin", 2, 1),
    ]


def test_write_bin_layout(tmp_path):
    # WHY: formats/tokens-bin.md's own example: ids [1, 2, 3] with vocab
    #      256 are 1030 bytes; a vocabulary above 65536 needs version 2 and
    #      4-byte ids, or ids above 65535 wrap.
    # KIND: unit
    # CATCHES: s01, s04, m02
    # CHAPTER: data.07 section 2, Principles
    write_bin(tmp_path / "a.bin", [1, 2, 3], 256)
    raw = (tmp_path / "a.bin").read_bytes()
    assert raw[:16] == bytes.fromhex("88d83401010000000300000000010000") and raw[
        1024:
    ] == bytes.fromhex("010002000300")
    write_bin(tmp_path / "b.bin", [0, 65536, 69999], 70000)
    assert header(tmp_path / "b.bin") == (MAGIC, 2, 3, 70000)
    assert ids_of(tmp_path / "b.bin") == [0, 65536, 69999]
    write_bin(tmp_path / "c.bin", [65535], 65536)
    assert header(tmp_path / "c.bin")[1] == 1
    h, ids = read_bin(tmp_path / "b.bin")
    assert h == {"version": 2, "n_tokens": 3, "vocab_size": 70000} and ids.tolist() == [
        0,
        65536,
        69999,
    ]


def test_write_bin_rejects_bad_ids(tmp_path):
    # WHY: a uint16 file silently wraps 65536 to 0 and -1 to 65535; an id
    #      outside the vocabulary is a bug upstream, and the writer is the
    #      last place to catch it before training reads garbage.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: data.07 section 5, Pitfalls, item 1
    for ids, v in [([256], 256), ([-1], 256), ([1], 0)]:
        with pytest.raises(ValueError):
            write_bin(tmp_path / "x.bin", ids, v)


def test_read_bin_rejects_bad_files(tmp_path):
    # WHY: the reader is how you check a file you did not write: a wrong
    #      magic, an unknown version, or a size that disagrees with n_tokens
    #      is an error, never a short or misread stream.
    # KIND: boundary
    # CATCHES: m04, m05
    # CHAPTER: data.07 section 4, The interface
    write_bin(tmp_path / "a.bin", [1, 2, 3], 256)
    good = (tmp_path / "a.bin").read_bytes()
    for bad in [
        b"\0" + good[1:],
        good[:4] + struct.pack("<i", 3) + good[8:],
        good[:-1],
        good + b"\0\0",
        good[:100],
    ]:
        (tmp_path / "x.bin").write_bytes(bad)
        with pytest.raises(ValueError):
            read_bin(tmp_path / "x.bin")


# --- real tokenizers -----------------------------------------------------------------


def test_gpt2_golden(tmp_path):
    # WHY: through L1.5's Rust BPE the stream must be exactly the Hugging
    #      Face ids of each document, each preceded by the separator
    #      <|endoftext|> (50256), documents in manifest order. GPT-2's 50257
    #      ids fit uint16, so the file is version 1.
    # KIND: golden
    # CATCHES: s01, s02, s05, s06, s14, m06, m08
    # CHAPTER: data.07 section 4, What the tests check
    cases = gpt2_cases()
    m = corpus(tmp_path, [c["text"] for c in cases], val_permille=0)
    tm = tokenize_shards(
        m, GPT2, tmp_path / "t", tokenizer_id="gpt2", doc_sep_id=50256
    )
    want = []
    for c in cases:  # ids a:0000 .. a:0299 sort in input order
        want += [50256] + c["ids"]
    assert header(tmp_path / "t" / "train-00000.bin") == (MAGIC, 1, len(want), 50257)
    assert ids_of(tmp_path / "t" / "train-00000.bin") == want
    assert tm.tokenizer_sha256 == hashlib.sha256(GPT2.read_bytes()).hexdigest()


def test_separator_from_generation_config(tmp_path):
    # WHY: the separator is a property of the tokenizer: eos_token_id of
    #      the generation_config.json next to tokenizer.json, when there is
    #      one. An explicit doc_sep_id wins; with neither there is none.
    # KIND: unit
    # CATCHES: s02, s06, s07, s14, s15, m06, m07
    # CHAPTER: data.07 section 2, Principles
    d = tmp_path / "tok"
    d.mkdir()
    shutil.copy(GPT2, d / "tokenizer.json")
    m = corpus(tmp_path, ["Hello world", "Hi"], val_permille=0)
    assert (
        tokenize_shards(
            m, d / "tokenizer.json", tmp_path / "a", tokenizer_id="g"
        ).doc_sep_id
        is None
    )
    (d / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 50256, "bos_token_id": 50256})
    )
    tm = tokenize_shards(m, d / "tokenizer.json", tmp_path / "b", tokenizer_id="g")
    assert tm.doc_sep_id == 50256
    assert ids_of(tmp_path / "b" / "train-00000.bin") == [
        50256,
        15496,
        995,
        50256,
        17250,
    ]
    assert (
        tokenize_shards(
            m, d / "tokenizer.json", tmp_path / "c", tokenizer_id="g", doc_sep_id=0
        ).doc_sep_id
        == 0
    )
    with pytest.raises(ValueError):
        tokenize_shards(
            m, d / "tokenizer.json", tmp_path / "e", tokenizer_id="g", doc_sep_id=50257
        )


# --- properties ----------------------------------------------------------------------


def test_tokens_are_conserved_and_splits_never_mix(tmp_path):
    # WHY: every token the tokenizer produces lands in exactly one file of
    #      its document's split: per split, n_tokens equals the sum of the
    #      documents' token counts plus one separator each, n_docs equals
    #      the split's documents, and the train stream is exactly the train
    #      documents in order (so no val text is trained on).
    # KIND: property
    # CATCHES: s02, s03, s05, s06, s08, s14, m06, m08
    # CHAPTER: data.07 section 4, What the tests check
    cases = gpt2_cases()
    m = corpus(tmp_path, [c["text"] for c in cases], val_permille=300)
    tm = tokenize_shards(
        m,
        GPT2,
        tmp_path / "t",
        tokenizer_id="gpt2",
        doc_sep_id=50256,
        max_file_tokens=700,
    )
    by_text = {c["text"]: c["ids"] for c in cases}
    for split in ("train", "val"):
        docs = list(read_shards(m.parent, split=split))
        files = [f for f in tm.files if f["split"] == split]
        assert sum(f["n_docs"] for f in files) == len(docs)
        assert sum(f["n_tokens"] for f in files) == sum(
            len(by_text[d.text]) + 1 for d in docs
        )
        stream = []
        for f in files:
            stream += ids_of(tmp_path / "t" / f["name"])
        want = []
        for d in docs:
            want += [50256] + by_text[d.text]
        assert stream == want


def test_files_hold_whole_documents(tmp_path):
    # WHY: a file ends at a document boundary, so a reader that opens one
    #      file never starts mid-document: the next file starts before the
    #      document that would cross max_file_tokens, and a document longer
    #      than the limit fills a file alone (also when it comes first: no
    #      empty file before it).
    # KIND: boundary
    # CATCHES: s06, s08, s09, s10, s14, m08, m09
    # CHAPTER: data.07 section 5, Pitfalls, item 2
    texts = ["aaaa", "bbbbbb", "cc", "d" * 15, "ee"]
    m = corpus(tmp_path, texts, val_permille=0)
    tm = tokenize_shards(
        m, None, tmp_path / "t", tokenizer_id="bytes", max_file_tokens=10
    )
    got = [
        (f["name"], f["n_tokens"], f["n_docs"])
        for f in tm.files
        if f["split"] == "train"
    ]
    assert got == [
        ("train-00000.bin", 10, 2),
        ("train-00001.bin", 2, 1),
        ("train-00002.bin", 15, 1),
        ("train-00003.bin", 2, 1),
    ]
    assert bytes(ids_of(tmp_path / "t" / "train-00001.bin")) == b"cc"
    m2 = corpus(tmp_path, ["x" * 12, "y"], val_permille=0, name="c2")
    tm2 = tokenize_shards(
        m2, None, tmp_path / "t2", tokenizer_id="bytes", max_file_tokens=10
    )
    assert [(f["name"], f["n_tokens"]) for f in tm2.files if f["split"] == "train"] == [
        ("train-00000.bin", 12),
        ("train-00001.bin", 1),
    ]
    with pytest.raises(ValueError):
        tokenize_shards(
            m, None, tmp_path / "x", tokenizer_id="bytes", max_file_tokens=0
        )


def test_manifest_and_empty_split(tmp_path):
    # WHY: the manifest validates against formats/tokens-manifest.schema.json,
    #      names the exact inputs (the corpus manifest's sha256, the
    #      tokenizer's; sha256 of nothing for bytes), lists files sorted by
    #      name with their real hashes, and an empty split still has its
    #      -00000 file with no tokens so every consumer can open it.
    # KIND: conformance
    # CATCHES: s01, s03, s11, s12, m10, m11
    # CHAPTER: data.07 section 4, What the tests check
    m = corpus(tmp_path, ["one", "two", "three"], val_permille=0)
    tm = tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes")
    j = json.loads((tmp_path / "t" / "_MANIFEST.json").read_text())
    assert (
        errors(j, json.loads((CONTRACTS / "tokens-manifest.schema.json").read_text()))
        == []
    )
    assert j == tm.to_json()
    assert list(j) == [
        "tokenizer_id",
        "tokenizer_sha256",
        "vocab_size",
        "doc_sep_id",
        "dataset",
        "version",
        "corpus_manifest_sha256",
        "files",
    ]
    assert j["corpus_manifest_sha256"] == hashlib.sha256(m.read_bytes()).hexdigest()
    assert j["tokenizer_sha256"] == hashlib.sha256(b"").hexdigest()
    assert [f["name"] for f in j["files"]] == ["train-00000.bin", "val-00000.bin"]
    for f in j["files"]:
        assert (
            hashlib.sha256((tmp_path / "t" / f["name"]).read_bytes()).hexdigest()
            == f["sha256"]
        )
    assert (header(tmp_path / "t" / "val-00000.bin"), tm.val_bytes_per_token) == (
        (MAGIC, 1, 0, 256),
        None,
    )
    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == [
        "_MANIFEST.json",
        "train-00000.bin",
        "val-00000.bin",
    ]


def test_val_bytes_per_token(tmp_path):
    # WHY: C1 reports validation loss in bits per byte; the factor is the
    #      val split's bytes per token as L1.6 defines it (total bytes over
    #      total tokens, separators not counted).
    # KIND: unit
    # CATCHES: s03, s06, s08, s13, m12
    # CHAPTER: data.07 section 6, Where it's used next
    cases = gpt2_cases()
    m = corpus(tmp_path, [c["text"] for c in cases], val_permille=400)
    tm = tokenize_shards(m, GPT2, tmp_path / "t", tokenizer_id="gpt2", doc_sep_id=50256)

    class Oracle:  # the Hugging Face ids, as a Tokenizer for L1.6's metric
        ids = {c["text"]: c["ids"] for c in cases}

        def encode(self, text, add_special=False):
            return self.ids[text]

    val = [d.text for d in read_shards(m.parent, split="val")]
    assert tm.val_bytes_per_token == bytes_per_token(Oracle(), val)
    assert tm.val_bytes_per_token == sum(len(t.encode()) for t in val) / sum(
        len(Oracle.ids[t]) for t in val
    )


def test_output_is_deterministic(tmp_path):
    # WHY: repeated runs give byte-identical files and manifest. A smoke
    #      test: dependents rerun it, because a
    #      nondeterministic stage breaks MS-corpus's output hash.
    # KIND: property
    # CHAPTER: data.07 section 2, Principles
    cases = gpt2_cases()
    m = corpus(tmp_path, [c["text"] for c in cases] * 4, val_permille=200)
    tokenize_shards(
        m, GPT2, tmp_path / "a", tokenizer_id="gpt2", doc_sep_id=50256
    )
    tokenize_shards(
        m, GPT2, tmp_path / "b", tokenizer_id="gpt2", doc_sep_id=50256
    )
    for p in sorted((tmp_path / "a").iterdir()):
        assert p.read_bytes() == (tmp_path / "b" / p.name).read_bytes(), p.name


def test_atomic_replace(tmp_path):
    # WHY: like data.06, the output appears whole or not at all: a stale
    #      .tmp from a crash is cleared, a failing run (a separator outside
    #      the vocabulary) leaves the previous output as it was.
    # KIND: fault
    # CATCHES: s15
    # CHAPTER: data.07 section 5, Pitfalls, item 3
    m = corpus(tmp_path, ["one", "two"], val_permille=0)
    (tmp_path / "t.tmp").mkdir()
    (tmp_path / "t.tmp" / "junk.bin").write_bytes(b"x")
    tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes")
    assert not (tmp_path / "t.tmp").exists()
    before = {p.name: p.read_bytes() for p in (tmp_path / "t").iterdir()}
    with pytest.raises(ValueError):
        tokenize_shards(m, None, tmp_path / "t", tokenizer_id="bytes", doc_sep_id=300)
    assert {p.name: p.read_bytes() for p in (tmp_path / "t").iterdir()} == before
