"""Course tests for data.06: Parquet shards, the manifest, and the
document-hash train/val split (corpus/shard.py).

Rung R0 for these course tests (your own tests for this module are rung R3:
red then green, section 4 of the chapter). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/data.06), and the chapter section it comes from.

The files are fixed by course/contracts/formats/corpus-shard.md; the
manifest is validated against corpus-manifest.schema.json with the stdlib
subset validator next to these tests (schema_lite.py). The shards are read
back with pyarrow, which the corpus package requires (allowed-deps.toml).

The chapter's worked example (section 3), val_permille 300, shard_rows 2:

    id        text                    sha256[:8] as an integer, mod 1000   split
    books:0   "Mia naps."             ...940248  -> 248 < 300               val
    web:10    "Tom has a red ball!"   ...149826  -> 826                     train
    web:2     "Tom has a red ball."   ...549697  -> 697                     train
    rows sorted by id: books:0, web:10, web:2 ("1" sorts before "2")
    web:2 is a near duplicate of web:10 (data.04 drop=False): cluster 1
    shards: shard-00000-of-00002 (2 rows), shard-00001-of-00002 (1 row)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from _lib.pcg32 import PCG32
from corpus.minhash import near_dedup
from corpus.pii import scrub
from corpus.shard import (
    DEDUP_DEFAULTS,
    SCHEMA,
    read_shards,
    shard_name,
    split_of,
    write_shards,
)
from corpus.stage import Doc
from schema_lite import errors

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "formats"


def doc(i: str, text: str, **meta) -> Doc:
    base = {"url": f"https://example.org/{i}", "license_spdx": "CC0-1.0", "lang": "en"}
    base.update(meta)
    return Doc(id=i, source_id=i.split(":")[0], text=text, meta=base)


def hand_docs() -> list[Doc]:
    return [
        doc("web:2", "Tom has a red ball.", minhash_cluster="web:10"),
        doc("web:10", "Tom has a red ball!", minhash_cluster="web:10"),
        doc("books:0", "Mia naps."),
    ]


def table(out: Path) -> pa.Table:
    m = json.loads((out / "_MANIFEST.json").read_text())
    return pa.concat_tables([pq.read_table(out / s["name"]) for s in m["shards"]])


def many(n: int, seed: int = 0) -> list[Doc]:
    rng = PCG32(seed)
    out = []
    for i in range(n):
        words = ["abcdefgh"[rng.below(8)] for _ in range(3 + rng.below(28))]
        out.append(doc(f"src:{i}", f"story {i} " + " ".join(words)))
    return out


# --- the worked example ------------------------------------------------------------


def test_hand_example(tmp_path):
    # WHY: section 3 by hand: sort by id in byte order, cut 2 rows per
    #      shard, split by the content hash, and turn the cluster id into a
    #      row index. These are the bytes data.07 tokenizes next.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, s08, s09, m04, m05
    # CHAPTER: data.06 section 3, Worked example by hand
    m = write_shards(
        hand_docs(),
        tmp_path / "c",
        dataset="tiny",
        version="v1",
        shard_rows=2,
        val_permille=300,
    )
    assert [s["name"] for s in m["shards"]] == [
        "shard-00000-of-00002.parquet",
        "shard-00001-of-00002.parquet",
    ]
    assert [s["rows"] for s in m["shards"]] == [2, 1]
    assert (m["n_docs"], m["n_shards"]) == (3, 2)
    t = table(tmp_path / "c").to_pydict()
    assert t["id"] == ["books:0", "web:10", "web:2"]
    assert t["split"] == ["val", "train", "train"]
    assert t["minhash_cluster"] == [0, 1, 1]
    assert t["n_chars"] == [9, 19, 19]
    assert t["sha256"][0] == hashlib.sha256(b"Mia naps.").digest()
    assert split_of(hashlib.sha256(b"Mia naps.").digest(), 300) == "val"
    assert split_of(hashlib.sha256(b"Mia naps.").digest(), 248) == "train"


# --- the files ---------------------------------------------------------------------


def test_schema_compression_and_manifest(tmp_path):
    # WHY: data.07, data.08, and anyone's reader rely on the exact column
    #      names, order, and Arrow types (sha256 is 32 raw bytes, counts
    #      are int32), zstd compression, and a manifest that validates
    #      against formats/corpus-manifest.schema.json with the shard
    #      hashes of the files actually written.
    # KIND: conformance
    # CATCHES: s07, s11, m02, m05
    # CHAPTER: data.06 section 4, What the tests check
    out = tmp_path / "c"
    m = write_shards(many(25), out, dataset="d", version="v1", shard_rows=10)
    schema = json.loads((CONTRACTS / "corpus-manifest.schema.json").read_text())
    on_disk = json.loads((out / "_MANIFEST.json").read_text())
    assert on_disk == m
    assert errors(on_disk, schema) == []
    for s in m["shards"]:
        f = pq.ParquetFile(out / s["name"])
        assert f.schema_arrow.equals(SCHEMA)
        assert SCHEMA.names == [
            "id",
            "text",
            "source_id",
            "url",
            "license_spdx",
            "lang",
            "n_chars",
            "sha256",
            "minhash_cluster",
            "pii_redactions",
            "split",
        ]
        assert str(SCHEMA.field("sha256").type) == "fixed_size_binary[32]"
        assert f.metadata.row_group(0).column(1).compression == "ZSTD"
        assert hashlib.sha256((out / s["name"]).read_bytes()).hexdigest() == s["sha256"]
        assert f.metadata.num_rows == s["rows"]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [s["name"] for s in m["shards"]] + ["_MANIFEST.json"]
    )


def test_sorted_by_id_in_byte_order(tmp_path):
    # WHY: the whole dataset is sorted by id as strings, so "a:10" comes
    #      before "a:2" and a non-ASCII id after every ASCII one; sorting
    #      numerically, or per shard, changes which shard a row lands in.
    #      The defaults of the other columns show here too: n_chars counts
    #      code points, a missing language is "und", and every untagged
    #      row is its own cluster.
    # KIND: unit
    # CATCHES: s01, s04, s08, m01, m03, m08
    # CHAPTER: data.06 section 5, Pitfalls, item 1
    docs = [doc(i, f"text {i}") for i in ["a:2", "é:0", "a:10", "b:1"]]
    docs.append(
        Doc(
            id="a:1",
            source_id="a",
            text="text a:1",
            meta={"url": "u", "license_spdx": "MIT"},
        )
    )
    write_shards(docs, tmp_path / "c", dataset="d", version="v1", shard_rows=2)
    t = table(tmp_path / "c").to_pydict()
    assert t["id"] == ["a:1", "a:10", "a:2", "b:1", "é:0"]
    assert t["n_chars"] == [8, 9, 8, 8, 8]  # code points: "text é:0" is 9 bytes
    assert t["lang"] == ["und", "en", "en", "en", "en"]  # no language tag: undetermined
    assert t["minhash_cluster"] == [
        0,
        1,
        2,
        3,
        4,
    ]  # untagged rows are their own clusters


def test_split_follows_the_hash(tmp_path):
    # WHY: a document's split is a function of its text alone: exact
    #      copies (different ids, same text) always land on the same side,
    #      so no evaluation text leaks into training through a duplicate,
    #      and the split does not move when documents are added or
    #      reordered. The big-endian first 8 bytes are the rule; reading
    #      them little-endian, or hashing the id, picks other documents.
    # KIND: property
    # CATCHES: s02, s03, m04
    # CHAPTER: data.06 section 5, Pitfalls, item 2
    docs = many(300, seed=1)
    docs += [doc(f"copy:{i}", d.text) for i, d in enumerate(docs[:60])]
    write_shards(
        docs, tmp_path / "a", dataset="d", version="v1", shard_rows=50, val_permille=200
    )
    t = table(tmp_path / "a").to_pydict()
    side = {}
    for text, sha, split in zip(t["text"], t["sha256"], t["split"]):
        want = "val" if int.from_bytes(sha[:8], "big") % 1000 < 200 else "train"
        assert split == want
        assert side.setdefault(text, split) == split
    write_shards(
        list(reversed(docs[:200])),
        tmp_path / "b",
        dataset="d",
        version="v1",
        shard_rows=7,
        val_permille=200,
    )
    t2 = table(tmp_path / "b").to_pydict()
    again = dict(zip(t2["id"], t2["split"]))
    assert all(again[i] == s for i, s in zip(t["id"], t["split"]) if i in again)


def test_val_share_matches_permille(tmp_path):
    # WHY: the hash spreads documents uniformly: with 4000 documents and
    #      val_permille 100 the val count is Binomial(4000, 0.1); it must
    #      be within 4 standard deviations (19) of 400.
    # KIND: statistical
    # CATCHES: m04
    # CHAPTER: data.06 section 2, Principles
    docs = [doc(f"s:{i}", f"document number {i}") for i in range(4000)]
    write_shards(
        docs,
        tmp_path / "c",
        dataset="d",
        version="v1",
        shard_rows=4000,
        val_permille=100,
    )
    n_val = table(tmp_path / "c")["split"].to_pylist().count("val")
    assert abs(n_val - 400) <= 4 * (4000 * 0.1 * 0.9) ** 0.5
    with pytest.raises(ValueError):
        split_of(b"\0" * 32, 1001)
    with pytest.raises(ValueError):
        split_of(b"\0" * 31, 5)


def test_shards_and_names(tmp_path):
    # WHY: 23 documents at 10 rows per shard are 3 shards of 10, 10, and 3,
    #      named i-of-n with five digits; an empty corpus still has one
    #      (empty) shard, so a reader never finds a manifest without files.
    # KIND: boundary
    # CATCHES: s04, m05, m06
    # CHAPTER: data.06 section 4, What the tests check
    m = write_shards(many(23), tmp_path / "c", dataset="d", version="v1", shard_rows=10)
    assert [(s["name"], s["rows"]) for s in m["shards"]] == [
        ("shard-00000-of-00003.parquet", 10),
        ("shard-00001-of-00003.parquet", 10),
        ("shard-00002-of-00003.parquet", 3),
    ]
    e = write_shards([], tmp_path / "e", dataset="d", version="v1", shard_rows=10)
    assert (e["n_docs"], e["n_shards"], e["shards"][0]["rows"]) == (0, 1, 0)
    assert (
        pq.ParquetFile(
            tmp_path / "e" / "shard-00000-of-00001.parquet"
        ).metadata.num_rows
        == 0
    )
    assert shard_name(4, 64) == "shard-00004-of-00064.parquet"
    with pytest.raises(ValueError):
        shard_name(3, 3)


def test_row_groups_are_cut_by_text_bytes(tmp_path):
    # WHY: a row group is the unit a reader loads at once; 64 MiB bounds
    #      the memory of every consumer. Groups hold consecutive rows whose
    #      text bytes fit the budget, and one longer document gets a group
    #      to itself instead of being split or dropped.
    # KIND: boundary
    # CATCHES: s10, m05, m07
    # CHAPTER: data.06 section 5, Pitfalls, item 3
    texts = ["x" * 10, "x" * 10, "x" * 10, "y" * 40, "x" * 10, "é" * 5]
    docs = [doc(f"s:{i}", t) for i, t in enumerate(texts)]
    write_shards(
        docs,
        tmp_path / "c",
        dataset="d",
        version="v1",
        shard_rows=100,
        row_group_bytes=25,
    )
    f = pq.ParquetFile(tmp_path / "c" / "shard-00000-of-00001.parquet")
    sizes = [f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)]
    assert sizes == [2, 1, 1, 2]  # 20 | 10 | 40 alone | 10 + 10 (5 two-byte chars)
    write_shards(
        [doc("t:0", "z" * 30), doc("t:1", "x" * 10)],
        tmp_path / "d",
        dataset="d",
        version="v1",
        shard_rows=100,
        row_group_bytes=25,
    )
    f = pq.ParquetFile(tmp_path / "d" / "shard-00000-of-00001.parquet")
    assert [
        f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)
    ] == [1, 1]


def test_output_is_deterministic(tmp_path):
    # WHY: MS-corpus and the durable CorpusBuild (data.09) compare output
    #      hashes: the same documents in any order give byte-identical
    #      shards and manifest. Nothing may depend on arrival order, time,
    #      or the machine.
    # KIND: property
    # CATCHES: s01, s11
    # CHAPTER: data.06 section 2, Principles
    docs = many(120, seed=4)
    shuffled = docs[:]
    rng = PCG32(9)
    for i in range(len(shuffled) - 1, 0, -1):  # Fisher-Yates
        j = rng.below(i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    write_shards(docs, tmp_path / "a", dataset="d", version="v1", shard_rows=25)
    write_shards(
        iter(shuffled), tmp_path / "b", dataset="d", version="v1", shard_rows=25
    )
    for p in sorted((tmp_path / "a").iterdir()):
        assert p.read_bytes() == (tmp_path / "b" / p.name).read_bytes(), p.name


def test_cluster_column_from_near_dedup(tmp_path):
    # WHY: minhash_cluster is the smallest row index of a document's
    #      cluster, so analysts can group near duplicates without the ids.
    #      data.04 with drop=False tags every member with the root id; a
    #      document without the tag is its own cluster.
    # KIND: unit
    # CATCHES: s01, s05
    # CHAPTER: data.06 section 6, Where it's used next
    base = " ".join(f"w{i}" for i in range(60))
    docs = [
        doc("s:5", base),
        doc("s:1", "something else entirely, alone"),
        doc("s:3", base + " tail"),
    ]
    tagged = list(near_dedup(docs, drop=False))
    tagged.append(doc("s:9", "untagged"))
    write_shards(tagged, tmp_path / "c", dataset="d", version="v1", shard_rows=10)
    t = table(tmp_path / "c").to_pydict()
    assert t["id"] == ["s:1", "s:3", "s:5", "s:9"]
    assert t["minhash_cluster"] == [0, 1, 1, 3]


def test_pii_totals(tmp_path):
    # WHY: the manifest's pii counts are the sums of data.05's per-document
    #      counts under the manifest's key names, and each row keeps its
    #      own pii_redactions; the datasheet (data.08) reports both.
    # KIND: unit
    # CATCHES: s06, m09
    # CHAPTER: data.06 section 4, The interface
    raw = [
        doc("s:0", "mail a@example.com or b@example.com"),
        doc("s:1", "server 203.0.113.5"),
        doc("s:2", "clean"),
    ]
    docs = [scrub(d)[0] for d in raw]
    m = write_shards(docs, tmp_path / "c", dataset="d", version="v1", shard_rows=10)
    assert m["pii"] == {"emails": 2, "phones": 0, "cards": 0, "ips": 1, "keys": 0}
    assert table(tmp_path / "c")["pii_redactions"].to_pylist() == [2, 1, 0]


def test_counts_and_key_order(tmp_path):
    # WHY: the manifest records what every stage dropped: the filter
    #      counts as given, the dedup block as the defaults overridden by
    #      the caller, and the keys in the format's order so two writers
    #      produce the same bytes.
    # KIND: unit
    # CATCHES: s07, s11, m10
    # CHAPTER: data.06 section 4, The interface
    m = write_shards(
        many(3),
        tmp_path / "c",
        dataset="d",
        version="v2",
        shard_rows=5,
        filters={"lang": 4, "gopher": 1},
        dedup={"near_dropped": 7, "decontaminated": 2},
        ledger_ref="corpus/LEDGER.jsonl",
        config_sha256="ab" * 32,
    )
    assert m["filters"] == {"lang": 4, "gopher": 1}
    assert m["dedup"] == {**DEDUP_DEFAULTS, "near_dropped": 7, "decontaminated": 2}
    assert list(m) == [
        "dataset",
        "version",
        "n_docs",
        "n_shards",
        "shards",
        "filters",
        "dedup",
        "pii",
        "ledger_ref",
        "config_sha256",
    ]
    text = (tmp_path / "c" / "_MANIFEST.json").read_text()
    assert text == json.dumps(m, indent=2, ensure_ascii=False) + "\n"


def test_rejects_bad_input(tmp_path):
    # WHY: a row without a url or license cannot be traced by the ledger
    #      (data.08), and a repeated id would make the sort ambiguous: both
    #      are errors before anything is written.
    # KIND: boundary
    # CATCHES: s12, m11
    # CHAPTER: data.06 section 5, Pitfalls, item 4
    with pytest.raises(ValueError):
        write_shards(
            [Doc(id="a:0", source_id="a", text="t", meta={"url": "u"})],
            tmp_path / "x",
            dataset="d",
            version="v1",
            shard_rows=1,
        )
    with pytest.raises(ValueError):
        write_shards(
            [doc("a:0", "t"), doc("a:0", "u")],
            tmp_path / "x",
            dataset="d",
            version="v1",
            shard_rows=1,
        )
    with pytest.raises(ValueError):
        write_shards(
            [doc("a:0", "t")], tmp_path / "x", dataset="d", version="v1", shard_rows=0
        )
    assert not (tmp_path / "x").exists()


def test_atomic_replace(tmp_path):
    # WHY: an activity can die mid-write and be retried (data.09). The
    #      writer builds everything in out.tmp and renames it at the end: a
    #      failed run leaves the previous output untouched, a stale .tmp
    #      from a crash is ignored, and a successful run replaces out whole.
    # KIND: fault
    # CATCHES: s13, m05
    # CHAPTER: data.06 section 5, Pitfalls, item 5
    out = tmp_path / "c"
    write_shards(many(10), out, dataset="d", version="v1", shard_rows=4)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def dying():
        yield from many(5)
        raise OSError("disk went away")

    with pytest.raises(OSError):
        write_shards(dying(), out, dataset="d", version="v1", shard_rows=4)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    (tmp_path / "c.tmp").mkdir(exist_ok=True)
    (tmp_path / "c.tmp" / "junk").write_text("left by a crash")
    write_shards(many(3), out, dataset="d", version="v1", shard_rows=4)
    assert sorted(p.name for p in out.iterdir()) == [
        "_MANIFEST.json",
        "shard-00000-of-00001.parquet",
    ]
    assert not (tmp_path / "c.tmp").exists()


def test_read_shards(tmp_path):
    # WHY: data.07 and data.08 read the corpus back through read_shards:
    #      manifest order, only one split when asked, the row's columns in
    #      meta, and a shard whose bytes no longer match the manifest (here
    #      a valid Parquet file with one text changed) is an error, not
    #      silently different data.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s08, s14, m04, m05, m12
    # CHAPTER: data.06 section 6, Where it's used next
    out = tmp_path / "c"
    write_shards(
        hand_docs(), out, dataset="tiny", version="v1", shard_rows=2, val_permille=300
    )
    got = list(read_shards(out))
    assert [d.id for d in got] == ["books:0", "web:10", "web:2"]
    assert [d.meta["split"] for d in got] == ["val", "train", "train"]
    assert (
        got[2].meta["shard"] == "shard-00001-of-00002.parquet"
        and got[2].meta["row"] == 0
    )
    assert got[0].meta["sha256"] == hashlib.sha256(b"Mia naps.").hexdigest()
    assert got[1].source_id == "web" and got[1].meta["license_spdx"] == "CC0-1.0"
    assert [d.id for d in read_shards(out, split="train")] == ["web:10", "web:2"]
    p = out / "shard-00001-of-00002.parquet"
    t = pq.read_table(p).to_pydict()
    t["text"] = ["Tom has a blue ball."]
    pq.write_table(
        pa.table(t, schema=SCHEMA), p, compression="zstd"
    )  # valid Parquet, other bytes
    with pytest.raises(ValueError):
        list(read_shards(out))
