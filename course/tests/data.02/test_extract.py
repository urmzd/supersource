"""data.02 course tests: corpus.stage.extract (raw parts to Docs).

Raw parts are written here with zstandard directly, in the
formats/corpus-shard.md layout, so these tests do not depend on your fetch.
"""

from __future__ import annotations

import json
from pathlib import Path

import zstandard

from corpus.fetch import Fetched, Manifest, Source
from corpus.stage import Doc, extract


def _part(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows
    )
    path.write_bytes(zstandard.ZstdCompressor(level=3).compress(body.encode("utf-8")))


def _raw(sid: str, k: int) -> dict:
    return {
        "url": f"http://h/{sid}.jsonl#{k + 1}",
        "fetched_at": "2026-01-01T00:00:00Z",
        "license_spdx": "CC0-1.0",
        "text": f"{sid} story {k}",
    }


def _entry(sid: str, raw_dir: Path, status: str = "fetched") -> Fetched:
    s = Source(sid, f"http://h/{sid}.jsonl", "0" * 64, "CC0-1.0")
    return Fetched(s, status, str(raw_dir), "0" * 64, 0, 0, "2026-01-01T00:00:00Z", 0)


def test_extract_numbers_documents_per_source_across_parts(tmp_path):
    # WHY: formats/corpus-shard.md: a document's id is <source_id>:<k> with
    #      k its 0-based position in the source's raw files read in name
    #      order, so ids are stable across reruns and cross part boundaries;
    #      the next source starts again at 0. meta carries url, fetched_at,
    #      and license_spdx from the raw line, which the shard and the
    #      datasheet need later.
    # KIND: unit
    # CATCHES: s29, s30
    a = tmp_path / "raw" / "a" / "20260101"
    _part(a / "part-00001.jsonl.zst", [_raw("a", 2)])
    _part(a / "part-00000.jsonl.zst", [_raw("a", 0), _raw("a", 1)])
    b = tmp_path / "raw" / "b" / "20260101"
    _part(b / "part-00000.jsonl.zst", [_raw("b", 0)])
    m = Manifest((_entry("a", a), _entry("b", b, "cached")))
    out = list(extract(m))
    assert [d.id for d in out] == ["a:0", "a:1", "a:2", "b:0"]
    assert [d.text for d in out] == ["a story 0", "a story 1", "a story 2", "b story 0"]
    assert out[0] == Doc(
        "a:0",
        "a",
        "a story 0",
        {
            "url": "http://h/a.jsonl#1",
            "fetched_at": "2026-01-01T00:00:00Z",
            "license_spdx": "CC0-1.0",
        },
    )


def test_extract_skips_quarantined_sources(tmp_path):
    # WHY: a quarantined source's bytes failed their checksum; none of its
    #      documents may enter the corpus, even when a directory of parts
    #      from somewhere is named in the entry, and its absence must not
    #      shift the ids of the sources after it. Decide by status.
    # KIND: boundary
    # CATCHES: s31
    a = tmp_path / "raw" / "a" / "20260101"
    _part(a / "part-00000.jsonl.zst", [_raw("a", 0)])
    b = tmp_path / "raw" / "b" / "20260101"
    _part(b / "part-00000.jsonl.zst", [_raw("b", 0)])
    m = Manifest((_entry("a", a, "quarantined"), _entry("b", b)))
    assert [d.id for d in extract(m)] == ["b:0"]
