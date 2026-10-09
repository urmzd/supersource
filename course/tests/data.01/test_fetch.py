"""data.01 course tests: corpus.fetch (async fetch with resume, checksums, license capture).

Annotated exemplars (DESIGN 5.12). No test touches the network: sources are
served by the course's `flakyhttp` fixture server (course/testkit/python),
which cuts bodies, fails with 5xx, ignores Range, and lies about checksums
on request, and by two tiny asyncio servers below (one counts requests in
flight, one redirects). Backoff waits go through a fake `sleep` that only
records what it was asked; time comes from the testkit FakeClock, which
starts at 2026-01-01T00:00:00Z.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from pathlib import Path

import pytest
import zstandard
from sstestkit.clock import FakeClock
from sstestkit.flakyhttp import FlakyHTTP as _FlakyHTTP

from corpus.fetch import (
    LEDGER_NAME,
    Fetched,
    FetchError,
    Manifest,
    Source,
    fetch,
    ledger_row,
    read_raw,
    sources_from_config,
    split_documents,
)

# The section 3 worked example: two JSON Lines documents, 54 bytes.
TINY = b'{"text": "Tom has a red ball."}\n{"text": "Mia naps."}\n'
TINY_SHA = hashlib.sha256(TINY).hexdigest()
SCHEMA = (
    Path(__file__).resolve().parents[2] / "contracts" / "formats" / "ledger.schema.json"
)


class FlakyHTTP(_FlakyHTTP):
    """The testkit server, polled every 10 ms so closing it is quick."""

    def __enter__(self) -> "FlakyHTTP":
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()
        return self


class FakeSleep:
    """Records each requested delay and returns at once."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def src(srv: FlakyHTTP, path: str, data: bytes, **kw) -> Source:
    return Source(
        id=kw.pop("id", path.strip("/").split(".")[0]),
        url=srv.url(path),
        sha256=kw.pop("sha256", sha(data)),
        license_spdx=kw.pop("license_spdx", "CC0-1.0"),
        **kw,
    )


def run(srcs, dest, **kw) -> Manifest:
    kw.setdefault("clock", FakeClock())
    kw.setdefault("sleep", FakeSleep())
    kw.setdefault("timeout", 2.0)
    return asyncio.run(fetch(srcs, dest, **kw))


def ledger(dest: Path) -> list[dict]:
    p = dest / LEDGER_NAME
    return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []


def stories(n: int, seed: str = "s") -> bytes:
    """n JSON Lines documents, each a little different."""
    return b"".join(
        json.dumps({"text": f"Story {seed}{i}: the cat sat on mat {i}."}).encode()
        + b"\n"
        for i in range(n)
    )


# --- the worked example ------------------------------------------------------


def test_hand_example_resume_after_a_cut(tmp_path):
    # WHY: the section 3 timeline. The first GET is cut after 20 bytes; the
    #      second asks for `Range: bytes=20-` and gets the other 34, so the
    #      file is whole after 2 requests and 54 bytes, the pinned sha256
    #      matches, and the two documents land in raw/tiny/20260101/ with
    #      the license and the fetch time on every line.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: data.01 section 3, Worked example by hand
    with FlakyHTTP({"/tiny.jsonl": TINY}) as srv:
        srv.truncate("/tiny.jsonl", after=20, times=1)
        s = src(srv, "/tiny.jsonl", TINY, license_spdx="CC-BY-4.0")
        m = run([s], tmp_path)
        sent = srv.headers("/tiny.jsonl")
    assert "Range" not in sent[0]
    assert sent[1].get("Range") == "bytes=20-"
    (f,) = m.entries
    assert (f.status, f.requests, f.bytes, f.sha256, f.n_docs) == (
        "fetched",
        2,
        54,
        TINY_SHA,
        2,
    )
    assert f.retrieved_at == "2026-01-01T00:00:00Z"
    assert Path(f.raw_dir) == tmp_path / "raw" / "tiny" / "20260101"
    docs = list(read_raw(f.raw_dir))
    assert [d["text"] for d in docs] == ["Tom has a red ball.", "Mia naps."]
    assert [d["url"] for d in docs] == [s.url + "#1", s.url + "#2"]
    assert {d["license_spdx"] for d in docs} == {"CC-BY-4.0"}
    assert {d["fetched_at"] for d in docs} == {"2026-01-01T00:00:00Z"}
    assert ledger(tmp_path) == [
        {
            "source_id": "tiny",
            "url": s.url,
            "license_spdx": "CC-BY-4.0",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "sha256": TINY_SHA,
            "n_docs": 2,
            "allowed_uses": ["train", "eval"],
            "pii_policy": "scrub",
            "filters_applied": [],
            "kept": 2,
            "dropped": 0,
            "notes": "",
        }
    ]
    assert m.ok


# --- formats: raw parts and ledger rows ---------------------------------------------


def test_raw_parts_are_zstd_json_lines(tmp_path):
    # WHY: formats/corpus-shard.md fixes the raw document bytes: zstd frames
    #      of compact JSON Lines with the keys url, fetched_at, license_spdx,
    #      text in that order and non-ASCII kept as UTF-8. data.02 reads
    #      them, and so does anyone auditing the corpus with stock tools, so
    #      they are decoded here with zstandard, not with your read_raw.
    # KIND: conformance
    # CATCHES: s13
    data = '{"text": "Zoë met a café owner."}\n'.encode()
    with FlakyHTTP({"/u.jsonl": data}) as srv:
        s = src(srv, "/u.jsonl", data)
        (f,) = run([s], tmp_path).entries
    parts = sorted(Path(f.raw_dir).iterdir())
    assert [p.name for p in parts] == ["part-00000.jsonl.zst"]
    blob = parts[0].read_bytes()
    assert blob[:4] == b"\x28\xb5\x2f\xfd", "not a zstd frame"
    text = zstandard.ZstdDecompressor().decompressobj().decompress(blob).decode("utf-8")
    expected = {
        "url": s.url + "#1",
        "fetched_at": "2026-01-01T00:00:00Z",
        "license_spdx": "CC0-1.0",
        "text": "Zoë met a café owner.",
    }
    assert (
        text == json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    )


def _check_schema(row: dict, schema: dict) -> list[str]:
    """The subset of JSON Schema that ledger.schema.json uses."""
    errs = []
    props = schema["properties"]
    for k in schema["required"]:
        if k not in row:
            errs.append(f"missing {k}")
    for k, v in row.items():
        p = props.get(k)
        if p is None:
            errs.append(f"unexpected key {k}")
            continue
        t = p.get("type")
        if t == "string" and not isinstance(v, str):
            errs.append(f"{k} is not a string")
        if t == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
            errs.append(f"{k} is not an integer")
        if t == "array" and not isinstance(v, list):
            errs.append(f"{k} is not an array")
        if "pattern" in p and isinstance(v, str) and not re.search(p["pattern"], v):
            errs.append(f"{k}={v!r} does not match {p['pattern']}")
        if "enum" in p and v not in p["enum"]:
            errs.append(f"{k}={v!r} not in {p['enum']}")
        if t == "array" and "enum" in p.get("items", {}):
            bad = [x for x in v if x not in p["items"]["enum"]]
            if bad:
                errs.append(f"{k} has {bad}")
        if "minimum" in p and isinstance(v, int) and v < p["minimum"]:
            errs.append(f"{k} below {p['minimum']}")
    return errs


def test_ledger_rows_validate_against_the_schema(tmp_path):
    # WHY: the ledger is a contract read by data.08 (verification), dur.12
    #      (the release gate), and the datasheet; a missing field or a local
    #      timestamp without the Z fails all three later. Two sources with
    #      different licenses and allowed uses, checked against
    #      formats/ledger.schema.json itself.
    # KIND: conformance
    # CATCHES: s18, s21
    schema = json.loads(SCHEMA.read_text())
    a, b = stories(3, "a"), stories(2, "b")
    with FlakyHTTP({"/a.jsonl": a, "/b.jsonl": b}) as srv:
        s1 = src(srv, "/a.jsonl", a, license_spdx="CDLA-Sharing-1.0")
        s2 = src(srv, "/b.jsonl", b, allowed_uses=("eval",), pii_policy="none")
        run([s1, s2], tmp_path, clock=FakeClock(1_791_547_200.5))
    rows = ledger(tmp_path)
    assert len(rows) == 2
    for r in rows:
        assert _check_schema(r, schema) == [], r
    assert rows[0]["retrieved_at"] == "2026-10-09T12:00:00Z"
    assert rows[1]["allowed_uses"] == ["eval"] and rows[1]["pii_policy"] == "none"
    assert rows[0]["license_spdx"] == "CDLA-Sharing-1.0"


def test_ledger_row_is_a_pure_function_of_fetched():
    # WHY: ledger_row is the one place the row's shape lives; data.08 and
    #      the datasheet regenerate rows from Fetched values with it.
    # KIND: unit
    # CATCHES: s21
    s = Source("x", "http://h/x.jsonl", "a" * 64, "MIT", ("train",), "jsonl", "drop")
    f = Fetched(s, "fetched", "/r", "b" * 64, 10, 3, "2026-01-02T03:04:05Z", 1)
    assert ledger_row(f) == {
        "source_id": "x",
        "url": "http://h/x.jsonl",
        "license_spdx": "MIT",
        "retrieved_at": "2026-01-02T03:04:05Z",
        "sha256": "b" * 64,
        "n_docs": 3,
        "allowed_uses": ["train"],
        "pii_policy": "drop",
        "filters_applied": [],
        "kept": 3,
        "dropped": 0,
        "notes": "",
    }


# --- resume, ranges, checksums ---------------------------------------------------------


def test_a_server_that_ignores_range_restarts_the_file(tmp_path):
    # WHY: a server may answer a Range request with 200 and the whole file.
    #      Appending that to the partial bytes gives prefix + whole file, a
    #      checksum mismatch, and a good source quarantined for nothing.
    #      200 means "start over".
    # KIND: fault
    # CATCHES: s01
    # CHAPTER: data.01 section 5, Pitfalls, item 2
    data = stories(20)
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        srv.no_ranges("/s.jsonl")
        srv.truncate("/s.jsonl", after=100, times=1)
        (f,) = run([src(srv, "/s.jsonl", data)], tmp_path).entries
        hits = srv.hits("/s.jsonl")
    assert (f.status, f.n_docs, f.bytes) == ("fetched", 20, len(data))
    assert hits == 2


def test_cut_bodies_resume_until_whole(tmp_path):
    # WHY: a long download can be cut several times; each attempt must
    #      continue from the bytes already on disk, so a 4-attempt budget
    #      survives 3 cuts and the server sends every byte exactly once.
    # KIND: fault
    # CATCHES: s08, m01
    data = stories(50)
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        srv.truncate("/s.jsonl", after=500, times=3)
        sleep = FakeSleep()
        (f,) = run(
            [src(srv, "/s.jsonl", data)], tmp_path, attempts=4, sleep=sleep
        ).entries
        ranges = [h.get("Range") for h in srv.headers("/s.jsonl")]
    assert f.status == "fetched" and f.requests == 4
    assert ranges == [None, "bytes=500-", "bytes=1000-", "bytes=1500-"]
    assert len(sleep.calls) == 3


def test_a_checksum_mismatch_is_quarantined_not_retried(tmp_path):
    # WHY: bytes that do not hash to the pinned sha256 will not hash to it
    #      next time either; retrying wastes the mirror's bandwidth and the
    #      run's time. The file moves to quarantine/ for a human, nothing
    #      reaches raw/ or the ledger, and the manifest says not ok.
    # KIND: fault
    # CATCHES: s02, s03, m02
    # CHAPTER: data.01 section 5, Pitfalls, item 3
    data = stories(5)
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        s = src(srv, "/s.jsonl", data, sha256=sha(b"something else"))
        m = run([s], tmp_path)
        hits = srv.hits("/s.jsonl")
    (f,) = m.entries
    assert (f.status, f.raw_dir, f.n_docs) == ("quarantined", "", 0)
    assert f.sha256 == sha(data)
    assert hits == 1
    assert (tmp_path / "quarantine" / s.id / "s.jsonl").read_bytes() == data
    assert not (tmp_path / "raw" / s.id).exists()
    assert ledger(tmp_path) == []
    assert not m.ok


def test_the_servers_checksum_header_is_not_trusted(tmp_path):
    # WHY: the pinned sha256 in your config is the only checksum you
    #      reviewed. A mirror's X-Content-Sha256 header (here a lie) proves
    #      nothing about the bytes; verifying against it would quarantine a
    #      good file, or pass a swapped one.
    # KIND: fault
    # CATCHES: s02
    # CHAPTER: data.01 section 5, Pitfalls, item 4
    data = stories(5)
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        srv.lie_checksum("/s.jsonl")
        (f,) = run([src(srv, "/s.jsonl", data)], tmp_path).entries
    assert f.status == "fetched"


def test_a_rerun_downloads_nothing(tmp_path):
    # WHY: CorpusBuild (data.09) reruns fetch after every crash and retry.
    #      A source whose pinned sha256 the ledger already records is
    #      cached: no request, the same raw directory, and the ledger is
    #      byte for byte unchanged (no duplicate rows).
    # KIND: fault
    # CATCHES: s05, s17
    # CHAPTER: data.01 section 5, Pitfalls, item 6
    a, b = stories(4, "a"), stories(6, "b")
    with FlakyHTTP({"/a.jsonl": a, "/b.jsonl": b}) as srv:
        srcs = [src(srv, "/a.jsonl", a), src(srv, "/b.jsonl", b)]
        first = run(srcs, tmp_path)
        before = (tmp_path / LEDGER_NAME).read_bytes()
        hits = srv.hits("/a.jsonl") + srv.hits("/b.jsonl")
        second = run(srcs, tmp_path, clock=FakeClock(1_767_225_600 + 86_400 * 3))
        hits2 = srv.hits("/a.jsonl") + srv.hits("/b.jsonl")
    assert hits == hits2 == 2
    assert [e.status for e in second.entries] == ["cached", "cached"]
    assert [e.requests for e in second.entries] == [0, 0]
    assert [e.raw_dir for e in second.entries] == [e.raw_dir for e in first.entries]
    assert [e.n_docs for e in second.entries] == [4, 6]
    assert (tmp_path / LEDGER_NAME).read_bytes() == before


def test_a_new_pinned_checksum_fetches_again(tmp_path):
    # WHY: "cached" means the ledger records THIS sha256. When the config
    #      pins a new version of a source, the old row must not satisfy it:
    #      the new bytes are fetched into a new dated directory and get their
    #      own ledger row.
    # KIND: boundary
    # CATCHES: s24
    old, new = stories(2, "old"), stories(3, "new")
    with FlakyHTTP({"/s.jsonl": old}) as srv:
        s = src(srv, "/s.jsonl", old)
        run([s], tmp_path)
        srv.files["/s.jsonl"] = new
        s2 = src(srv, "/s.jsonl", new)
        (f,) = run([s2], tmp_path, clock=FakeClock(1_767_225_600 + 86_400)).entries
    assert f.status == "fetched" and f.n_docs == 3
    assert Path(f.raw_dir).name == "20260102"
    assert [r["sha256"] for r in ledger(tmp_path)] == [sha(old), sha(new)]


def test_a_whole_partial_file_gets_416_and_is_kept(tmp_path):
    # WHY: a crash after the last byte but before the rename leaves a .part
    #      that is already whole. Asking for the bytes after its end gets
    #      416 Range Not Satisfiable; that means "done", not "failed".
    # KIND: fault
    # CATCHES: s14
    data = stories(3)
    part = tmp_path / "downloads" / "s" / "s.jsonl.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(data)
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        (f,) = run([src(srv, "/s.jsonl", data)], tmp_path).entries
        hits = srv.hits("/s.jsonl")
    assert f.status == "fetched" and f.n_docs == 3
    assert hits == 1


def test_a_verified_download_left_by_a_crash_is_not_downloaded_again(tmp_path):
    # WHY: a crash after verification but before the raw parts were written
    #      leaves downloads/<id>/<name> complete. Its sha256 still matches,
    #      so the rerun converts it without a request.
    # KIND: fault
    # CATCHES: s23
    data = stories(3)
    done = tmp_path / "downloads" / "s" / "s.jsonl"
    done.parent.mkdir(parents=True)
    done.write_bytes(data)
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        (f,) = run([src(srv, "/s.jsonl", data)], tmp_path).entries
        hits = srv.hits("/s.jsonl")
    assert (f.status, f.requests, f.n_docs) == ("fetched", 0, 3)
    assert hits == 0


# --- retries and timeouts -------------------------------------------------------------


def test_5xx_is_retried_with_exponential_backoff(tmp_path):
    # WHY: two 503s, then success: waits of base_delay * 2**(n-1) after
    #      attempt n (0.5 s, then 1 s), so a struggling mirror sees fewer
    #      and fewer requests.
    # KIND: unit
    # CATCHES: s09
    data = stories(2)
    sleep = FakeSleep()
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        srv.fail("/s.jsonl", status=503, times=2)
        (f,) = run(
            [src(srv, "/s.jsonl", data)], tmp_path, base_delay=0.5, sleep=sleep
        ).entries
    assert f.status == "fetched" and f.requests == 3
    assert sleep.calls == [0.5, 1.0]


def test_a_404_is_not_retried(tmp_path):
    # WHY: a wrong URL will not fix itself. One request, no wait, and a
    #      FetchError that says retrying is pointless (exit 65), so the
    #      durable workflow does not retry the activity either.
    # KIND: boundary
    # CATCHES: s04
    # CHAPTER: data.01 section 5, Pitfalls, item 5
    sleep = FakeSleep()
    with FlakyHTTP({"/s.jsonl": b"x"}) as srv:
        srv.fail("/s.jsonl", status=404, times=5)
        with pytest.raises(FetchError) as err:
            run([src(srv, "/s.jsonl", b"x")], tmp_path, sleep=sleep)
        hits = srv.hits("/s.jsonl")
    assert err.value.retryable is False
    assert err.value.source_id == "s"
    assert hits == 1 and sleep.calls == []


def test_exhausted_retries_raise_a_retryable_error(tmp_path):
    # WHY: when every attempt hits a 503 the mirror may be back in an hour:
    #      the error is retryable (exit 75), raised after exactly `attempts`
    #      requests and no wait after the last.
    # KIND: boundary
    # CATCHES: s11, m01
    sleep = FakeSleep()
    with FlakyHTTP({"/s.jsonl": b"x"}) as srv:
        srv.fail("/s.jsonl", status=503, times=50)
        with pytest.raises(FetchError) as err:
            run(
                [src(srv, "/s.jsonl", b"x")],
                tmp_path,
                attempts=3,
                base_delay=0.1,
                sleep=sleep,
            )
        hits = srv.hits("/s.jsonl")
    assert err.value.retryable is True
    assert hits == 3 and sleep.calls == [0.1, 0.2]


def test_a_silent_server_times_out(tmp_path):
    # WHY: a server that accepts and then says nothing would hang the run
    #      forever. `timeout` bounds the wait for each byte, and a timeout is
    #      retryable.
    # KIND: fault
    # CATCHES: s22
    with FlakyHTTP({"/s.jsonl": b"x"}) as srv:
        srv.delay("/s.jsonl", 1.0)
        with pytest.raises(FetchError) as err:
            run([src(srv, "/s.jsonl", b"x")], tmp_path, attempts=1, timeout=0.1)
    assert err.value.retryable is True


# --- concurrency and order -----------------------------------------------------------


class CountingServer:
    """An HTTP/1.1 server in the current event loop: each path answers with
    the body registered for it after `delay` seconds and counts how many
    requests are in flight at once. A path in `redirect` answers 302 with
    that Location instead."""

    def __init__(self, bodies: dict[str, bytes], delay: float = 0.05, redirect=None):
        self.bodies, self.delay = bodies, delay
        self.redirect = redirect or {}
        self.in_flight = self.max_in_flight = 0
        self.requests: list[str] = []

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    async def _handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return
        path = head.split(b" ", 2)[1].decode()
        self.requests.append(path)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if path in self.redirect:
                writer.write(
                    b"HTTP/1.1 302 Found\r\nLocation: %s\r\nContent-Length: 0\r\n\r\n"
                    % self.redirect[path].encode()
                )
            else:
                body = self.bodies[path]
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s"
                    % (len(body), body)
                )
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            self.in_flight -= 1
            writer.close()


def test_concurrency_is_bounded(tmp_path):
    # WHY: eight sources with concurrency 3 run exactly three at a time:
    #      one at a time means the fetches wait on each other (or block the
    #      event loop); eight at a time means the mirror sees no bound.
    # KIND: unit
    # CATCHES: s10
    bodies = {f"/f{i}.jsonl": stories(2, f"f{i}") for i in range(8)}

    async def go():
        async with CountingServer(bodies, delay=0.05) as srv:
            srcs = [
                Source(f"f{i}", srv.url(p), sha(b), "CC0-1.0")
                for i, (p, b) in enumerate(bodies.items())
            ]
            m = await fetch(
                srcs, tmp_path, concurrency=3, clock=FakeClock(), sleep=FakeSleep()
            )
            return srv, m

    srv, m = asyncio.run(go())
    assert srv.max_in_flight == 3, (
        f"{srv.max_in_flight} requests in flight at once; want 3"
    )
    assert all(e.status == "fetched" for e in m.entries)


def test_ledger_rows_follow_the_input_order(tmp_path):
    # WHY: the ledger and the manifest are compared across runs (MS-corpus
    #      wants a deterministic output hash). Downloads finish in whatever
    #      order the network decides; rows must still follow the config.
    #      Here the first source is the slowest.
    # KIND: property
    # CATCHES: s06
    a, b, c = stories(1, "a"), stories(1, "b"), stories(1, "c")
    with FlakyHTTP({"/a.jsonl": a, "/b.jsonl": b, "/c.jsonl": c}) as srv:
        srv.delay("/a.jsonl", 0.3)
        srv.delay("/b.jsonl", 0.15)
        srcs = [
            src(srv, "/a.jsonl", a),
            src(srv, "/b.jsonl", b),
            src(srv, "/c.jsonl", c),
        ]
        m = run(srcs, tmp_path, concurrency=3)
    assert [r["source_id"] for r in ledger(tmp_path)] == ["a", "b", "c"]
    assert [e.source.id for e in m.entries] == ["a", "b", "c"]


def test_a_failure_keeps_the_rows_of_finished_sources(tmp_path):
    # WHY: one bad source fails the run, but a source that finished must
    #      still get its ledger row: its raw directory exists, and without
    #      the row the next run could not see it as cached and would
    #      download it again.
    # KIND: fault
    # CATCHES: s15
    a = stories(2, "a")
    with FlakyHTTP({"/a.jsonl": a, "/gone.jsonl": b"x"}) as srv:
        srv.fail("/gone.jsonl", status=404, times=1)
        srv.delay("/gone.jsonl", 0.3)
        with pytest.raises(FetchError):
            run(
                [src(srv, "/a.jsonl", a), src(srv, "/gone.jsonl", b"x")],
                tmp_path,
                concurrency=2,
            )
    assert [r["source_id"] for r in ledger(tmp_path)] == ["a"]


def test_redirects_are_followed(tmp_path):
    # WHY: dataset hosts answer a file URL with 302 to a CDN. The fetch
    #      follows the Location header; the ledger keeps the URL you
    #      configured, not the CDN's.
    # KIND: unit
    # CATCHES: s20
    data = stories(2)

    async def go():
        async with CountingServer({"/cdn/s.jsonl": data}, delay=0.0) as srv:
            srv.redirect = {"/s.jsonl": "/cdn/s.jsonl"}
            s = Source("s", srv.url("/s.jsonl"), sha(data), "CC0-1.0")
            m = await fetch([s], tmp_path, clock=FakeClock(), sleep=FakeSleep())
            return srv, s, m

    srv, s, m = asyncio.run(go())
    assert srv.requests == ["/s.jsonl", "/cdn/s.jsonl"]
    assert m.entries[0].status == "fetched"
    assert ledger(tmp_path)[0]["url"] == s.url


# --- documents and config -------------------------------------------------------------


def test_text_sources_split_on_blank_lines():
    # WHY: a `text` source is one document per block of non-blank lines; a
    #      line holding only spaces is blank too. The line number is where
    #      the block starts, so a raw document points back into its file.
    # KIND: unit
    # CATCHES: s16
    data = b"The cat sat.\nIt purred.\n\n  \nA dog ran.\n\n\nEnd\n"
    assert list(split_documents(data, "text")) == [
        (1, "The cat sat.\nIt purred."),
        (5, "A dog ran."),
        (8, "End"),
    ]


def test_jsonl_errors_name_the_line(tmp_path):
    # WHY: a corrupt source is a data error: say where (the line), and do
    #      not retry (FetchError with retryable False, exit 65).
    # KIND: boundary
    # CATCHES: m04
    with pytest.raises(ValueError, match="line 2"):
        list(split_documents(b'{"text": "a"}\n{"txt": "b"}\n', "jsonl"))
    with pytest.raises(ValueError, match="line 3"):
        list(split_documents(b'{"text": "a"}\n\nnot json\n', "jsonl"))
    with pytest.raises(ValueError, match="line 2"):
        list(split_documents(b"ok\n\xff\xfe\n", "text"))
    bad = b'{"text": "a"}\n{"text": 7}\n'
    with FlakyHTTP({"/bad.jsonl": bad}) as srv:
        with pytest.raises(FetchError) as err:
            run([src(srv, "/bad.jsonl", bad)], tmp_path)
    assert err.value.retryable is False and "line 2" in str(err.value)


def test_parts_hold_at_most_part_docs_documents(tmp_path):
    # WHY: raw parts bound the memory of every later stage; 5 documents with
    #      part_docs=2 make parts of 2, 2, and 1, and read_raw returns them
    #      in order across part boundaries.
    # KIND: boundary
    # CATCHES: s12
    data = stories(5)
    with FlakyHTTP({"/s.jsonl": data}) as srv:
        (f,) = run([src(srv, "/s.jsonl", data)], tmp_path, part_docs=2).entries
    names = sorted(p.name for p in Path(f.raw_dir).iterdir())
    assert names == [
        "part-00000.jsonl.zst",
        "part-00001.jsonl.zst",
        "part-00002.jsonl.zst",
    ]
    assert [d["url"].rsplit("#", 1)[1] for d in read_raw(f.raw_dir)] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]


def test_sources_from_config_fills_defaults_and_rejects_bad_entries():
    # WHY: the corpus config is the run's input; defaults come from
    #      corpus-config.schema.json, and a duplicate id or a malformed
    #      checksum must fail before any download.
    # KIND: unit
    # CATCHES: m03
    cfg = {
        "sources": [
            {
                "id": "a",
                "url": "http://h/a",
                "sha256": "0" * 64,
                "license_spdx": "CC0-1.0",
            },
            {
                "id": "b",
                "url": "http://h/b",
                "sha256": "f" * 64,
                "license_spdx": "MIT",
                "allowed_uses": ["eval"],
                "format": "text",
            },
        ]
    }
    a, b = sources_from_config(cfg)
    assert a == Source(
        "a", "http://h/a", "0" * 64, "CC0-1.0", ("train", "eval"), "jsonl", "scrub"
    )
    assert (b.allowed_uses, b.format) == (("eval",), "text")
    dup = {"sources": [cfg["sources"][0], dict(cfg["sources"][0])]}
    with pytest.raises(ValueError, match="duplicate"):
        sources_from_config(dup)
    for bad in (
        {"sha256": "0" * 63},
        {"sha256": "A" * 64},
        {"format": "csv"},
        {"license_spdx": ""},
    ):
        with pytest.raises(ValueError):
            sources_from_config({"sources": [dict(cfg["sources"][0], **bad)]})
