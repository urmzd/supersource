"""lang.08 course tests: the bounded concurrent downloader (primers/lang.08/downloader.py).

Annotated exemplars (DESIGN 5.12). Two servers stand in for the network,
both on 127.0.0.1 and both started by the tests:

    FlakyHTTP        the course fixture server (course/testkit/python/sstestkit):
                     a thread outside the event loop that fails on request
                     (503s, bodies cut short, slow answers)
    CountingServer   below: an asyncio server inside the test's own event
                     loop that counts how many requests are in flight at once

No test sleeps for real to wait for something; the backoff sleeps go
through a fake `sleep` that only records what it was asked.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from sstestkit.flakyhttp import FlakyHTTP

import downloader
from downloader import HTTPError, Response, download_all, fetch, http_get, is_retryable

BODY = b"hello world"


class FakeSleep:
    """Records each requested delay and returns at once."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class CountingServer:
    """An HTTP/1.1 server in the current event loop. Each request answers its
    own path as the body after `delay` seconds, or, for a path in `hold`,
    never answers and records whether the client closed the connection."""

    def __init__(self, delay: float = 0.05, hold: frozenset[str] = frozenset()) -> None:
        self.delay = delay
        self.hold = hold
        self.in_flight = 0
        self.max_in_flight = 0
        self.requests: list[str] = []
        self.closed_by_client: list[str] = []

    async def __aenter__(self) -> "CountingServer":
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        self.server.close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
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
            if path in self.hold:
                try:
                    data = await asyncio.wait_for(reader.read(1), timeout=10)
                except (TimeoutError, ConnectionError):
                    return
                if data == b"":
                    self.closed_by_client.append(path)
                return
            await asyncio.sleep(self.delay)
            body = path.encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
                % (len(body), body)
            )
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            self.in_flight -= 1
            writer.close()


# --- http_get: one request over asyncio streams --------------------------------


def test_http_get_reads_status_headers_and_body():
    # WHY: the section 3 exchange: status 200, header names lower-cased so
    #      lookups do not depend on the server's capitalization, and exactly
    #      Content-Length bytes of body.
    # KIND: unit
    # CHAPTER: lang.08 section 3, Worked example by hand
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        resp = asyncio.run(http_get(srv.url("/hello.txt")))
    assert isinstance(resp, Response)
    assert resp.status == 200
    assert resp.headers["content-length"] == "11"
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.body == BODY


def test_http_get_sends_extra_headers():
    # WHY: a Range header asks for the rest of a file from a byte offset;
    #      data.01 resumes cut downloads this way, so headers must reach the
    #      server unchanged and a 206 must come back as a response.
    # KIND: unit
    # CHAPTER: lang.08 section 4
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        resp = asyncio.run(
            http_get(srv.url("/hello.txt"), headers={"Range": "bytes=6-"})
        )
        sent = srv.headers("/hello.txt")[0]
    assert sent.get("Range") == "bytes=6-"
    assert resp.status == 206
    assert resp.body == b"world"


def test_http_get_returns_an_error_status_as_a_response():
    # WHY: http_get reports what the server said; deciding that 404 is a
    #      failure is fetch's job. Mixing the two makes retries impossible
    #      to reason about.
    # KIND: boundary
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        resp = asyncio.run(http_get(srv.url("/missing")))
    assert resp.status == 404


def test_http_get_never_returns_a_short_body():
    # WHY: a connection cut mid-body must be an error. `reader.read(n)`
    #      returns whatever arrived, so a short body would be returned as if
    #      it were the whole file; `readexactly` raises IncompleteReadError.
    # KIND: boundary
    # CHAPTER: lang.08 section 5, Pitfalls, item 3
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        srv.truncate("/hello.txt", after=4, times=1)
        with pytest.raises(asyncio.IncompleteReadError):
            asyncio.run(http_get(srv.url("/hello.txt")))


# --- fetch: retries, backoff, timeouts ---------------------------------------------


def test_is_retryable_classifies_failures():
    # WHY: retrying a 404 wastes time and hides a broken URL; not retrying a
    #      503 or a cut connection turns a mirror's hiccup into a failed run.
    #      The table in section 2.6, row by row.
    # KIND: unit
    # CHAPTER: lang.08 section 2.6
    url = "http://x/y"
    assert is_retryable(HTTPError(url, 503))
    assert is_retryable(HTTPError(url, 500))
    assert is_retryable(HTTPError(url, 429))
    assert not is_retryable(HTTPError(url, 404))
    assert not is_retryable(HTTPError(url, 400))
    assert is_retryable(asyncio.IncompleteReadError(b"abc", 10))
    assert is_retryable(ConnectionRefusedError())
    assert is_retryable(ConnectionResetError())
    assert is_retryable(TimeoutError())
    assert not is_retryable(ValueError("malformed status line"))


def test_fetch_retries_5xx_with_exponential_backoff():
    # WHY: the section 3 retry timeline: two 503s, then success. The waits
    #      are base_delay * 2**(n-1) after attempt n (0.1 s, then 0.2 s), so
    #      a struggling server sees fewer and fewer requests.
    # KIND: unit
    # CHAPTER: lang.08 section 3, Worked example by hand
    sleep = FakeSleep()
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        srv.fail("/hello.txt", status=503, times=2)
        body = asyncio.run(
            fetch(srv.url("/hello.txt"), attempts=3, base_delay=0.1, sleep=sleep)
        )
        hits = srv.hits("/hello.txt")
    assert body == BODY
    assert hits == 3
    assert sleep.calls == [0.1, 0.2]


def test_fetch_does_not_retry_a_404():
    # WHY: a client error will not fix itself; one request, no wait, and
    #      the HTTPError carries the status for the caller.
    # KIND: boundary
    # CHAPTER: lang.08 section 5, Pitfalls, item 4
    sleep = FakeSleep()
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        srv.fail("/hello.txt", status=404, times=5)
        with pytest.raises(HTTPError) as err:
            asyncio.run(fetch(srv.url("/hello.txt"), attempts=3, sleep=sleep))
        hits = srv.hits("/hello.txt")
    assert err.value.status == 404
    assert hits == 1
    assert sleep.calls == []


def test_fetch_gives_up_after_its_attempts():
    # WHY: retries are bounded: `attempts` requests in all, no wait after
    #      the last one, and the last failure is raised (not swallowed into
    #      an empty body).
    # KIND: boundary
    sleep = FakeSleep()
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        srv.fail("/hello.txt", status=503, times=10)
        with pytest.raises(HTTPError) as err:
            asyncio.run(
                fetch(srv.url("/hello.txt"), attempts=3, base_delay=0.1, sleep=sleep)
            )
        hits = srv.hits("/hello.txt")
    assert err.value.status == 503
    assert hits == 3
    assert sleep.calls == [0.1, 0.2]


def test_fetch_retries_a_cut_body():
    # WHY: a connection reset mid-body is the most common failure on a long
    #      download; it is retryable, and the retry returns the full body.
    # KIND: fault
    sleep = FakeSleep()
    with FlakyHTTP({"/hello.txt": BODY}) as srv:
        srv.truncate("/hello.txt", after=3, times=1)
        body = asyncio.run(fetch(srv.url("/hello.txt"), attempts=3, sleep=sleep))
        hits = srv.hits("/hello.txt")
    assert body == BODY
    assert hits == 2
    assert len(sleep.calls) == 1


def test_fetch_times_out_each_attempt():
    # WHY: a server that accepts and then says nothing would otherwise hang
    #      the download forever. Each attempt gets its own deadline, and a
    #      timeout counts as retryable.
    # KIND: fault
    # CHAPTER: lang.08 section 5, Pitfalls, item 5
    sleep = FakeSleep()
    with FlakyHTTP({"/slow.txt": BODY}) as srv:
        srv.delay("/slow.txt", 0.6)
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            asyncio.run(
                fetch(srv.url("/slow.txt"), attempts=2, timeout=0.05, sleep=sleep)
            )
        elapsed = time.monotonic() - t0
        # Wait until both requests reached the server's handler threads.
        deadline = time.monotonic() + 2
        while srv.hits("/slow.txt") < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        hits = srv.hits("/slow.txt")
    assert hits == 2
    assert elapsed < 0.5, (
        f"two 0.05 s attempts took {elapsed:.2f} s: is the timeout per attempt?"
    )
    assert len(sleep.calls) == 1


# --- download_all: tasks, a semaphore, cancellation -------------------------------------


def test_download_all_keeps_limit_requests_in_flight():
    # WHY: the point of the exercise. Twelve downloads with limit 3 must run
    #      exactly three at a time: one at a time means the coroutines await
    #      each other in turn (pitfall 1); twelve at a time means there is no
    #      bound at all, as with a semaphore made inside each task (pitfall 2).
    # KIND: unit
    # CHAPTER: lang.08 section 3, Worked example by hand
    paths = [f"/f{i:02d}" for i in range(12)]

    async def go():
        async with CountingServer(delay=0.05) as srv:
            got = await download_all([srv.url(p) for p in paths], limit=3)
            return srv, got

    srv, got = asyncio.run(go())
    assert srv.max_in_flight == 3, (
        f"at most {srv.max_in_flight} request(s) were in flight at once; limit=3 wants exactly 3"
    )
    assert [got[srv.url(p)] for p in paths] == [p.encode() for p in paths]


def test_download_all_keeps_order_and_fetches_each_url_once():
    # WHY: the result is a dict in the order the urls were first given, so
    #      callers can zip it with their input; a url listed twice is
    #      downloaded once.
    # KIND: unit
    paths = ["/b", "/a", "/b", "/c"]

    async def go():
        async with CountingServer(delay=0.0) as srv:
            got = await download_all([srv.url(p) for p in paths], limit=2)
            return srv, got

    srv, got = asyncio.run(go())
    assert list(got) == [srv.url("/b"), srv.url("/a"), srv.url("/c")]
    assert sorted(srv.requests) == ["/a", "/b", "/c"]


def test_a_failure_cancels_the_other_downloads():
    # WHY: when one download fails for good, the run has failed; the others
    #      must be cancelled, closing their connections, rather than left
    #      running in the background with nobody waiting for them. That is
    #      what asyncio.TaskGroup does and asyncio.gather does not. The
    #      failure itself is raised unwrapped, as an HTTPError.
    # KIND: fault
    # CHAPTER: lang.08 section 5, Pitfalls, item 6
    hold = frozenset({"/h1", "/h2", "/h3"})

    async def go(flaky: FlakyHTTP):
        async with CountingServer(hold=hold) as srv:
            urls = [srv.url(p) for p in sorted(hold)] + [flaky.url("/gone")]
            t0 = time.monotonic()
            with pytest.raises(HTTPError) as err:
                await download_all(urls, limit=4, attempts=1)
            elapsed = time.monotonic() - t0
            await asyncio.sleep(0.2)  # let the server see the closed connections
            me = asyncio.current_task()
            lingering = [
                t
                for t in asyncio.all_tasks()
                if t is not me
                and not t.done()
                and "CountingServer" not in t.get_coro().__qualname__
            ]
            return err.value, elapsed, sorted(srv.closed_by_client), lingering

    with FlakyHTTP({"/gone": b"x"}) as flaky:
        flaky.fail("/gone", status=404, times=1)
        flaky.delay("/gone", 0.3)  # fail only once the other three are connected
        err, elapsed, closed, lingering = asyncio.run(go(flaky))
    assert err.status == 404
    assert elapsed < 2.0
    assert lingering == [], (
        f"tasks still running after download_all raised: {lingering}"
    )
    assert closed == sorted(hold), f"connections closed by the client: {closed}"


def test_limit_must_be_positive():
    # WHY: Semaphore(0) never lets anything through, so limit=0 would hang
    #      forever instead of failing; reject it up front.
    # KIND: boundary
    async def go():
        # A hang (the symptom) fails after 2 s instead of blocking the suite.
        return await asyncio.wait_for(
            download_all(["http://127.0.0.1:9/x"], limit=0), 2.0
        )

    with pytest.raises(ValueError):
        asyncio.run(go())


def test_module_imports_only_the_standard_library():
    # WHY: the primer is about asyncio itself; aiohttp or httpx would hide
    #      the event loop this chapter teaches (section 4, rules).
    # KIND: unit
    import ast
    import inspect
    import sys

    tree = ast.parse(inspect.getsource(downloader))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    outside = sorted(
        n for n in names if n not in sys.stdlib_module_names and n != "__future__"
    )
    assert not outside, f"downloader.py imports outside the standard library: {outside}"
