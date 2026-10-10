"""The Python testkit (course/testkit/python/sstestkit): flakyhttp faults,
failpoints, and the fake clock."""

import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[2] / "testkit" / "python"
sys.path.insert(0, str(KIT))
from sstestkit import clock, failpoint  # noqa: E402
from sstestkit.flakyhttp import FlakyHTTP  # noqa: E402

DATA = bytes(range(256)) * 40


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def test_flakyhttp_5xx_then_ok_and_ranges():
    with FlakyHTTP({"/a.bin": DATA}) as srv:
        srv.fail("/a.bin", status=503, times=2)
        for _ in range(2):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(srv.url("/a.bin"))
            assert e.value.code == 503
        status, h, body = get(srv.url("/a.bin"))
        assert (
            status == 200
            and body == DATA
            and h["X-Content-Sha256"] == hashlib.sha256(DATA).hexdigest()
        )
        status, h, body = get(srv.url("/a.bin"), {"Range": "bytes=100-"})
        assert (
            status == 206
            and body == DATA[100:]
            and h["Content-Range"] == f"bytes 100-{len(DATA) - 1}/{len(DATA)}"
        )
        assert srv.hits("/a.bin") == 4


def test_flakyhttp_truncation_and_checksum_lies():
    with FlakyHTTP({"/a.bin": DATA}) as srv:
        srv.truncate("/a.bin", after=1000)
        with pytest.raises(Exception):  # IncompleteRead: the body was cut short
            get(srv.url("/a.bin"))
        srv.lie_checksum("/a.bin")
        _, h, body = get(srv.url("/a.bin"))
        assert (
            body == DATA and h["X-Content-Sha256"] != hashlib.sha256(DATA).hexdigest()
        )
        srv.no_ranges("/a.bin")
        status, h, body = get(srv.url("/a.bin"), {"Range": "bytes=10-19"})
        assert status == 200 and "Accept-Ranges" not in h and body == DATA


def test_failpoints():
    failpoint.load("a=error(disk full);b=2*error(second);c=off;s=sleep(1ms)")
    with pytest.raises(failpoint.FailpointError, match="disk full"):
        failpoint.inject("a")
    failpoint.inject("b")  # first evaluation: not yet
    with pytest.raises(failpoint.FailpointError):
        failpoint.inject("b")
    failpoint.inject("c")
    failpoint.inject("s")
    failpoint.inject("unknown")
    assert failpoint.count("b") == 2
    with pytest.raises(ValueError):
        failpoint.load("x=boom")
    code = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); from sstestkit import failpoint; failpoint.inject('w')",
            str(KIT),
        ],
        env={"TL_FAILPOINTS": "w=crash", "PATH": "/usr/bin:/bin"},
    ).returncode
    assert code == 137


def test_fake_clock():
    c = clock.FakeClock()
    t0, m0 = c.now(), c.monotonic()
    c.sleep(1.5)
    c.advance(0.5)
    assert c.now() - t0 == 2.0 and c.monotonic() - m0 == 2.0 and c.sleeps == [1.5]
    with pytest.raises(ValueError):
        c.advance(-1)
