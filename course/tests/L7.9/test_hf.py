"""Course tests for L7.9: the Hugging Face downloader (tinyllm/io/hf.py).

Annotated exemplars (DESIGN 5.12). No test touches the network: a local
HTTP server on 127.0.0.1 plays the Hub. It serves /<repo>/resolve/<rev>/<file>
the way huggingface.co does: a large file answers with a 302 redirect to a
"CDN" path, and the redirect carries X-Linked-Etag (the sha256) and
X-Linked-Size; Range requests get 206. Switches make it ignore ranges, cut a
response short, or serve corrupt bytes, and it records every request.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from tinyllm.io.hf import hf_download

BLOB = bytes(range(256)) * 40 + b"tail"  # 10244 bytes
SMALL = b'{"vocab_size": 256}\n'
SHA = hashlib.sha256(BLOB).hexdigest()


class Hub:
    def __init__(self) -> None:
        self.files = {"model.safetensors": BLOB, "config.json": SMALL}
        self.requests: list[tuple[str, str | None]] = []
        self.ignore_range = False
        self.cut_after: int | None = None  # serve only this many bytes, once
        self.corrupt = False
        hub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                hub.requests.append((self.path, self.headers.get("Range")))
                parts = self.path.strip("/").split("/")
                if parts[0] == "cdn":
                    return self.body(hub.files[parts[1]])
                if (
                    len(parts) != 5
                    or parts[2] != "resolve"
                    or parts[4] not in hub.files
                ):
                    self.send_response(404)
                    self.end_headers()
                    return
                name = parts[4]
                if (
                    name == "config.json"
                ):  # small files are served directly, no LFS headers
                    return self.body(hub.files[name])
                self.send_response(302)
                self.send_header("Location", f"/cdn/{name}")
                self.send_header(
                    "X-Linked-Etag", f'"{hashlib.sha256(hub.files[name]).hexdigest()}"'
                )
                self.send_header("X-Linked-Size", str(len(hub.files[name])))
                self.send_header("Content-Length", "0")
                self.end_headers()

            def body(self, data: bytes):
                if hub.corrupt:
                    data = data[:-1] + b"X"
                rng = self.headers.get("Range")
                start = (
                    int(rng.split("=")[1].split("-")[0])
                    if rng and not hub.ignore_range
                    else 0
                )
                chunk = data[start:]
                if hub.cut_after is not None:
                    chunk, hub.cut_after = chunk[: hub.cut_after], None
                    self.send_response(206 if start else 200)
                    if start:
                        self.send_header(
                            "Content-Range",
                            f"bytes {start}-{len(data) - 1}/{len(data)}",
                        )
                    self.send_header(
                        "Content-Length", str(len(data) - start)
                    )  # promises more than it sends
                    self.end_headers()
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.send_response(206 if start else 200)
                if start:
                    self.send_header(
                        "Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}"
                    )
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        ).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hub():
    h = Hub()
    yield h
    h.close()


@pytest.fixture
def cache():
    with tempfile.TemporaryDirectory(prefix="l79-hf-") as d:
        yield Path(d)


def test_hand_example_layout_and_redirect(hub, cache):
    # WHY: the chapter's walk through one download: the URL
    #      {endpoint}/{repo}/resolve/{rev}/{file}, the 302 to the CDN whose
    #      X-Linked-Etag is the sha256, and the cache layout
    #      <cache>/owner--name/<rev>/<file>. Both files land intact; no .part
    #      file is left.
    # KIND: unit
    # CATCHES: m10
    # CHAPTER: L7.9 section 3, Worked example by hand
    d = hf_download(
        "org/tiny",
        ["config.json", "model.safetensors"],
        str(cache),
        revision="abc123",
        endpoint=hub.url,
    )
    assert Path(d) == cache / "org--tiny" / "abc123"
    assert (Path(d) / "model.safetensors").read_bytes() == BLOB
    assert (Path(d) / "config.json").read_bytes() == SMALL
    assert not list(Path(d).glob("*.part"))
    assert hub.requests[0] == ("/org/tiny/resolve/abc123/config.json", None)
    assert ("/cdn/model.safetensors", None) in hub.requests


def test_resume_from_part_file(hub, cache):
    # WHY: a dropped connection must not restart a 270 MB download: the
    #      second call sends Range: bytes=<what is there>-, appends the rest,
    #      and the sha256 over the WHOLE file (old bytes included) checks out.
    # KIND: fault
    # CATCHES: s16, s17, m10
    # CHAPTER: L7.9 section 2.7, Downloading weights
    hub.cut_after = 4000
    with pytest.raises(Exception):
        hf_download("org/tiny", ["model.safetensors"], str(cache), endpoint=hub.url)
    part = cache / "org--tiny" / "main" / "model.safetensors.part"
    assert part.exists() and part.stat().st_size == 4000
    assert not (cache / "org--tiny" / "main" / "model.safetensors").exists()
    hub.requests.clear()
    d = hf_download("org/tiny", ["model.safetensors"], str(cache), endpoint=hub.url)
    assert (Path(d) / "model.safetensors").read_bytes() == BLOB
    assert ("/cdn/model.safetensors", "bytes=4000-") in hub.requests


def test_server_that_ignores_ranges(hub, cache):
    # WHY: a server may answer a Range request with 200 and the whole file;
    #      appending that to the partial bytes would corrupt the file, so the
    #      download starts over.
    # KIND: fault
    # CATCHES: s14
    # CHAPTER: L7.9 section 2.7, Downloading weights
    part = cache / "org--tiny" / "main" / "model.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(BLOB[:1000])
    hub.ignore_range = True
    d = hf_download("org/tiny", ["model.safetensors"], str(cache), endpoint=hub.url)
    assert (Path(d) / "model.safetensors").read_bytes() == BLOB


def test_corrupt_bytes_are_rejected(hub, cache):
    # WHY: the published sha256 is the only proof the weights are the ones
    #      the parity numbers were computed on. A corrupt file raises, the
    #      .part file is deleted, and nothing appears under the final name.
    #      An explicit sha256 argument is checked too (here for a small file
    #      with no X-Linked-Etag).
    # KIND: fault
    # CATCHES: s13, s15
    # CHAPTER: L7.9 section 2.7, Downloading weights
    hub.corrupt = True
    with pytest.raises(ValueError):
        hf_download("org/tiny", ["model.safetensors"], str(cache), endpoint=hub.url)
    base = cache / "org--tiny" / "main"
    assert (
        not (base / "model.safetensors").exists()
        and not (base / "model.safetensors.part").exists()
    )
    hub.corrupt = False
    with pytest.raises(ValueError):
        hf_download(
            "org/tiny",
            ["config.json"],
            str(cache),
            endpoint=hub.url,
            sha256={"config.json": "0" * 64},
        )
    assert not (base / "config.json").exists()


def test_existing_files_are_not_downloaded_again(hub, cache):
    # WHY: `pull` twice costs nothing the second time when the file is there
    #      and matches its pinned sha256; a stale file with a known hash is
    #      replaced.
    # KIND: unit
    # CATCHES: s18, m10
    # CHAPTER: L7.9 section 2.7, Downloading weights
    pins = {"model.safetensors": SHA}
    hf_download(
        "org/tiny", ["model.safetensors"], str(cache), endpoint=hub.url, sha256=pins
    )
    hub.requests.clear()
    hf_download(
        "org/tiny", ["model.safetensors"], str(cache), endpoint=hub.url, sha256=pins
    )
    assert hub.requests == []
    (cache / "org--tiny" / "main" / "model.safetensors").write_bytes(b"stale")
    hf_download(
        "org/tiny", ["model.safetensors"], str(cache), endpoint=hub.url, sha256=pins
    )
    assert (cache / "org--tiny" / "main" / "model.safetensors").read_bytes() == BLOB


def test_endpoint_from_environment_and_validation(hub, cache, monkeypatch):
    # WHY: HF_ENDPOINT points every download at a mirror (or this fake hub)
    #      without code changes; a repo id without an owner and a filename
    #      that climbs out of the cache are refused before any request.
    # KIND: boundary
    # CATCHES: m05, m06
    # CHAPTER: L7.9 section 4, The interface
    monkeypatch.setenv("HF_ENDPOINT", hub.url)
    d = hf_download("org/tiny", ["config.json"], str(cache))
    assert (Path(d) / "config.json").read_bytes() == SMALL
    n = len(hub.requests)
    for repo, files in (
        ("tiny", ["config.json"]),
        ("org/tiny", ["../escape"]),
        ("org/tiny", ["/abs"]),
    ):
        with pytest.raises(ValueError):
            hf_download(repo, files, str(cache))
    assert len(hub.requests) == n
    assert os.environ["HF_ENDPOINT"] == hub.url
