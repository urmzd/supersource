"""otlpsink: an OTLP/HTTP trace receiver for tests (DESIGN 4.4 testkit, tracer subset).

    with Sink() as sink:                       # 127.0.0.1:<free port>
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = sink.endpoint
        ...
        spans = sink.wait(lambda s: len(s) >= 2, timeout=5)

It accepts POST /v1/traces in both OTLP/HTTP encodings:

  application/json        the JSON mapping (ids as hex strings, enums as
                          numbers or names), what a hand-written exporter sends
  application/x-protobuf  ExportTraceServiceRequest, what the OTel SDKs send

and keeps every span as a `Span`. `hang=True` makes it accept connections and
never answer, to prove an exporter never blocks the request path.

Only the fields the course checks are decoded: resource service.name, trace
and span ids, parent span id, name, kind, start and end times, and string or
int attributes. The protobuf field numbers are those of
opentelemetry/proto/trace/v1/trace.proto and common/v1/common.proto.
"""

from __future__ import annotations

import gzip
import http.server
import json
import threading
import time
from dataclasses import dataclass, field

KINDS = {
    "SPAN_KIND_UNSPECIFIED": 0,
    "SPAN_KIND_INTERNAL": 1,
    "SPAN_KIND_SERVER": 2,
    "SPAN_KIND_CLIENT": 3,
    "SPAN_KIND_PRODUCER": 4,
    "SPAN_KIND_CONSUMER": 5,
}


@dataclass
class Span:
    service: str
    trace_id: str  # 32 lowercase hex digits
    span_id: str  # 16 lowercase hex digits
    parent_span_id: str  # "" for a root span
    name: str
    kind: int  # 2 SERVER, 3 CLIENT
    start_ns: int
    end_ns: int
    attributes: dict = field(default_factory=dict)
    encoding: str = ""


# -- protobuf wire format (just enough) -----------------------------------------


def _varint(b: bytes, i: int) -> tuple[int, int]:
    shift = val = 0
    while True:
        byte = b[i]
        i += 1
        val |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return val, i
        shift += 7


def _fields(b: bytes):
    """Yield (field number, wire type, value) for one message."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = int.from_bytes(b[i : i + 8], "little"), i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i : i + n], i + n
        elif wt == 5:
            v, i = int.from_bytes(b[i : i + 4], "little"), i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield num, wt, v


def _pb_anyvalue(b: bytes):
    for num, wt, v in _fields(b):
        if num == 1 and wt == 2:
            return v.decode("utf-8", "replace")  # string_value
        if num == 2 and wt == 0:
            return bool(v)  # bool_value
        if num == 3 and wt == 0:
            return v - (1 << 64) if v >= 1 << 63 else v  # int_value
    return None


def _pb_attrs(blobs: list[bytes]) -> dict:
    out = {}
    for kv in blobs:
        key, val = "", None
        for num, wt, v in _fields(kv):
            if num == 1:
                key = v.decode("utf-8", "replace")
            elif num == 2:
                val = _pb_anyvalue(v)
        out[key] = val
    return out


def decode_protobuf(body: bytes) -> list[Span]:
    spans = []
    for num, _, rs in _fields(body):
        if num != 1:
            continue
        service, scope_spans = "", []
        for n2, _, v2 in _fields(rs):
            if n2 == 1:  # Resource
                service = str(
                    _pb_attrs([v for n3, _, v in _fields(v2) if n3 == 1]).get(
                        "service.name"
                    )
                    or ""
                )
            elif n2 == 2:
                scope_spans.append(v2)
        for ss in scope_spans:
            for n3, _, sp in _fields(ss):
                if n3 != 2:
                    continue
                f: dict = {"attrs": []}
                for n4, _, v in _fields(sp):
                    if n4 in (1, 2, 4):
                        f[n4] = v.hex()
                    elif n4 == 5:
                        f["name"] = v.decode("utf-8", "replace")
                    elif n4 in (6, 7, 8):
                        f[n4] = v
                    elif n4 == 9:
                        f["attrs"].append(v)
                spans.append(
                    Span(
                        service,
                        f.get(1, ""),
                        f.get(2, ""),
                        f.get(4, ""),
                        f.get("name", ""),
                        int(f.get(6, 0)),
                        int(f.get(7, 0)),
                        int(f.get(8, 0)),
                        _pb_attrs(f["attrs"]),
                        "protobuf",
                    )
                )
    return spans


# -- JSON mapping ------------------------------------------------------------------


def _json_value(v: dict):
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    return None


def decode_json(body: bytes) -> list[Span]:
    doc = json.loads(body)
    spans = []
    for rs in doc.get("resourceSpans") or []:
        attrs = {
            a.get("key"): _json_value(a.get("value") or {})
            for a in (rs.get("resource") or {}).get("attributes") or []
        }
        service = str(attrs.get("service.name") or "")
        for ss in rs.get("scopeSpans") or []:
            for sp in ss.get("spans") or []:
                kind = sp.get("kind", 0)
                kind = KINDS.get(kind, 0) if isinstance(kind, str) else int(kind)
                spans.append(
                    Span(
                        service,
                        str(sp.get("traceId", "")).lower(),
                        str(sp.get("spanId", "")).lower(),
                        str(sp.get("parentSpanId", "") or "").lower(),
                        str(sp.get("name", "")),
                        kind,
                        int(sp.get("startTimeUnixNano", 0) or 0),
                        int(sp.get("endTimeUnixNano", 0) or 0),
                        {
                            a.get("key"): _json_value(a.get("value") or {})
                            for a in sp.get("attributes") or []
                        },
                        "json",
                    )
                )
    return spans


# -- the server ----------------------------------------------------------------------


class Sink:
    def __init__(self, hang: bool = False) -> None:
        self.hang = hang
        self.spans: list[Span] = []
        self.errors: list[str] = []
        self.requests = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        sink = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with sink._lock:
                    sink.requests += 1
                if sink.hang:
                    sink._stop.wait(30)  # never answer while the test runs
                    return
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
                try:
                    if self.headers.get("Content-Encoding", "") == "gzip":
                        body = gzip.decompress(body)
                    if self.path.rstrip("/") != "/v1/traces":
                        raise ValueError(
                            f"POST {self.path}: OTLP traces go to /v1/traces"
                        )
                    if ctype == "application/json":
                        got, reply = decode_json(body), b"{}"
                    elif ctype == "application/x-protobuf":
                        got, reply = decode_protobuf(body), b""
                    else:
                        raise ValueError(f"unsupported Content-Type {ctype!r}")
                    with sink._lock:
                        sink.spans.extend(got)
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                except Exception as e:  # noqa: BLE001  a bad export is reported, not fatal
                    with sink._lock:
                        sink.errors.append(str(e))
                    reply = json.dumps({"error": str(e)}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def snapshot(self) -> list[Span]:
        with self._lock:
            return list(self.spans)

    def wait(self, done, timeout: float = 5.0) -> list[Span]:
        """Poll until done(spans) is true or the timeout passes; return the spans."""
        deadline = time.monotonic() + timeout
        while True:
            spans = self.snapshot()
            if done(spans) or time.monotonic() > deadline:
                return spans
            time.sleep(0.05)

    def __enter__(self) -> "Sink":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self.server.shutdown()
        self.server.server_close()
