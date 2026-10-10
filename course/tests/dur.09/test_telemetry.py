"""Course tests for dur.09: OTLP/HTTP export from TRACEPARENT (python/tinyllm/io/telemetry.py).

Rung R0. The D9 exception: the only platform service a Python activity talks
to is the OpenTelemetry collector. These tests run a tiny OTLP/HTTP receiver
in a thread (standard library only) and check what reaches it: the trace and
parent ids from TRACEPARENT, the JSON encoding of otel/semconv.md spans, the
sampling flag, and that a dead collector never fails the work.

The chapter's worked example (section 3, telemetry): TRACEPARENT is the W3C
example 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01, ids are drawn
from a counter (span ids 0000000000000001, 0000000000000002, ...), and the
clock reads 1760000000000000000 ns then steps by 1000 ns:

    train.run   trace 4bf9...4736  span ...0001  parent 00f067aa0ba902b7
    train.step  trace 4bf9...4736  span ...0002  parent ...0001   tl.train.step=50
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from tinyllm.io.telemetry import (
    SpanContext,
    Tracer,
    format_traceparent,
    otlp_traces_body,
    parse_traceparent,
    should_sample_step,
)

TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
TRACE, PARENT = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"


class Sink:
    """A minimal OTLP/HTTP receiver: records (path, content-type, JSON body)."""

    def __init__(self, status: int = 200) -> None:
        self.got: list[tuple[str, str, dict]] = []
        sink = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n)
                sink.got.append((self.path, self.headers.get("Content-Type", ""), json.loads(body)))
                self.send_response(status)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()

    def spans(self) -> list[dict]:
        out = []
        for path, _, body in self.got:
            if path == "/v1/traces":
                for rs in body["resourceSpans"]:
                    for ss in rs["scopeSpans"]:
                        out += ss["spans"]
        return out


@pytest.fixture
def sink():
    s = Sink()
    yield s
    s.close()


def counter_ids():
    n = [0]

    def ids(k: int) -> bytes:
        n[0] += 1
        return n[0].to_bytes(k, "big")

    return ids


def stepping_clock(start: int = 1760000000000000000, step: int = 1000):
    t = [start - step]

    def now() -> int:
        t[0] += step
        return t[0]

    return now


def attrs(span: dict) -> dict:
    out = {}
    for a in span["attributes"]:
        (kind, v), = a["value"].items()
        out[a["key"]] = int(v) if kind == "intValue" else v
    return out


def test_hand_example_span_tree(sink):
    # WHY: the chapter's worked example (section 3): the activity span's
    #      context arrives as TRACEPARENT, train.run is its child in the same
    #      trace, and a sampled train.step is train.run's child. Exactly these
    #      ids reach the collector, as OTLP JSON (hex ids, times as decimal
    #      strings of nanoseconds, kind 1 INTERNAL, intValue as a string).
    # KIND: conformance
    # CATCHES: s30, s31
    # CHAPTER: dur.09 section 3, worked example (telemetry)
    tr = Tracer(sink.url, "forge-python", parse_traceparent(TP), {"service.namespace": "forge"},
                ids=counter_ids(), clock_ns=stepping_clock())
    with tr.span("train.run", {"tl.run.id": "train-1"}) as run_span:
        for step in (49, 50):
            if should_sample_step(step):
                with tr.span("train.step", {"tl.train.step": step, "tl.train.loss": 2.5}):
                    pass
    assert run_span.parent_span_id == PARENT
    assert tr.flush() is True
    path, ctype, body = sink.got[0]
    assert (path, ctype) == ("/v1/traces", "application/json")
    res = {a["key"]: a["value"] for a in body["resourceSpans"][0]["resource"]["attributes"]}
    assert res["service.name"] == {"stringValue": "forge-python"}
    assert res["service.namespace"] == {"stringValue": "forge"}
    step_span, run = sink.spans()
    assert run["name"] == "train.run" and step_span["name"] == "train.step"
    assert run["traceId"] == step_span["traceId"] == TRACE
    assert run["spanId"] == "0000000000000001" and run["parentSpanId"] == PARENT
    assert step_span["spanId"] == "0000000000000002" and step_span["parentSpanId"] == "0000000000000001"
    assert run["kind"] == 1 and step_span["kind"] == 1
    assert run["startTimeUnixNano"] == "1760000000000000000"
    assert int(step_span["endTimeUnixNano"]) >= int(step_span["startTimeUnixNano"])
    assert attrs(step_span) == {"tl.train.step": 50, "tl.train.loss": 2.5}
    assert step_span["attributes"][0]["value"] == {"intValue": "50"}


@pytest.mark.parametrize(
    "value,want",
    [
        (TP, (TRACE, PARENT, True)),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00", (TRACE, PARENT, False)),
        ("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra", (TRACE, PARENT, True)),
        (None, None),
        ("", None),
        ("garbage", None),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7", None),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra", None),
        ("ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", None),
        ("00-00000000000000000000000000000000-00f067aa0ba902b7-01", None),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01", None),
        ("00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01", None),
        ("00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01", None),
        ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1", None),
    ],
)
def test_parse_traceparent(value, want):
    # WHY: W3C trace-context is strict on purpose: an id of all zeros, a
    #      wrong length, uppercase hex, or version ff means "invalid, start a
    #      new trace". Accepting them glues unrelated runs into one trace.
    #      Version 00 has exactly four fields; later versions may add more.
    # KIND: boundary
    # CATCHES: s32
    # CHAPTER: dur.09 section 2.6
    got = parse_traceparent(value)
    if want is None:
        assert got is None
    else:
        assert (got.trace_id, got.span_id, got.sampled) == want


def test_format_roundtrip():
    # WHY: the helper also hands a context on (to a grandchild process or a
    #      log line); formatting and parsing must agree.
    # KIND: unit
    # CHAPTER: dur.09 section 2.6
    for sampled in (True, False):
        ctx = SpanContext(TRACE, PARENT, sampled)
        s = format_traceparent(ctx)
        assert s == f"00-{TRACE}-{PARENT}-{'01' if sampled else '00'}"
        back = parse_traceparent(s)
        assert (back.trace_id, back.span_id, back.sampled) == (TRACE, PARENT, sampled)


def test_unsampled_parent_exports_nothing(sink):
    # WHY: the sampling decision belongs to the root of the trace. A worker
    #      that did not sample this run passes flags 00, and the Python side
    #      must honor it: exporting anyway creates orphan spans whose parents
    #      were never recorded.
    # KIND: boundary
    # CATCHES: s33
    # CHAPTER: dur.09 section 2.6
    tr = Tracer(sink.url, "p", parse_traceparent(TP[:-2] + "00"))
    with tr.span("train.run"):
        pass
    tr.gauge("tl.train.loss", 2.0)
    assert tr.queued() == []
    assert tr.flush() is True and sink.got == []


def test_no_traceparent_starts_a_root(sink):
    # WHY: run by hand (no worker), or with an invalid TRACEPARENT, the run
    #      is its own root trace: a fresh trace id and no parent, never a
    #      parent id of zeros.
    # KIND: boundary
    # CHAPTER: dur.09 section 2.6
    tr = Tracer.from_env({"OTEL_EXPORTER_OTLP_ENDPOINT": sink.url, "TRACEPARENT": "bogus"})
    assert tr.parent is None and tr.service_name == "tinyllm-python"
    with tr.span("train.run") as s:
        pass
    assert s.parent_span_id == "" and len(s.trace_id) == 32 and s.trace_id != "0" * 32
    tr.flush()
    assert "parentSpanId" not in sink.spans()[0]


def test_from_env(sink):
    # WHY: the runner passes everything through the environment
    #      (spec/subprocess-activity.md); from_env is the one place that
    #      reads it.
    # KIND: unit
    # CATCHES: s34
    # CHAPTER: dur.09 section 4, The interface
    tr = Tracer.from_env({
        "OTEL_EXPORTER_OTLP_ENDPOINT": sink.url,
        "OTEL_SERVICE_NAME": "forge-python",
        "OTEL_RESOURCE_ATTRIBUTES": "service.namespace=forge, service.version=0.1.0",
        "TRACEPARENT": TP,
    })
    assert (tr.service_name, tr.parent.trace_id, tr.sampled) == ("forge-python", TRACE, True)
    with tr.span("corpus.stage shard", {"tl.corpus.stage": "shard"}):
        pass
    tr.flush()
    res = {a["key"]: a["value"]["stringValue"] for a in sink.got[0][2]["resourceSpans"][0]["resource"]["attributes"]}
    assert res == {"service.name": "forge-python", "service.namespace": "forge", "service.version": "0.1.0"}


def test_dead_collector_never_fails_the_work():
    # WHY: telemetry observes training; it must never stop it. A collector
    #      that is down (connection refused) makes flush return False within
    #      its timeout, never raise, and the queue is dropped, not retried
    #      forever (a training run would otherwise grow without bound).
    # KIND: fault
    # CATCHES: s35
    # CHAPTER: dur.09 section 5, Pitfalls
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tr = Tracer(f"http://127.0.0.1:{port}", "p", None, timeout_s=0.5)
    with tr.span("train.run"):
        pass
    tr.gauge("tl.train.loss", 1.0)
    assert tr.flush() is False
    assert tr.queued() == []


def test_collector_error_status_is_false():
    # WHY: a collector answering 500 (or 429) did not take the spans; flush
    #      must say so (False) so the caller can count drops.
    # KIND: fault
    # CATCHES: s35
    # CHAPTER: dur.09 section 5, Pitfalls
    bad = Sink(status=500)
    try:
        tr = Tracer(bad.url, "p", None)
        with tr.span("x"):
            pass
        assert tr.flush() is False
    finally:
        bad.close()


def test_error_marks_the_span(sink):
    # WHY: a failed stage must show as failed in the trace (status ERROR,
    #      error.type), and the exception must still propagate: telemetry
    #      never swallows the program's errors.
    # KIND: unit
    # CATCHES: s36
    # CHAPTER: dur.09 section 2.6
    tr = Tracer(sink.url, "p", parse_traceparent(TP))
    with pytest.raises(KeyError):
        with tr.span("corpus.stage pii"):
            raise KeyError("x")
    tr.flush()
    (span,) = sink.spans()
    assert span["status"] == {"code": 2} and attrs(span)["error.type"] == "KeyError"


def test_step_sampling():
    # WHY: train.step is sampled every 50th step (otel/semconv.md); one span
    #      per step would be millions of spans per run.
    # KIND: unit
    # CATCHES: s37
    # CHAPTER: dur.09 section 2.6
    assert [s for s in range(0, 201) if should_sample_step(s)] == [0, 50, 100, 150, 200]
    assert [s for s in range(1, 7) if should_sample_step(s, every=3)] == [3, 6]
    with pytest.raises(ValueError):
        should_sample_step(5, every=0)


def test_gauges_are_pushed(sink):
    # WHY: a subprocess cannot be scraped, so Python pushes its gauges
    #      (tl.train.loss, otel/metrics.yaml) with the spans; only the latest
    #      value per name and attribute set is sent.
    # KIND: unit
    # CATCHES: s38
    # CHAPTER: dur.09 section 2.6
    tr = Tracer(sink.url, "p", parse_traceparent(TP))
    tr.gauge("tl.train.loss", 3.0, {"tl.workflow.id": "train-1"})
    tr.gauge("tl.train.loss", 2.5, {"tl.workflow.id": "train-1"})
    assert tr.flush() is True
    ((path, _, body),) = sink.got
    assert path == "/v1/metrics"
    (metric,) = body["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    (dp,) = metric["gauge"]["dataPoints"]
    assert metric["name"] == "tl.train.loss" and dp["asDouble"] == 2.5


def test_shutdown_stops_export(sink):
    # WHY: after shutdown (at process exit) nothing is queued: spans ended
    #      by atexit handlers after the final flush are dropped, not left
    #      half-sent.
    # KIND: unit
    # CATCHES: s39
    # CHAPTER: dur.09 section 4, The interface
    tr = Tracer(sink.url, "p", None)
    with tr.span("a"):
        pass
    assert tr.shutdown() is True
    with tr.span("b"):
        pass
    assert tr.queued() == [] and [s["name"] for s in sink.spans()] == ["a"]


def test_body_shape_without_parent():
    # WHY: otlp_traces_body is the wire format; a root span has no
    #      parentSpanId key at all (an empty string is not a valid id).
    # KIND: unit
    # CATCHES: s31
    # CHAPTER: dur.09 section 2.6
    tr = Tracer(None, "svc", None, ids=counter_ids(), clock_ns=stepping_clock())
    s = tr.start_span("root", {"ok": True, "n": 3, "f": 0.5, "s": "x"})
    tr.end_span(s)
    body = otlp_traces_body([s], "svc")
    span = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert "parentSpanId" not in span
    assert body["resourceSpans"][0]["scopeSpans"][0]["scope"] == {"name": "tinyllm"}
    assert {a["key"]: a["value"] for a in span["attributes"]} == {
        "ok": {"boolValue": True}, "n": {"intValue": "3"}, "f": {"doubleValue": 0.5}, "s": {"stringValue": "x"},
    }
    assert tr.flush() is True, "no endpoint: nothing to send is success"
