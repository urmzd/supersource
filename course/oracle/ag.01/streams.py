"""Fixture SSE streams for ag.01 and the deltas a client must decode from them.

    python course/oracle/ag.01/streams.py   # rewrites course/fixtures/ag.01/streams.json

Written to the contract by hand (openai-subset.v1.yaml, Streaming and Tool
calls): no network, no third-party output. The expected deltas are derived
here from the contract's rules, independently of the Go reference:

- a non-empty `delta.content` is one text delta; an empty one is nothing;
- the first fragment of tool call `index` (it carries `id` and
  `function.name`) opens the call; every non-empty `function.arguments`
  fragment is an args delta for that index; fragments of different calls may
  interleave;
- a chunk with `finish_reason` closes every open call in index order (end
  deltas carry the assembled call; empty arguments become `{}`);
- a usage chunk (`choices: []`) is a usage delta;
- `[DONE]` closes any call still open, then ends the stream with `done`;
- `data: {"error": ...}` ends it with `error`, and so does EOF before
  `[DONE]` (kind `no_done`): a cut stream is never a quiet success.

Each entry: name, sse (exact bytes as text), deltas (list of objects:
text/start/args/end/usage/done/error), and the read sizes the test replays
it at.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "fixtures" / "ag.01" / "streams.json"


def ev(obj) -> str:
    return (
        "data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n\n"
    )


def chunk(delta: dict, finish=None, cid="chatcmpl-1") -> str:
    return ev(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def usage(p: int, c: int, cid="chatcmpl-1") -> str:
    return ev(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": {
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": p + c,
            },
        }
    )


def frag(
    index: int, args: str, cid: str | None = None, name: str | None = None
) -> dict:
    f: dict = {"index": index, "function": {"arguments": args}}
    if cid is not None:
        f["id"] = cid
        f["type"] = "function"
        f["function"]["name"] = name
    return {"tool_calls": [f]}


DONE = "data: [DONE]\n\n"

STREAMS = []

# 1. Text: the chapter's worked example. "Hel", "lo", an empty-content
#    chunk (a multi-byte character still incomplete), finish, usage.
STREAMS.append(
    {
        "name": "text",
        "sse": chunk({"role": "assistant", "content": ""})
        + chunk({"content": "Hel"})
        + chunk({"content": "lo"})
        + chunk({"content": ""})
        + chunk({}, "stop")
        + usage(5, 2)
        + DONE,
        "deltas": [
            {"type": "text", "text": "Hel"},
            {"type": "text", "text": "lo"},
            {"type": "usage", "in": 5, "out": 2},
            {"type": "done", "finish": "stop"},
        ],
    }
)

# 2. Two tool calls whose argument fragments interleave.
STREAMS.append(
    {
        "name": "tool_interleaved",
        "sse": chunk({"role": "assistant", "content": None})
        + chunk(frag(0, "", "call_a", "get_weather"))
        + chunk(frag(1, '{"tz"', "call_b", "get_time"))
        + chunk(frag(0, '{"city":'))
        + chunk(frag(1, ':"UTC"}'))
        + chunk(frag(0, '"Paris"}'))
        + chunk({}, "tool_calls")
        + usage(12, 9)
        + DONE,
        "deltas": [
            {"type": "start", "index": 0, "id": "call_a", "name": "get_weather"},
            {"type": "start", "index": 1, "id": "call_b", "name": "get_time"},
            {"type": "args", "index": 1, "fragment": '{"tz"'},
            {"type": "args", "index": 0, "fragment": '{"city":'},
            {"type": "args", "index": 1, "fragment": ':"UTC"}'},
            {"type": "args", "index": 0, "fragment": '"Paris"}'},
            {
                "type": "end",
                "index": 0,
                "id": "call_a",
                "name": "get_weather",
                "args": '{"city":"Paris"}',
            },
            {
                "type": "end",
                "index": 1,
                "id": "call_b",
                "name": "get_time",
                "args": '{"tz":"UTC"}',
            },
            {"type": "usage", "in": 12, "out": 9},
            {"type": "done", "finish": "tool_calls"},
        ],
    }
)

# 3. Text then a call with no arguments, ping comments, CRLF line ends, and
#    one event whose JSON is split over two data lines (joined with "\n").
crlf = (
    ": ping\r\n\r\n"
    + (
        chunk({"role": "assistant", "content": ""})
        + chunk({"content": "Let me check."})
    ).replace("\n", "\r\n")
    + ": ping\n\n"
    + 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,\n'
    + 'data: "model":"m","choices":[{"index":0,"delta":'
    + json.dumps(frag(0, "", "call_c", "list_models"), separators=(",", ":"))
    + ',"finish_reason":null}]}\n\n'
    + chunk({}, "tool_calls")
    + DONE
)
STREAMS.append(
    {
        "name": "ping_crlf_multiline",
        "sse": crlf,
        "deltas": [
            {"type": "text", "text": "Let me check."},
            {"type": "start", "index": 0, "id": "call_c", "name": "list_models"},
            {
                "type": "end",
                "index": 0,
                "id": "call_c",
                "name": "list_models",
                "args": "{}",
            },
            {"type": "done", "finish": "tool_calls"},
        ],
    }
)

# 4. An error event after the first token: the stream ends with an error.
STREAMS.append(
    {
        "name": "error_event",
        "sse": chunk({"role": "assistant", "content": ""})
        + chunk({"content": "Hi"})
        + ev(
            {
                "error": {
                    "message": "engine lost",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            }
        ),
        "deltas": [
            {"type": "text", "text": "Hi"},
            {"type": "error", "kind": "stream_error"},
        ],
    }
)

# 5. EOF after finish_reason but before [DONE].
STREAMS.append(
    {
        "name": "no_done",
        "sse": chunk({"role": "assistant", "content": ""})
        + chunk({"content": "Hi"})
        + chunk({}, "stop"),
        "deltas": [
            {"type": "text", "text": "Hi"},
            {"type": "error", "kind": "no_done"},
        ],
    }
)

# 6. A call left open at [DONE] (no finish_reason chunk): closed at [DONE].
STREAMS.append(
    {
        "name": "open_call_at_done",
        "sse": chunk(frag(0, '{"q":"x"}', "call_d", "search_docs")) + DONE,
        "deltas": [
            {"type": "start", "index": 0, "id": "call_d", "name": "search_docs"},
            {"type": "args", "index": 0, "fragment": '{"q":"x"}'},
            {
                "type": "end",
                "index": 0,
                "id": "call_d",
                "name": "search_docs",
                "args": '{"q":"x"}',
            },
            {"type": "done", "finish": ""},
        ],
    }
)

for s in STREAMS:
    s["read_sizes"] = [1, 7, 65536]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"streams": STREAMS}, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"wrote {OUT} ({len(STREAMS)} streams)")


if __name__ == "__main__":
    main()
