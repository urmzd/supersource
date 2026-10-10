"""Fixture SSE streams for gw.04 in the openai-subset.v1 framing.

    python course/oracle/gw.04/streams.py   # rewrites course/fixtures/gw.04/streams.json

Written to the contract by hand (no network, no third-party output): one
`data: <json>\\n\\n` event per chunk, all chunks of a stream with the same
id, the first with delta.role, the last content chunk with finish_reason,
an optional usage chunk with `choices: []`, then `data: [DONE]\\n\\n`. They
cover what a gateway must pass through untouched: multi-byte UTF-8 split
across chunks (an empty-content chunk while a character is incomplete),
a `: ping` comment, tool-call argument fragments, a text_completion stream,
and a stream without usage.

Each entry: name, sse (the exact bytes as text), events (data events,
[DONE] included; comments excluded), usage (the last usage object or null).
"""

from __future__ import annotations

import json
from pathlib import Path

CREATED = 1760000000


def ev(obj) -> str:
    return (
        "data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n\n"
    )


def chat(
    cid: str,
    model: str,
    deltas: list[dict],
    finish: str,
    usage: dict | None,
    ping_after: int | None = None,
) -> tuple[str, int, dict | None]:
    out, n = [], 0
    for i, d in enumerate(deltas):
        fr = finish if i == len(deltas) - 1 else None
        out.append(
            ev(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": CREATED,
                    "model": model,
                    "choices": [{"index": 0, "delta": d, "finish_reason": fr}],
                }
            )
        )
        n += 1
        if ping_after is not None and i == ping_after:
            out.append(": ping\n\n")
    if usage is not None:
        out.append(
            ev(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": CREATED,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
        )
        n += 1
    out.append("data: [DONE]\n\n")
    return "".join(out), n + 1, usage


def streams() -> list[dict]:
    res = []

    def add(name, triple):
        sse, n, usage = triple
        res.append({"name": name, "sse": sse, "events": n, "usage": usage})

    add(
        "chat-basic-usage",
        chat(
            "chatcmpl-1",
            "smol-135m",
            [
                {"role": "assistant", "content": ""},
                {"content": "Once"},
                {"content": " upon"},
                {"content": " a time"},
            ],
            "stop",
            {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        ),
    )
    add(
        "chat-utf8-split",
        chat(
            "chatcmpl-2",
            "smol-135m",
            [
                {"role": "assistant", "content": ""},
                {"content": "caf"},
                {"content": ""},
                {"content": "é"},
                {"content": " 東"},
                {"content": ""},
                {"content": "京"},
                {"content": " 🙂"},
            ],
            "length",
            {"prompt_tokens": 4, "completion_tokens": 7, "total_tokens": 11},
        ),
    )
    add(
        "chat-ping-no-usage",
        chat(
            "chatcmpl-3",
            "tinystories-10m",
            [
                {"role": "assistant", "content": ""},
                {"content": "The"},
                {"content": " cat"},
            ],
            "stop",
            None,
            ping_after=0,
        ),
    )
    add(
        "chat-tool-call-fragments",
        chat(
            "chatcmpl-4",
            "smol-135m-instruct",
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search_docs", "arguments": ""},
                        }
                    ],
                },
                {"tool_calls": [{"index": 0, "function": {"arguments": '{"query":'}}]},
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": ' "rate limits"}'}}
                    ]
                },
                {},
            ],
            "tool_calls",
            {"prompt_tokens": 61, "completion_tokens": 14, "total_tokens": 75},
        ),
    )
    text = []
    for i, t in enumerate(["H", "", "é", "!"]):
        text.append(
            ev(
                {
                    "id": "cmpl-5",
                    "object": "text_completion",
                    "created": CREATED,
                    "model": "tracer",
                    "choices": [
                        {
                            "index": 0,
                            "text": t,
                            "finish_reason": "length" if i == 3 else None,
                        }
                    ],
                }
            )
        )
    text.append(
        ev(
            {
                "id": "cmpl-5",
                "object": "text_completion",
                "created": CREATED,
                "model": "tracer",
                "choices": [],
                "usage": {
                    "prompt_tokens": 16,
                    "completion_tokens": 4,
                    "total_tokens": 20,
                },
            }
        )
    )
    text.append("data: [DONE]\n\n")
    res.append(
        {
            "name": "completions-text",
            "sse": "".join(text),
            "events": 6,
            "usage": {"prompt_tokens": 16, "completion_tokens": 4, "total_tokens": 20},
        }
    )
    return res


def main() -> None:
    dst = Path(__file__).resolve().parents[2] / "fixtures" / "gw.04" / "streams.json"
    dst.write_text(
        json.dumps(
            {"generator": "course/oracle/gw.04/streams.py", "streams": streams()},
            ensure_ascii=False,
            indent=1,
        )
        + "\n"
    )
    print(dst, dst.stat().st_size)


if __name__ == "__main__":
    main()
