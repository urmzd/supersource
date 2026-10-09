"""Fixture engine, API v0 and v1 (+ /v2 side by side): /v1/completions,
/v1/chat/completions with streaming, stop sequences, seeds, tool calls,
usage, /v1/models, and /metrics with tl_engine_active_sequences."""

import json
import threading
import time

import bigram
from common import Handler, config, serve

MODEL = "tracer"
ACTIVE = [0]
LOCK = threading.Lock()
UNSUPPORTED = ("n", "logprobs", "functions", "best_of")


def _active(delta: int) -> None:
    with LOCK:
        ACTIVE[0] += delta


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class Engine(Handler):
    def do_GET(self):
        if self.path in ("/healthz", "/readyz"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        elif self.path == "/metrics":
            body = f"# TYPE tl_engine_active_sequences gauge\ntl_engine_active_sequences {ACTIVE[0]}\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "forge"}]})
        else:
            self.error(404, "not found", "invalid_request_error", "not_found")

    def do_POST(self):
        if self.path == "/v1/completions":
            return self.completions()
        if self.path in ("/v1/chat/completions", "/v2/chat/completions"):
            return self.chat(v2=self.path.startswith("/v2/"))
        return self.error(404, "not found", "invalid_request_error", "not_found")

    # -- v0 ---------------------------------------------------------------------

    def completions(self):
        req = self.body()
        if req is None:
            return self.error(400, "malformed JSON", "invalid_request_error")
        temp = req.get("temperature", 1.0)
        n = req.get("max_tokens", 16)
        if not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
            return self.error(400, "temperature must be in [0, 2]", "invalid_request_error", param="temperature")
        if not isinstance(n, int) or n < 1:
            return self.error(400, "max_tokens must be >= 1", "invalid_request_error", param="max_tokens")
        prompt = str(req.get("prompt", ""))
        ids = bigram.generate(prompt, n, float(temp), int(req.get("seed") or 0))
        base = {"id": "cmpl-0", "object": "text_completion", "created": int(time.time()), "model": req.get("model", "bigram")}
        if not req.get("stream"):
            usage = {"prompt_tokens": len(prompt.encode()), "completion_tokens": n, "total_tokens": len(prompt.encode()) + n}
            return self.send_json(200, {**base, "choices": [{"index": 0, "text": bytes(ids).decode(), "finish_reason": "length"}], "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i, t in enumerate(ids):
            fin = "length" if i == len(ids) - 1 else None
            chunk = {**base, "choices": [{"index": 0, "text": chr(t), "finish_reason": fin}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    # -- v1 chat ------------------------------------------------------------------

    def chat(self, v2: bool):
        req = self.body()
        if req is None:
            return self.error(400, "malformed JSON", "invalid_request_error")
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return self.error(400, "messages must be a non-empty array", "invalid_request_error", param="messages")
        for k in UNSUPPORTED:
            if k in req and not (k == "n" and req[k] == 1):
                return self.error(422, f"{k} is not supported", "invalid_request_error", "unsupported_parameter", k)
        temp = req.get("temperature", 1.0)
        n = req.get("max_tokens", 16)
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) or not 0 <= temp <= 2:
            return self.error(400, "temperature must be in [0, 2]", "invalid_request_error", param="temperature")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            return self.error(400, "max_tokens must be >= 1", "invalid_request_error", param="max_tokens")
        stops = req.get("stop") or []
        stops = [stops] if isinstance(stops, str) else [s for s in stops if s]
        templated = "".join(f"{m.get('role')}: {_text(m.get('content'))}\n" for m in msgs) + "assistant: "
        last = _text(msgs[-1].get("content"))
        base = {"id": "chatcmpl-0", "created": int(time.time()), "model": req.get("model", MODEL)}
        tools = req.get("tools") or []
        choice = req.get("tool_choice", "auto")
        call = None
        if tools and choice != "none":
            names = [t.get("function", {}).get("name") for t in tools]
            if isinstance(choice, dict):
                call = choice.get("function", {}).get("name")
            elif choice == "required" or (msgs[-1].get("role") == "user" and "weather" in last.lower()):
                call = names[0]
        _active(+1)
        try:
            if call:
                return self.tool_reply(base, call, req.get("stream"), len(templated.encode()), v2)
            ids = bigram.generate(last, n, float(temp), int(req.get("seed") or 0))
            if not req.get("stream"):
                text, fin, used = "", "length", 0
                for t in ids:
                    text += chr(t)
                    used += 1
                    hit = min((text.find(s) for s in stops if s in text), default=-1)
                    if hit >= 0:
                        text, fin = text[:hit], "stop"
                        break
                usage = {"prompt_tokens": len(templated.encode()), "completion_tokens": used, "total_tokens": len(templated.encode()) + used}
                body = {**base, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": fin}], "usage": usage}
                if v2:
                    body["api_version"] = 2
                return self.send_json(200, body)
            self.stream(base, ids, stops)
        finally:
            _active(-1)

    def _chunk(self, base, delta, fin=None) -> None:
        obj = {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def stream(self, base, ids, stops) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        hold = max((len(s) for s in stops), default=1) - 1
        try:
            self._chunk(base, {"role": "assistant", "content": ""})
            text, sent, fin = "", 0, "length"
            for t in ids:
                time.sleep(0.003)
                text += chr(t)
                hit = min((text.find(s) for s in stops if s in text), default=-1)
                if hit >= 0:
                    text, fin = text[:hit], "stop"
                    break
                safe = len(text) - hold
                if safe > sent:
                    self._chunk(base, {"content": text[sent:safe]})
                    sent = safe
            if len(text) > sent:
                self._chunk(base, {"content": text[sent:]})
            self._chunk(base, {}, fin)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client left: stop generating

    def tool_reply(self, base, name, stream, prompt_tokens, v2) -> None:
        args = json.dumps({"city": "Paris"})
        if not stream:
            body = {**base, "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                                 "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": name, "arguments": args}}]},
                                 "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}}
            if v2:
                body["api_version"] = 2
            return self.send_json(200, body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._chunk(base, {"role": "assistant", "content": None})
        self._chunk(base, {"tool_calls": [{"index": 0, "id": "call_0", "type": "function", "function": {"name": name, "arguments": args[:6]}}]})
        self._chunk(base, {"tool_calls": [{"index": 0, "function": {"arguments": args[6:]}}]})
        self._chunk(base, {}, "tool_calls")
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    cfg = config("engine")
    serve(Engine, cfg["http_listen"], cfg["health_listen"])
