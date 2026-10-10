"""Fixture tracer engine: OpenAI subset v0 over HTTP+SSE (DESIGN 2.6)."""

import json
import time

import bigram
from common import Handler, config, serve


class Engine(Handler):
    def do_GET(self):
        if self.path in ("/healthz", "/readyz"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.error(404, "not found", "invalid_request_error", "not_found")

    def do_POST(self):
        if self.path != "/v1/completions":
            return self.error(404, "not found", "invalid_request_error", "not_found")
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
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for i, t in enumerate(ids):
            fin = "length" if i == len(ids) - 1 else None
            chunk = {**base, "choices": [{"index": 0, "text": chr(t), "finish_reason": fin}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    cfg = config("engine")
    serve(Engine, cfg["http_listen"], cfg["health_listen"])
