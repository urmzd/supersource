"""Fixture gateway, API v0 and v1: key auth with scopes, a token-bucket rate
limit, model routing against the upstream's /v1/models, a temperature-0
response cache (X-TL-Cache), and the internal X-TL-Priority header."""

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

from common import Handler, config, serve

CFG = {}
CACHE = {}
BUCKETS = {}
LOCK = threading.Lock()
RATE, BURST = 20.0, 20.0


def _models() -> set:
    try:
        with urllib.request.urlopen(CFG["upstream"] + "/v1/models", timeout=5) as r:
            return {m["id"] for m in json.load(r)["data"]}
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return set()


def _allow(key: str) -> bool:
    with LOCK:
        tokens, t = BUCKETS.get(key, (BURST, time.monotonic()))
        now = time.monotonic()
        tokens = min(BURST, tokens + (now - t) * RATE)
        if tokens < 1:
            BUCKETS[key] = (tokens, now)
            return False
        BUCKETS[key] = (tokens - 1, now)
        return True


class Gateway(Handler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok\n")
        elif self.path == "/readyz":
            try:
                urllib.request.urlopen(CFG["upstream"] + "/healthz", timeout=2).read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ready\n")
            except (urllib.error.URLError, OSError):
                self.send_response(503)
                self.end_headers()
        elif self.path == "/v1/models":
            if self.auth() is None:
                return
            self.forward("GET", b"")
        else:
            self.error(404, "not found", "invalid_request_error", "not_found")

    def auth(self) -> str | None:
        got = self.headers.get("Authorization", "")
        full, noscope = os.environ.get("TL_API_KEY"), os.environ.get("TL_API_KEY_NOSCOPE")
        if full and got == f"Bearer {full}":
            return "standard"
        if noscope and got == f"Bearer {noscope}":
            self.error(403, "this key may not call chat", "permission_error", "insufficient_scope")
            return None
        self.error(401, "missing or unknown API key", "invalid_request_error", "invalid_api_key")
        return None

    def do_POST(self):
        tier = self.auth()
        if tier is None:
            return
        if not _allow(self.headers.get("Authorization", "")):
            body = json.dumps({"error": {"message": "rate limited", "type": "rate_limit_error", "param": None, "code": "rate_limit_exceeded"}}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "1")
            self.send_header("x-ratelimit-limit-requests", str(int(BURST)))
            self.send_header("x-ratelimit-remaining-requests", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self.error(400, "malformed JSON", "invalid_request_error")
        if self.path.endswith("/chat/completions"):
            if req.get("model") not in _models():
                return self.error(404, f"model {req.get('model')!r} is not served", "invalid_request_error", "model_not_found", "model")
            if req.get("temperature", 1) == 0 and not req.get("stream"):
                key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
                hit = CACHE.get(key)
                if hit is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("X-TL-Cache", "hit")
                    self.send_header("Content-Length", str(len(hit)))
                    self.end_headers()
                    self.wfile.write(hit)
                    return
                return self.forward("POST", raw, tier, cache_key=key)
        self.forward("POST", raw, tier)

    def forward(self, method, raw, tier="standard", cache_key=None):
        req = urllib.request.Request(
            CFG["upstream"] + self.path,
            data=raw if method == "POST" else None,
            method=method,
            headers={"Content-Type": "application/json", "X-TL-Priority": tier},  # never the client's
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            resp = e
        status = resp.status if hasattr(resp, "status") else resp.code
        ctype = resp.headers.get("Content-Type", "application/json")
        if cache_key is not None and status == 200:
            body = resp.read()
            CACHE[cache_key] = body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("X-TL-Cache", "miss")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        while True:
            chunk = resp.read(256)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()


if __name__ == "__main__":
    CFG.update(config("gateway"))
    serve(Gateway, CFG["listen"], CFG["health_listen"])
