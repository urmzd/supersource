"""Fixture tracer gateway: checks the API key, proxies to the engine."""

import os
import urllib.error
import urllib.request

from common import Handler, config, serve

CFG = {}


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
        else:
            self.error(404, "not found", "invalid_request_error", "not_found")

    def do_POST(self):
        key = os.environ.get("TL_API_KEY")
        if not key or self.headers.get("Authorization") != f"Bearer {key}":
            return self.error(401, "missing or unknown API key", "invalid_request_error", "invalid_api_key")
        n = int(self.headers.get("Content-Length") or 0)
        req = urllib.request.Request(CFG["upstream"] + self.path, data=self.rfile.read(n), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            resp = e
        self.send_response(resp.status if hasattr(resp, "status") else resp.code)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
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
