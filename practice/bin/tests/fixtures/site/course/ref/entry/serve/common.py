"""Shared bits of the fixture servers: config, listen addresses, error shape."""

import argparse
import json
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def config(section: str) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config, "rb") as f:
        return tomllib.load(f)[section]


def addr(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return host or "127.0.0.1", int(port)


def serve(handler: type, *listens: str) -> None:
    servers = [ThreadingHTTPServer(addr(x), handler) for x in listens]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    servers[0].serve_forever()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, status: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error(self, status: int, message: str, typ: str, code=None, param=None) -> None:
        self.send_json(status, {"error": {"message": message, "type": typ, "param": param, "code": code}})

    def body(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            v = json.loads(self.rfile.read(n) or b"null")
        except json.JSONDecodeError:
            return None
        return v if isinstance(v, dict) else None
