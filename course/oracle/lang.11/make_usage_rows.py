"""Maintainer generator for course/fixtures/lang.11/usage_rows.jsonl.

Synthetic usage-ledger rows in the shape of contracts/formats/usage.v1.sql:
six hours of traffic from four tenants (one of them, o'brien, has a quote in
its name), refused requests with their codes, streams that failed after their first
byte (status 200 with an error code), NULL TTFTs for embeddings and
refusals, rows at exact hour boundaries, and two keys whose token totals tie.
Deterministic: a 64-bit LCG (Knuth's MMIX constants), no third-party code.

    python3 course/oracle/lang.11/make_usage_rows.py > course/fixtures/lang.11/usage_rows.jsonl
"""

import json
import sys

T0 = 1790848800000  # 2026-10-01T10:00:00Z in Unix milliseconds
HOUR = 3_600_000


class LCG:
    def __init__(self, seed: int) -> None:
        self.s = seed

    def next(self) -> int:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return self.s >> 33

    def below(self, n: int) -> int:
        return self.next() % n

    def pick(self, weighted):
        total = sum(w for _, w in weighted)
        x = self.below(total)
        for v, w in weighted:
            if x < w:
                return v
            x -= w
        raise AssertionError


def rows():
    r = LCG(11)
    tenants = [("acme", 40), ("globex", 30), ("o'brien", 15), ("initech", 15)]
    keys = {
        "acme": ["acmekeyaaaaa", "acmekeybbbbb"],
        "globex": ["globexkeyccc", "globexkeyddd"],
        "o'brien": ["obrienkeyeee", "obrienkeyfff"],
        "initech": ["initechkeygg", "initechkeyhh"],
    }
    times = sorted(r.below(6 * HOUR) for _ in range(232))
    times += [
        HOUR,
        2 * HOUR,
        3 * HOUR,
        3 * HOUR - 1,
    ]  # exact boundaries and one ms before
    out = []
    for i, dt in enumerate(sorted(times)):
        tenant = r.pick(tenants)
        model = r.pick([("smol", 60), ("tiny", 40)])
        route = r.pick(
            [
                ("/v1/chat/completions", 60),
                ("/v1/completions", 25),
                ("/v1/embeddings", 15),
            ]
        )
        status = r.pick([(200, 85), (429, 8), (400, 4), (503, 3)])
        stream = 0 if route == "/v1/embeddings" else r.below(2)
        row = {
            "request_id": f"req-{i:04d}",
            "ts_ms": T0 + dt,
            "tenant": tenant,
            "key_id": keys[tenant][r.below(2)],
            "model": model,
            "served_model": "",
            "route": route,
            "api_version": "1",
            "status": status,
            "error_code": None,
            "stream": stream,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "ttft_ms": None,
            "e2e_ms": float(1 + r.below(40)),
            "cache_hit": 0,
            "worker_id": "",
            "trace_id": "",
        }
        if status == 200:
            row["served_model"] = {"smol": "smol-135m@v3", "tiny": "tiny-10m@v1"}[model]
            if model == "smol" and r.below(10) == 0:
                row["served_model"] = "smol-135m@v4"  # the canary
            row["prompt_tokens"] = 5 + r.below(196)
            if route != "/v1/embeddings":
                row["completion_tokens"] = 1 + r.below(300)
                row["ttft_ms"] = (500 + r.below(8500)) / 10
            if r.below(5) == 0:
                row["cached_tokens"] = r.below(row["prompt_tokens"] + 1)
            row["e2e_ms"] = (row["ttft_ms"] or 0) + 20.0 * row["completion_tokens"]
            row["worker_id"] = f"w-{r.below(3)}"
            row["cache_hit"] = int(r.below(12) == 0)
            if stream and r.below(12) == 0:
                row["error_code"] = (
                    "upstream_error"  # a stream that failed after its first byte
                )
        else:
            row["error_code"] = {
                429: "rate_limit_exceeded",
                400: None,
                503: "no_capacity",
            }[status]
            if status == 400:
                row["error_code"] = "invalid_request_error"
        out.append(row)
    # Two keys of one tenant whose token totals tie inside 12:00 to 13:00.
    for j, key in enumerate(["tiekeyaaaaaa", "tiekeybbbbbb"]):
        for k in range(2):
            out.append(
                {
                    "request_id": f"tie-{j}{k}",
                    "ts_ms": T0 + 2 * HOUR + 600_000 * (k + 1) + j,
                    "tenant": "initech",
                    "key_id": key,
                    "model": "tiny",
                    "served_model": "tiny-10m@v1",
                    "route": "/v1/completions",
                    "api_version": "1",
                    "status": 200,
                    "error_code": None,
                    "stream": 0,
                    "prompt_tokens": 400 + 50 * k,
                    "completion_tokens": 600 - 50 * k,
                    "cached_tokens": 0,
                    "ttft_ms": 250.0,
                    "e2e_ms": 900.0,
                    "cache_hit": 0,
                    "worker_id": "w-0",
                    "trace_id": "",
                }
            )
    return out


if __name__ == "__main__":
    for row in rows():
        sys.stdout.write(json.dumps(row, sort_keys=True) + "\n")
