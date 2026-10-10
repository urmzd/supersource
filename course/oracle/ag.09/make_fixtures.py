"""Fixtures for ag.09 (the eval runner), computed independently of the Go
reference.

    uv run python course/oracle/ag.09/make_fixtures.py

Writes course/fixtures/ag.09/:

  bootstrap.json
      percentile bootstrap intervals of the mean, by the rule the chapter
      states: the stream is PCG32 seeded with SplitMix64's finalizer of
      seed + 4 * 0x9E3779B97F4A7C15 on sequence 4 (spec/pcg32.md, purpose
      "sample"); each resample draws len(groups) group indices with
      below(len(groups)) and averages every value of the drawn groups; the
      sorted means are read at floor(B * alpha / 2) and
      ceil(B * (1 - alpha / 2)) - 1. Uses the frozen generator,
      course/tests/_lib/pcg32.py.
  suite.jsonl
      a small eval suite in formats/eval-case.schema.json, written by hand.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
COURSE = HERE.parents[2]
sys.path.insert(0, str(COURSE / "tests" / "_lib"))
from pcg32 import PCG32  # noqa: E402

OUT = COURSE / "fixtures" / "ag.09"
M64 = (1 << 64) - 1


def mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def stream(seed: int, purpose: int) -> PCG32:
    return PCG32(mix64((seed + purpose * 0x9E3779B97F4A7C15) & M64), purpose)


def mean(xs):
    s = 0.0
    for x in xs:
        s += x
    return s / len(xs)


def bootstrap(groups, n_boot: int, alpha: float, seed: int):
    allv = [x for g in groups for x in g]
    m = mean(allv)
    r = stream(seed, 4)
    means = []
    for _ in range(n_boot):
        s, c = 0.0, 0
        for _ in range(len(groups)):
            g = groups[r.below(len(groups))]
            for x in g:
                s += x
            c += len(g)
        means.append(s / c)
    means.sort()
    lo = math.floor(n_boot * alpha / 2)
    hi = math.ceil(n_boot * (1 - alpha / 2)) - 1
    return m, means[lo], means[hi]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = PCG32(2026, 54)
    cases = []
    binary = [[1.0 if r.uniform() < 0.7 else 0.0] for _ in range(20)]
    cases.append({"name": "binary-20", "groups": binary, "n_boot": 2000, "alpha": 0.05, "seed": 0})
    grouped = [[round(r.uniform(), 3) for _ in range(3)] for _ in range(10)]
    cases.append({"name": "grouped-10x3", "groups": grouped, "n_boot": 1000, "alpha": 0.05, "seed": 7})
    cont = [[round(r.normal(), 4)] for _ in range(30)]
    cases.append({"name": "normal-30-alpha10", "groups": cont, "n_boot": 500, "alpha": 0.1, "seed": 123456789})
    for c in cases:
        c["mean"], c["lo"], c["hi"] = bootstrap(c["groups"], c["n_boot"], c["alpha"], c["seed"])
    (OUT / "bootstrap.json").write_text(json.dumps({"cases": cases}, indent=1) + "\n")

    suite = [
        {"case_id": "docsqa-001", "input": "Which hash names a KV block?",
         "ground_truth": {"answer": "chained FNV-1a 64", "chunks": ["docs/kv-cache.md#2"]},
         "tags": ["rag"], "scorer_args": {"regex": {"pattern": "FNV-1a"}}},
        {"case_id": "docsqa-002", "input": "What does a 429 from the gateway carry?",
         "ground_truth": {"answer": "Retry-After", "chunks": ["docs/gateway.md#2"]},
         "tags": ["rag"], "scorer_args": {"regex": {"pattern": "Retry-After"}}},
        {"case_id": "chat-003", "input": {"messages": [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "Is temperature zero greedy?"}]},
         "ground_truth": "yes", "tags": ["chat"], "scorer_args": {}},
        {"case_id": "tool-004", "input": "How many requests did acme make today?",
         "ground_truth": {"tool": "query_usage"}, "tags": ["tools"], "scorer_args": {"tool_called": {"name": "query_usage"}}},
        {"case_id": "plain-005", "input": "Say hello.", "tags": [], "scorer_args": {}},
    ]
    (OUT / "suite.jsonl").write_text("".join(json.dumps(c) + "\n" for c in suite))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
