"""Recompute every auto-checked answer of review.01 independently (see _confirm.py).

q1 reads the model shape from SmolLM2-135M's own config.json when it is in the
Hugging Face cache (else the published values) and measures the bytes of one
token's KV entry as a numpy float16 array, instead of multiplying by hand.
q2 allocates sequences block by block until the pool is exhausted. q3 is a
deterministic event simulation: arrivals every 1/4 s, each request in flight
from arrival to its last token, averaged exactly over one steady-state period
with Fractions. q4 to q6 count the budget in minutes and hours with Fractions.

    uv run --project course/harness python course/oracle/solve/review.01.py
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def smollm2_shape() -> tuple[int, int, int, str]:
    hub = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--HuggingFaceTB--SmolLM2-135M-Instruct"
    )
    for cfg in sorted(hub.glob("snapshots/*/config.json")):
        d = json.loads(cfg.read_text())
        heads = d["num_attention_heads"]
        head_dim = d.get("head_dim") or d["hidden_size"] // heads
        return d["num_hidden_layers"], d["num_key_value_heads"], head_dim, str(cfg)
    return 30, 3, 64, "published values (config.json not in the local cache)"


def fr(x: F) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def main() -> int:
    c: dict[str, str] = {}

    layers, kv_heads, head_dim, src = smollm2_shape()
    print(
        f"q1 shape from {src}: layers {layers}, kv heads {kv_heads}, head dim {head_dim}"
    )
    one_token = np.zeros(
        (2, layers, kv_heads, head_dim), dtype=np.float16
    )  # key and value
    per_token = one_token.nbytes
    c["q1"] = str(per_token)

    blocks, block_size, seq = 2048, 16, 512
    pool = np.zeros(blocks, dtype=bool)  # True = allocated
    c["q2.a"] = str(blocks * block_size * per_token)
    fitted = 0
    while True:
        need = -(-seq // block_size)
        free = np.flatnonzero(~pool)
        if len(free) < need:
            break
        pool[free[:need]] = True
        fitted += 1
    c["q2.b"] = str(fitted)

    # q3: deterministic arrivals every 1/4 s; request i is in flight on
    # [i/4, i/4 + E2E), E2E = TTFT + 31 gaps. Average the in-flight count over
    # one period [T, T + 1/4) far from the start, exactly.
    ttft, tpot, n_out = F(1, 2), F(6, 100), 32
    e2e = ttft + (n_out - 1) * tpot
    gap = F(1, 4)
    t0 = 100 * gap
    events = []
    for i in range(0, 200):
        start, end = i * gap, i * gap + e2e
        lo, hi = max(start, t0), min(end, t0 + gap)
        if hi > lo:
            events.append(hi - lo)
    c["q3"] = fr(sum(events) / gap)

    period_min = 30 * 24 * 60
    budget = F(5, 1000) * period_min
    c["q4"] = fr(budget)
    burn = F(144, 10)
    spent_per_hour = burn * 60 / period_min  # fraction of the budget per hour at rate b
    c["q5"] = fr(spent_per_hour)
    c["q6"] = fr(1 / spent_per_hour)
    return confirm("review.01", c)


if __name__ == "__main__":
    sys.exit(main())
