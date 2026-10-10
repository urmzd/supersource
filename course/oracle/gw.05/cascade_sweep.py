"""Fixture for gw.05's cascade threshold sweep (synthetic, seeded).

    python course/oracle/gw.05/cascade_sweep.py   # rewrites course/fixtures/gw.05/cascade_sweep.json

200 evaluation prompts of a two-step cascade (small model, then large).
Each has the small model's 8 token logprobs and whether each model's answer
was correct. The data are synthetic: a per-prompt difficulty d in [0, 1)
lowers the small model's logprobs and its chance of being right, so a
confident small answer is more often correct, which is the property a
cascade exploits. The expected curve is computed here, independently of the
Go code: for each tau, accept the small answer when mean_logprob >= tau,
else escalate; cost counts the small call on every prompt (small_cost) plus
1 per escalation; accuracy is the fraction of correct final answers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

N, TOKENS, SMALL_COST = 200, 8, 0.1
TAUS = [-3.0 + 0.25 * i for i in range(13)]  # -3.0 .. 0.0


def samples() -> list[dict]:
    rng = random.Random(20261009)
    out = []
    for _ in range(N):
        d = rng.random()
        lps = [
            round(-(0.05 + 2.5 * d) * rng.uniform(0.5, 1.5), 6) for _ in range(TOKENS)
        ]
        out.append(
            {
                "small_logprobs": lps,
                "small_correct": rng.random() < 0.95 - 0.8 * d,
                "large_correct": rng.random() < 0.92,
            }
        )
    return out


def curve(ss: list[dict]) -> list[dict]:
    pts = []
    for tau in TAUS:
        esc = cost = ok = 0.0
        for s in ss:
            cost += SMALL_COST
            lp = s["small_logprobs"]
            if lp and sum(lp) / len(lp) >= tau:
                ok += s["small_correct"]
                continue
            esc += 1
            cost += 1
            ok += s["large_correct"]
        pts.append(
            {"tau": tau, "escalated": esc / N, "cost": cost / N, "accuracy": ok / N}
        )
    return pts


def main() -> None:
    ss = samples()
    doc = {
        "generator": "course/oracle/gw.05/cascade_sweep.py",
        "small_cost": SMALL_COST,
        "taus": TAUS,
        "samples": ss,
        "curve": curve(ss),
    }
    dst = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "gw.05"
        / "cascade_sweep.json"
    )
    dst.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(dst, dst.stat().st_size)


if __name__ == "__main__":
    main()
