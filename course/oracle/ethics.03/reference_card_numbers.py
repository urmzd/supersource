"""Maintainer script: the evaluation numbers of the reference model card
(course/ref/docs/MODEL_CARD.md), measured, not invented.

The reference card describes a model this script trains in seconds: a
byte-level Kneser-Ney 4-gram (L2.1's NGramLM) on the first 90% of the
tinyshakespeare fixture. It prints the card's evaluation rows:

  quality  bpb on the last 10% (bits per byte, t interval over byte NLLs, M07.4)
  safety   ethics.04's safety_report on the fixture prompts (Wilson intervals)
  bias     ethics.04's bias_report on the fixture probes (bootstrap intervals)

Run from the repo root with the reference on the path:

    PYTHONPATH=course/ref/python uv run --project course/harness \
        python course/oracle/ethics.03/reference_card_numbers.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "course/tests")
from _lib.pcg32 import PCG32  # noqa: E402  (the frozen generator)
from tinyllm.eval.bias import bias_report  # noqa: E402
from tinyllm.eval.safety import safety_report  # noqa: E402
from tinyllm.lm.ngram import NGramLM  # noqa: E402
from tinyllm.prob.stats import mean_ci  # noqa: E402

FIX = Path("course/fixtures")
SEED = 0


def main() -> None:
    data = (FIX / "small-corpora/tinyshakespeare.txt").read_bytes()
    cut = int(len(data) * 0.9)
    train, val = list(data[:cut]), list(data[cut:])
    lm = NGramLM(4, vocab_size=256)
    lm.fit([train])
    nll_bits = lm.nll(val[:20000]) / math.log(2)
    bpb, lo, hi = mean_ci(nll_bits)
    print(f"quality | bpb (val, 20000 bytes) | {bpb:.3f} ({lo:.3f}, {hi:.3f})")

    def score(text: str) -> float:
        return -float(lm.nll(list(text.encode("utf-8"))).sum())

    def generate(prompt: str, rng) -> str:
        ctx = list(prompt.encode("utf-8"))[-3:]
        out = []
        for _ in range(80):
            p = np.exp(lm.logprobs(ctx))
            u, c, tok = rng.uniform(), 0.0, 255
            for i, pi in enumerate(p):
                c += pi
                if u < c:
                    tok = i
                    break
            out.append(tok)
            ctx = (ctx + [tok])[-3:]
        return bytes(out).decode("utf-8", errors="replace")

    cfg = json.loads((FIX / "ethics.04/lexicon.json").read_text())
    prompts = [
        json.loads(x)
        for x in (FIX / "ethics.04/prompts.jsonl").read_text().splitlines()
        if x.strip()
    ]
    labelled = [
        json.loads(x)
        for x in (FIX / "ethics.04/labelled.jsonl").read_text().splitlines()
        if x.strip()
    ]
    rep = safety_report(
        "shakespeare-kn4",
        generate,
        prompts,
        cfg["lexicon"],
        cfg["refusal_patterns"],
        PCG32(SEED, 4),
        threshold=cfg["threshold"],
        seed=SEED,
        labelled=[(r["text"], r["label"]) for r in labelled],
    )
    for r in rep["rows"]:
        ci = r.get("ci95")
        print(
            f"safety | {r['task']} (n={r['n']}) | {r['value']:.3f}"
            + (f" ({ci[0]:.3f}, {ci[1]:.3f})" if ci else "")
        )
    b = json.loads((FIX / "ethics.04/bias.json").read_text())
    rep = bias_report(
        "shakespeare-kn4",
        score,
        b["axes"],
        b["stereotype_pairs"],
        PCG32(SEED, 5),
        n_boot=1000,
        seed=SEED,
    )
    for r in rep["rows"]:
        lo, hi = r["ci95"]
        print(
            f"bias | {r['task']} (n={r['n']}) | {r['value']:.3f} ({lo:.3f}, {hi:.3f})"
        )


if __name__ == "__main__":
    main()
