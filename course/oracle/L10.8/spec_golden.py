"""Maintainer generator for course/fixtures/L10.8/spec_golden.json.

The Rust port (L10.8) is held to L8.6, the Python specification, on shared
inputs: this script runs the reference Python (course/ref/python) on seeded
cases and records its answers.

* `verify`: target logits (exact f32 values, so f32 -> f64 widening is the
  same on both sides), a draft, optional draft distributions, sampling
  params, prompt, output history, and a seed. Recorded: the emitted tokens
  and n_accepted of tinyllm.infer.spec.verify_draft with
  request_rng(seed), and the generator's next u32 afterwards (so the number
  of draws matches too).
* `prompt_lookup`: contexts over a 3-symbol alphabet and
  PromptLookupDraft(max_ngram, min_ngram).propose(ctx, k).

    uv run --project course/harness python course/oracle/L10.8/spec_golden.py
Run from the repo root; it prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "course" / "ref" / "python"))

from tinyllm.infer.sample import SamplingParams, request_rng  # noqa: E402
from tinyllm.infer.spec import PromptLookupDraft, verify_draft  # noqa: E402

OUT = ROOT / "course" / "fixtures" / "L10.8" / "spec_golden.json"

PARAMS = [
    {"temperature": 0.0},
    {"temperature": 1.0},
    {"temperature": 0.7, "top_k": 3},
    {"temperature": 1.3, "top_p": 0.9},
    {"temperature": 1.0, "min_p": 0.1},
    {"temperature": 0.0, "repetition_penalty": 1.3},
    {
        "temperature": 0.9,
        "repetition_penalty": 1.2,
        "presence_penalty": 0.4,
        "frequency_penalty": 0.3,
    },
]


def verify_cases(rs: np.random.Generator) -> list[dict]:
    cases = []
    for ci in range(120):
        V = int(rs.integers(4, 9))
        m = int(rs.integers(0, 5))
        logits = (rs.normal(0, 2, (m + 1, V))).astype(np.float32)
        params = dict(PARAMS[ci % len(PARAMS)])
        # Drafts that agree with the target's argmax for a while, then not.
        agree = int(rs.integers(0, m + 1))
        draft = []
        for i in range(m):
            g = int(np.argmax(logits[i]))
            draft.append(g if i < agree else int(rs.integers(0, V)))
        probs = None
        if params["temperature"] > 0 and ci % 3 == 0 and m > 0:
            q = rs.dirichlet(np.ones(V), size=m)
            for i, x in enumerate(draft):
                q[i, x] += 0.05  # q[x] > 0: the draft proposed x
                q[i] /= q[i].sum()
            probs = q
        prompt = [int(x) for x in rs.integers(0, V, int(rs.integers(1, 4)))]
        history = [int(x) for x in rs.integers(0, V, int(rs.integers(0, 4)))]
        seed = int(rs.integers(0, 2**63))
        p = SamplingParams(**params)
        rng = request_rng(seed)
        emitted, n = verify_draft(
            logits.astype(np.float64), draft, probs, p, history, rng, prompt
        )
        cases.append(
            {
                "name": f"verify-{ci}",
                "logits": [[float(x) for x in row] for row in logits],
                "draft": draft,
                "draft_probs": None
                if probs is None
                else [[float(x) for x in row] for row in probs],
                "params": params,
                "prompt": prompt,
                "output": history,
                "seed": seed,
                "emitted": [int(t) for t in emitted],
                "n_accepted": int(n),
                "next_u32": int(rng.next_u32()),
            }
        )
    return cases


def lookup_cases(rs: np.random.Generator) -> list[dict]:
    cases = []
    for ci in range(150):
        L = int(rs.integers(1, 21))
        ctx = [int(x) for x in rs.integers(0, 3, L)]
        k = int(rs.integers(0, 6))
        mx, mn = [(3, 1), (2, 2), (4, 2), (1, 1)][ci % 4]
        got, _ = PromptLookupDraft(mx, mn).propose(ctx, k, None)
        cases.append(
            {
                "ctx": ctx,
                "k": k,
                "max_ngram": mx,
                "min_ngram": mn,
                "draft": [int(t) for t in got],
            }
        )
    return cases


def main() -> None:
    rs = np.random.default_rng(20261009)
    doc = {
        "generator": "course/oracle/L10.8/spec_golden.py (reference Python L8.6 verify_draft and PromptLookupDraft)",
        "verify": verify_cases(rs),
        "prompt_lookup": lookup_cases(rs),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    rel = OUT.relative_to(ROOT).as_posix()
    print(
        f"{rel}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L10.8/spec_golden.py\tnumpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
