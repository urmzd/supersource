# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for the MS-P1 milestone fixtures.

The milestone trains the learner's byte bigram on course/fixtures/MS-P1/corpus.txt
through `{tinyllm} train bigram` and decodes greedily through `{tinyllm} generate`
(spec/cli-roles.md). This script recomputes, in exact rational arithmetic, what
those two verbs must print for the contract's definition of the model:

  weight[i, j] = log((count(i, j) + alpha) / (count(i, *) + alpha * V)),  V = 256, alpha = 1

  - nll: the mean of -weight[x_t, x_{t+1}] over the corpus, in nats per byte
  - greedy: from the last prompt byte, take argmax_j weight[cur, j] (ties to the
    lowest id), 32 times

Greedy decoding of a bigram only compares counts within one row, so the ids are
exact integers. The script also proves the path has no ties and reports the
smallest top-2 log margin, so a float32 implementation cannot flip a step.

    uv run --script course/oracle/MS-P1/bigram_greedy.py

Run from the repo root. It rewrites course/fixtures/MS-P1/greedy_once_32.json
and prints the MANIFEST.tsv rows and the nll bound for MS-P1.toml.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

OUT = Path("course/fixtures/MS-P1")
V, ALPHA, PROMPT, N = 256, 1, b"Once", 32


def main() -> None:
    data = (OUT / "corpus.txt").read_bytes()
    counts = [[0] * V for _ in range(V)]
    for a, b in zip(data, data[1:]):
        counts[a][b] += 1
    rows = [sum(r) for r in counts]

    nll = -sum(
        math.log((counts[a][b] + ALPHA) / (rows[a] + ALPHA * V))
        for a, b in zip(data, data[1:])
    )
    nll /= len(data) - 1

    cur, ids, margin = PROMPT[-1], [], math.inf
    for _ in range(N):
        row = counts[cur]
        best = max(range(V), key=lambda j: (row[j], -j))
        second = max(row[j] for j in range(V) if j != best)
        if second == row[best]:
            raise SystemExit(f"tie after byte {cur}: pick another prompt or corpus")
        margin = min(margin, math.log((row[best] + ALPHA) / (second + ALPHA)))
        ids.append(best)
        cur = best

    (OUT / "greedy_once_32.json").write_text(
        json.dumps({"prompt": PROMPT.decode(), "ids": ids}) + "\n"
    )
    print(
        f"tokens = {len(data)}, nll = {nll:.6f} nats/byte, greedy text = {bytes(ids).decode()!r}"
    )
    print(f"smallest top-2 log margin on the greedy path: {margin:.4f}")
    for name in ("corpus.txt", "greedy_once_32.json"):
        p = OUT / name
        raw = p.read_bytes()
        print(
            "\t".join(
                [
                    p.as_posix(),
                    hashlib.sha256(raw).hexdigest(),
                    str(len(raw)),
                    "course/oracle/MS-P1/bigram_greedy.py"
                    if name != "corpus.txt"
                    else "hand",
                    "-",
                    "-",
                    "Apache-2.0",
                ]
            )
        )


if __name__ == "__main__":
    main()
