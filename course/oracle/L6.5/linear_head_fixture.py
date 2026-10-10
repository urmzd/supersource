"""Maintainer generator for course/fixtures/L6.5/linear_head.json: the shared
fixture of the D33 policy head. Python (L6.5) and Go (gw.08) must score it
the same within 1e-6.

It fits a head with the course reference (course/ref/python, L6.5's
fit_linear_head over M07.7's IRLS) on 48 synthetic 8-dimensional
"embeddings" in two clusters, exports it with export_linear_head, and
records the reference's head_probs on 12 probe embeddings, among them a zero
vector (normalization must leave it zero), a vector of norm 1e6, and one
with a negative zero. No third-party package is involved.

    PYTHONPATH=course/ref/python course/harness/.venv/bin/python course/oracle/L6.5/linear_head_fixture.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from tinyllm.obj.heads import (
    export_linear_head,
    fit_linear_head,
    head_metrics,
    head_probs,
)

OUT = Path("course/fixtures/L6.5/linear_head.json")
SEED = 20261012


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = 8
    mu = rng.normal(size=d)
    X = np.concatenate([rng.normal(size=(24, d)) + mu, rng.normal(size=(24, d)) - mu])
    y = np.array([1] * 24 + [0] * 24)
    head = fit_linear_head(X, y, ["safe", "unsafe"], l2=1.0, threshold=0.9)
    probes = rng.normal(size=(12, d))
    probes[0] = 0.0
    probes[1] *= 1e6
    probes[2, 3] = -0.0
    with tempfile.TemporaryDirectory() as t:
        export_linear_head(
            head, "smol-135m", f"{t}/head.json", metrics=head_metrics(head, X, y)
        )
        doc = json.loads(Path(f"{t}/head.json").read_text())
    probs = head_probs(doc, probes)
    out = {
        "generator": "course/oracle/L6.5/linear_head_fixture.py",
        "seed": SEED,
        "head": doc,
        "embeddings": [[float(v) for v in row] for row in probes],
        "probs": [[float(v) for v in row] for row in probs],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    data = OUT.read_bytes()
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L6.5/linear_head_fixture.py\t"
        f"numpy=={np.__version__}\t-\tApache-2.0"
    )


main()
