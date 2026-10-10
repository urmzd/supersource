# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the M11.2 golden evaluation numbers.

Each case is a list of batches (float32 NLLs in nats, a mask, a byte count)
drawn from the frozen course PCG32 (course/tests/_lib/pcg32.py) with integer
arithmetic only, so the course test rebuilds the identical batches on any
machine. The expected totals are computed here with mpmath at 50 significant
digits, independently of numpy and of float64 rounding, and stored as
decimal strings in course/fixtures/M11.2/golden.json.

    uv run --script course/oracle/M11.2/ppl_golden.py

Run from the repo root, then update the row in course/fixtures/MANIFEST.tsv
with the sha256 and size the script prints.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mpmath
import numpy as np

sys.path.insert(0, "course/tests")
from _lib.pcg32 import PCG32  # noqa: E402

OUT = Path("course/fixtures/M11.2/golden.json")
mpmath.mp.dps = 50

CASES = {
    # seed, batch sizes, NLL scale, masked fraction
    "three_masked_batches": (11, (2000, 1500, 777), 5.0, 0.2),
    "many_small_batches": (12, (16,) * 400, 0.5, 0.1),
    "high_loss": (13, (300, 300), 40.0, 0.0),
}


def batches(seed: int, sizes, scale: float, masked: float):
    """The case's batches. Must stay identical to batches() in
    course/tests/M11.2/test_ppl.py."""
    rng = PCG32(seed=seed)
    out = []
    for n in sizes:
        x = (rng.uniform_array((n,)) * scale).astype(np.float32)
        keep = rng.uniform_array((n,)) >= masked
        x[~keep] = np.nan  # padding holds garbage
        n_bytes = 3 * int(keep.sum()) + rng.below(50)
        out.append((x, keep, n_bytes))
    return out


def main() -> None:
    doc = {"generator": "course/oracle/M11.2/ppl_golden.py", "digits": 50, "cases": {}}
    for name, (seed, sizes, scale, masked) in CASES.items():
        total = mpmath.mpf(0)
        n_tok = n_bytes = 0
        for x, keep, nb in batches(seed, sizes, scale, masked):
            for v in x[keep].tolist():
                total += mpmath.mpf(v)  # each float32 value is exact in mpf
            n_tok += int(keep.sum())
            n_bytes += nb
        mean = total / n_tok
        doc["cases"][name] = {
            "seed": seed,
            "sizes": list(sizes),
            "scale": scale,
            "masked": masked,
            "n_tokens": n_tok,
            "n_bytes": n_bytes,
            "nll_sum": mpmath.nstr(total, 30),
            "nll_mean": mpmath.nstr(mean, 30),
            "ppl": mpmath.nstr(mpmath.exp(mean), 30),
            "bits_per_token": mpmath.nstr(mean / mpmath.log(2), 30),
            "bpb": mpmath.nstr(total / (n_bytes * mpmath.log(2)), 30),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1) + "\n").encode()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\t"
        f"course/oracle/M11.2/ppl_golden.py\tmpmath=={mpmath.__version__},numpy=={np.__version__}"
    )


if __name__ == "__main__":
    main()
