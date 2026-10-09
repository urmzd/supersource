# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Maintainer generator for the M00.1 golden values (units of information).

mpmath at 60 significant digits computes log_b(x), nats -> bits, and bits per
byte; each value is stored as the float64 nearest the exact result, so the
course test compares with the float64 tolerance of tests/_lib/close.py.

    uv run --script course/oracle/M00.1/units_golden.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mpmath

OUT = Path("course/fixtures/M00.1/units.json")
mpmath.mp.dps = 60

XS = [
    "1e-300",
    "1e-10",
    "0.001",
    "0.5",
    "1",
    "2",
    "3",
    "10",
    "1000",
    "536870912",
    "1e300",
]
BASES = ["2", "e", "10", "0.5", "1.5", "256"]
NATS = ["0", "1", "0.6931471805599453", "2.5", "100", "1e-20"]
# (summed nll in nats, bytes): the chapter's "abbacab" bigram, a uniform byte
# model over 1000 bytes, and two made-up model runs.
BPB = [
    ("3*log(2) + 2*log(2.5) + log(3)", 6),
    ("1000*log(256)", 1000),
    ("12345.678", 10000),
    ("1e-3", 7),
]


def f(v) -> float:
    return float(mpmath.mpf(v))


def main() -> None:
    logs = []
    for x in XS:
        for b in BASES:
            bb = mpmath.e if b == "e" else mpmath.mpf(b)
            logs.append(
                {
                    "x": float(x),
                    "b": f(bb),
                    "base_name": b,
                    "log": f(
                        mpmath.log(mpmath.mpf(float(x))) / mpmath.log(mpmath.mpf(f(bb)))
                    ),
                }
            )
    nats = [
        {"nats": float(n), "bits": f(mpmath.mpf(float(n)) / mpmath.log(2))}
        for n in NATS
    ]
    bpb = []
    for expr, n in BPB:
        total = mpmath.mpf(
            eval(expr, {"log": mpmath.log})
        )  # maintainer-only, fixed strings
        bpb.append(
            {
                "expr": expr,
                "nll_nats_sum": f(total),
                "n_bytes": n,
                "bpb": f(mpmath.mpf(f(total)) / (n * mpmath.log(2))),
            }
        )
    doc = {
        "generator": "course/oracle/M00.1/units_golden.py",
        "library": f"mpmath=={mpmath.__version__}",
        "dps": mpmath.mp.dps,
        "log_base": logs,
        "nats_to_bits": nats,
        "bits_per_byte": bpb,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M00.1/units_golden.py",
                f"mpmath=={mpmath.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
