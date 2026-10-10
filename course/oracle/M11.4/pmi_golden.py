# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Maintainer generator for the M11.4 golden PMI, PPMI, and mutual information.

The tables are small integer (or quarter-integer) co-occurrence counts, drawn
here from the frozen course PCG32 (course/tests/_lib/pcg32.py) and stored in
the fixture verbatim, so the course test needs no generator. Every expected
value is computed from the definitions with mpmath at 50 significant digits,
term by term with Python loops (no numpy), independently of the reference's
vectorized float64 code, and stored as a decimal string ("-inf" for an
unseen pair).

    uv run --script course/oracle/M11.4/pmi_golden.py

Run from the repo root, then update the row in course/fixtures/MANIFEST.tsv
with the sha256 and size the script prints.
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath

sys.path.insert(0, "course/tests")
from _lib.pcg32 import PCG32  # noqa: E402

OUT = Path("course/fixtures/M11.4/pmi_golden.json")
mpmath.mp.dps = 50


def draw(seed: int, rows: int, cols: int, top: int, quarters: bool = False):
    g = PCG32(seed=seed)
    t = [[g.below(top) for _ in range(cols)] for _ in range(rows)]
    if quarters:
        return [[Fraction(v, 4) for v in r] for r in t]
    return [[Fraction(v) for v in r] for r in t]


def mp(q: Fraction):
    return mpmath.mpf(q.numerator) / q.denominator


def pmi(table, alpha):
    R, C = len(table), len(table[0])
    D = sum(sum(r) for r in table)
    nw = [sum(r) for r in table]
    nc = [sum(table[i][j] for i in range(R)) for j in range(C)]
    a = mpmath.mpf(alpha)
    nca = [mp(x) for x in nc]
    nca = [x**a if x > 0 else mpmath.mpf(0) for x in nca]
    z = mpmath.fsum(nca)
    out = []
    for i in range(R):
        row = []
        for j in range(C):
            if table[i][j] == 0:
                row.append("-inf")
                continue
            pwc = mp(table[i][j] / D)
            pw = mp(nw[i] / D)
            pc = nca[j] / z
            row.append(
                mpmath.nstr(mpmath.log(pwc) - mpmath.log(pw) - mpmath.log(pc), 30)
            )
        out.append(row)
    return out


def mutual_information(table):
    R, C = len(table), len(table[0])
    D = sum(sum(r) for r in table)
    nw = [sum(r) for r in table]
    nc = [sum(table[i][j] for i in range(R)) for j in range(C)]
    terms = []
    for i in range(R):
        for j in range(C):
            if table[i][j] == 0:
                continue
            ratio = (table[i][j] * D) / (nw[i] * nc[j])  # exact rational
            terms.append(mp(table[i][j] / D) * mpmath.log(mp(ratio)))
    return mpmath.nstr(mpmath.fsum(terms), 30)


def main() -> int:
    tables = {
        "hand": [[Fraction(v) for v in r] for r in [[2, 0, 2], [2, 4, 0]]],
        "random_6x5": draw(41, 6, 5, 6),
        "quarters_4x7": draw(42, 4, 7, 9, quarters=True),
        "independent_3x4": [[Fraction(a * b) for b in (1, 2, 3, 6)] for a in (1, 4, 5)],
        "one_row_1x5": draw(43, 1, 5, 5),
    }
    cases = {}
    for name, t in tables.items():
        assert sum(sum(r) for r in t) > 0, name
        cases[name] = {
            "table": [[float(v) for v in r] for r in t],
            "pmi_alpha_1": pmi(t, 1),
            "pmi_alpha_0.75": pmi(t, mpmath.mpf(3) / 4),
            "pmi_alpha_0.5": pmi(t, mpmath.mpf(1) / 2),
            "mutual_information": mutual_information(t),
        }
    data = {
        "__meta__": {
            "generator": "course/oracle/M11.4/pmi_golden.py",
            "mpmath": mpmath.__version__,
            "dps": 50,
            "seeds": [41, 42, 43],
            "units": "nats",
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = (json.dumps(data, indent=1) + "\n").encode()
    OUT.write_bytes(blob)
    print(f"{OUT}\t{hashlib.sha256(blob).hexdigest()}\t{len(blob)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
