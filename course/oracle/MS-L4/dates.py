# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6"]
# ///
"""Maintainer generator for the MS-L4 dates task (original synthetic data).

Writes two UTF-8 files, one example per line, `source<TAB>target`:

  course/fixtures/small-corpora/dates.tsv        8000 training pairs
  course/fixtures/small-corpora/dates-test.txt    600 held-out pairs

The target is always the ISO date YYYY-MM-DD. The source is the same date in
one of ten human formats, short ("3.5.2021", "2021/05/03") or long
("Monday, May 3, 2021", "the 3rd of May 2021"); the weekday, when present,
is the true one. Dates are uniform over 1950-01-01 .. 2049-12-31. Every
test source is absent from the training file.

A source of at least 20 characters is in the "long" bucket the `translate`
verb reports separately (course/milestones/MS-L4.toml): there the encoder's
final state must carry the whole string, which is where attention matters.

    uv run --python 3.12 --script course/oracle/MS-L4/dates.py

Run from the repo root, then update the MANIFEST.tsv rows it prints.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

SEED = 20261009
OUT = Path("course/fixtures/small-corpora")
N_TRAIN, N_TEST = 8000, 600
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
START = dt.date(1950, 1, 1)
SPAN = (dt.date(2049, 12, 31) - START).days + 1


def ordinal(d: int) -> str:
    suffix = (
        "th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    )
    return f"{d}{suffix}"


def render(day: dt.date, fmt: int) -> str:
    d, m, y = day.day, day.month, day.year
    mon, wd = MONTHS[m - 1], DAYS[day.weekday()]
    return [
        f"{d} {mon} {y}",  # 3 May 2021
        f"{mon} {d}, {y}",  # May 3, 2021
        f"{d}.{m}.{y}",  # 3.5.2021
        f"{y}/{m:02d}/{d:02d}",  # 2021/05/03
        f"{wd}, {mon} {d}, {y}",  # Monday, May 3, 2021
        f"the {ordinal(d)} of {mon} {y}",  # the 3rd of May 2021
        f"{wd[:3]} {d} {mon[:3]} {y}",  # Mon 3 May 2021
        f"{mon.upper()} {d} {y}",  # MAY 3 2021
        f"{d:02d}-{mon[:3]}-{y}",  # 03-May-2021
        f"{wd} the {ordinal(d)} of {mon}, {y}",  # Monday the 3rd of May, 2021
    ][fmt]


def main() -> None:
    rng = np.random.default_rng(SEED)
    seen: set[str] = set()
    rows: list[str] = []
    while len(rows) < N_TRAIN + N_TEST:
        day = START + dt.timedelta(days=int(rng.integers(0, SPAN)))
        src = render(day, int(rng.integers(0, 10)))
        if src in seen:
            continue
        seen.add(src)
        rows.append(f"{src}\t{day.isoformat()}")
    train, test = rows[:N_TRAIN], rows[N_TRAIN:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lines in (("dates.tsv", train), ("dates-test.txt", test)):
        data = ("\n".join(lines) + "\n").encode()
        (OUT / name).write_bytes(data)
        print(
            "\t".join(
                [
                    str(OUT / name),
                    hashlib.sha256(data).hexdigest(),
                    str(len(data)),
                    "course/oracle/MS-L4/dates.py",
                    f"numpy=={np.__version__}",
                    "-",
                    "Apache-2.0",
                ]
            )
        )
    long = sum(len(r.split("\t")[0]) >= 20 for r in test)
    print(f"# test: {len(test)} pairs, {long} in the long bucket (source >= 20 chars)")


if __name__ == "__main__":
    main()
