# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Maintainer oracle for the MS-L2 KN-4 perplexity bar.

MS-L2 trains `{tinyllm} lm train ngram --n 4` on course/fixtures/MS-L2/train.bin
(one byte stream, so one <s> before its first byte) and scores
course/fixtures/MS-L2/val.bin as one sequence. The milestone accepts a
perplexity within 0.5% of the value this script prints.

This is an independent, indexed implementation of interpolated modified
Kneser-Ney as contracts/py/tinyllm/lm/ngram.pyi defines it (the same
definition course/oracle/L2.1/kn_golden.py checks on small corpora; that
script scans every n-gram per query and is too slow for 140 KB). Counts,
discounts, and every probability are exact fractions (fractions.Fraction);
the total negative log-likelihood is summed in mpmath at 50 digits.

    uv run --script course/oracle/MS-L2/kn4_ref.py

Run from the repo root. It prints ppl, nll_mean, and the two bounds that go
into course/milestones/MS-L2.toml.
"""

from __future__ import annotations

import struct
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import mpmath

mpmath.mp.dps = 50
BOS = -1
N, V = 4, 256
FALLBACK = (Fraction(1, 2), Fraction(1), Fraction(3, 2))


def read_bin(path: Path) -> list[int]:
    raw = path.read_bytes()
    magic, version, n, _vocab = struct.unpack("<4i", raw[:16])
    assert magic == 20240520 and version == 1 and len(raw) == 1024 + 2 * n
    return list(struct.unpack(f"<{n}H", raw[1024:]))


def build(seq: list[int]):
    padded = [BOS] + seq
    raw = {k: defaultdict(int) for k in range(1, N + 1)}
    for end in range(1, len(padded)):
        for k in range(1, min(N, end + 1) + 1):
            raw[k][tuple(padded[end - k + 1 : end + 1])] += 1
    adj = {N: dict(raw[N])}
    for k in range(N - 1, 0, -1):
        left = defaultdict(set)
        for g in raw[k + 1]:
            left[g[1:]].add(g[0])
        adj[k] = {g: (c if g[0] == BOS else len(left[g])) for g, c in raw[k].items()}
    D = {}
    for k in range(1, N + 1):
        cnt = [sum(1 for a in adj[k].values() if a == j) for j in (1, 2, 3, 4)]
        if 0 in cnt:
            D[k] = FALLBACK
            continue
        n1, n2, n3, n4 = (Fraction(x) for x in cnt)
        Y = n1 / (n1 + 2 * n2)
        d = (1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3)
        D[k] = d if all(0 < d[j] <= j + 1 for j in range(3)) else FALLBACK
    rows = {k: defaultdict(dict) for k in range(1, N + 1)}
    for k in range(1, N + 1):
        for g, a in adj[k].items():
            rows[k][g[:-1]][g[-1]] = a
    stats = {}
    for k in range(1, N + 1):
        for h, row in rows[k].items():
            T = sum(row.values())
            removed = sum(D[k][min(a, 3) - 1] for a in row.values())
            stats[(k, h)] = (T, removed / T)
    return rows, D, stats


def prob(model, context: list[int], w: int) -> Fraction:
    rows, D, stats = model
    hist = (
        tuple([BOS] + context)
        if len(context) < N - 1
        else tuple(context[len(context) - (N - 1) :])
    )
    K = min(N, len(hist) + 1)
    p = Fraction(1, V)
    for k in range(1, K + 1):
        h = hist[len(hist) - (k - 1) :] if k > 1 else ()
        if (k, h) not in stats:
            continue  # unseen history: p_k = p_{k-1}
        T, gamma = stats[(k, h)]
        a = rows[k][h].get(w, 0)
        disc = D[k][min(a, 3) - 1] if a else 0
        p = max(a - disc, 0) / Fraction(T) + gamma * p
    return p


def main() -> None:
    train = read_bin(Path("course/fixtures/MS-L2/train.bin"))
    val = read_bin(Path("course/fixtures/MS-L2/val.bin"))
    model = build(train)
    total = mpmath.mpf(0)
    for t in range(len(val)):
        q = prob(model, val[max(0, t - (N - 1)) : t] if t >= N - 1 else val[:t], val[t])
        total += -mpmath.log(mpmath.mpf(q.numerator) / q.denominator)
    mean = total / len(val)
    ppl = mpmath.exp(mean)
    print(f"tokens {len(val)}")
    print(f"discounts {[[float(x) for x in model[1][k]] for k in range(1, N + 1)]}")
    print(f"ppl      {mpmath.nstr(ppl, 20)}")
    print(f"nll_mean {mpmath.nstr(mean, 20)}")
    print(f"bound ppl <= {mpmath.nstr(ppl * mpmath.mpf('1.005'), 12)}")
    print(f"bound nll_mean >= {mpmath.nstr(mpmath.log(ppl * mpmath.mpf('0.995')), 12)}")


if __name__ == "__main__":
    main()
