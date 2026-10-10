# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Maintainer generator for the L2.1 Kneser-Ney golden values.

An independent, by-the-book implementation of interpolated (modified)
Kneser-Ney (Chen and Goodman 1998, section 3; KenLM's start-symbol rule)
in exact rational arithmetic (fractions.Fraction): every count, discount,
and probability is a fraction, written to the fixture as "p/q"; only the
final perplexities go through mpmath logs (50 digits). It shares no code
with the reference: counts are recomputed from the padded sequences by
scanning, continuation counts as sets of left neighbours, and each
probability by recursion on the order.

Cases:
  hand          the chapter's worked example (bigram, d = 1/2, V = 3)
  hand_trigram  the same corpus as a trigram with modified discounts
  ms_p1_*       course/fixtures/MS-P1/corpus.txt as bytes (V = 256), lines
                [:-5] for training and the last 5 for evaluation, n = 1 .. 4,
                modified and d = 3/4

    uv run --script course/oracle/L2.1/kn_golden.py

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

OUT = Path("course/fixtures/L2.1/kn_golden.json")
CORPUS = Path("course/fixtures/MS-P1/corpus.txt")
mpmath.mp.dps = 50
S = "<s>"
FALLBACK = (Fraction(1, 2), Fraction(1), Fraction(3, 2))


class KN:
    def __init__(self, seqs, n, V, d=None):
        self.n, self.V = n, V
        self.padded = [[S] + list(s) for s in seqs if len(s) > 0]
        # raw counts of every k-gram ending at a token
        self.raw = {k: {} for k in range(1, n + 1)}
        for p in self.padded:
            for end in range(1, len(p)):
                for k in range(1, n + 1):
                    start = end - k + 1
                    if start < 0:
                        break
                    g = tuple(p[start : end + 1])
                    self.raw[k][g] = self.raw[k].get(g, 0) + 1
        self.adj = {}
        for k in range(1, n + 1):
            a = {}
            for g, c in self.raw[k].items():
                if k == n or g[0] == S:
                    a[g] = c
                else:
                    left = {big[0] for big in self.raw[k + 1] if big[1:] == g}
                    a[g] = len(left)
            self.adj[k] = a
        self.D = {}
        for k in range(1, n + 1):
            if d is not None:
                self.D[k] = (d, d, d)
                continue
            cnt = [sum(1 for v in self.adj[k].values() if v == j) for j in (1, 2, 3, 4)]
            if 0 in cnt:
                self.D[k] = FALLBACK
                continue
            n1, n2, n3, n4 = (Fraction(x) for x in cnt)
            Y = n1 / (n1 + 2 * n2)
            D = (1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3)
            ok = all(0 < D[j] <= j + 1 for j in range(3))
            self.D[k] = D if ok else FALLBACK

    def disc(self, k, a):
        if a == 0:
            return Fraction(0)
        return self.D[k][min(a, 3) - 1]

    def p(self, k, h, w):
        """p_k(w | h), h a tuple of k - 1 symbols."""
        if k == 0:
            return Fraction(1, self.V)
        lower = self.p(k - 1, h[1:], w)
        row = {g[-1]: a for g, a in self.adj[k].items() if g[:-1] == h}
        if not row:
            return lower
        T = sum(row.values())
        a = row.get(w, 0)
        num = sum(self.disc(k, x) for x in row.values())
        return max(a - self.disc(k, a), 0) / Fraction(T) + num / T * lower

    def prob(self, context, w):
        ctx = tuple(context)
        if len(ctx) < self.n - 1:
            hist = (S,) + ctx
        else:
            hist = ctx[len(ctx) - (self.n - 1) :] if self.n > 1 else ()
        K = min(self.n, len(hist) + 1)
        h = hist[len(hist) - (K - 1) :] if K > 1 else ()
        return self.p(K, h, w)


def frac(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def ppl(model: KN, seq) -> tuple[str, str]:
    total = mpmath.mpf(0)
    for t in range(len(seq)):
        q = model.prob(seq[:t], seq[t])
        total += -mpmath.log(mpmath.mpf(q.numerator) / q.denominator)
    return mpmath.nstr(total, 30), mpmath.nstr(mpmath.exp(total / len(seq)), 30)


def main() -> int:
    a, b, c = 0, 1, 2
    hand = [[a, b, c], [a, b, a, b], [c, a, b, c]]
    cases = {}

    m = KN(hand, 2, 3, d=Fraction(1, 2))
    contexts = [[], [a], [b], [c], [a, b]]
    cases["hand"] = {
        "sequences": hand,
        "n": 2,
        "discount": "0.5",
        "vocab_size": 3,
        "discounts": {str(k): [frac(x) for x in m.D[k]] for k in m.D},
        "adjusted": {
            str(k): {
                " ".join(map(str, g)): v for g, v in sorted(m.adj[k].items(), key=str)
            }
            for k in m.adj
        },
        "probs": {
            " ".join(map(str, ctx)) or "<s>": [frac(m.prob(ctx, w)) for w in range(3)]
            for ctx in contexts
        },
    }
    m3 = KN(hand, 3, 3)
    contexts3 = [[], [a], [b], [a, b], [b, a], [c, a], [b, b]]
    cases["hand_trigram"] = {
        "sequences": hand,
        "n": 3,
        "discount": "modified",
        "vocab_size": 3,
        "discounts": {str(k): [frac(x) for x in m3.D[k]] for k in m3.D},
        "probs": {
            " ".join(map(str, ctx)) or "<s>": [frac(m3.prob(ctx, w)) for w in range(3)]
            for ctx in contexts3
        },
    }

    lines = [list(l) for l in CORPUS.read_bytes().split(b"\n") if l]
    train, held = lines[:-5], lines[-5:]
    for n in (1, 2, 3, 4):
        for mode, d in (("modified", None), ("0.75", Fraction(3, 4))):
            km = KN(train, n, 256, d=d)
            rows = [ppl(km, s) for s in held]
            probe_ctx = list(b"the ")
            cases[f"ms_p1_n{n}_{mode}"] = {
                "n": n,
                "discount": mode,
                "vocab_size": 256,
                "train_lines": [0, len(lines) - 5],
                "eval_lines": [len(lines) - 5, len(lines)],
                "discounts": {str(k): [frac(x) for x in km.D[k]] for k in km.D},
                "nll_sum": [r[0] for r in rows],
                "ppl": [r[1] for r in rows],
                "probe_context": probe_ctx,
                "probe_probs": {
                    str(w): frac(km.prob(probe_ctx, w))
                    for w in (32, 97, 99, 104, 115, 116, 0, 255)
                },
            }
            print(n, mode, [r[1][:8] for r in rows])

    data = {
        "__meta__": {
            "generator": "course/oracle/L2.1/kn_golden.py",
            "mpmath": mpmath.__version__,
            "corpus": str(CORPUS),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "start_symbol": "<s> written as -1 in model files; contexts here list token ids only",
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
