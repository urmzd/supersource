# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy==1.17.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M07.5 golden fixture (scipy).

Records, as JSON with float64 values written by repr:
  exact_paired     exact two-sided sign-flip p-values: scipy.stats.permutation_test
                   (permutation_type="samples", n_resamples=inf) of |sum(x - y)|
                   with alternative="greater", cross-checked against a plain
                   enumeration of all 2^n sign vectors
  exact_two        exact two-sample p-values: scipy.stats.permutation_test
                   (permutation_type="independent", n_resamples=inf) of
                   |mean(x) - mean(y)|, alternative="greater", cross-checked
                   against itertools.combinations over the pooled sample
  replay_paired,   the Monte Carlo p-values the contract defines for given
  replay_two       seeds, replayed here from the frozen PCG32
                   (course/tests/_lib/pcg32.py) by a separate transcription
                   of the draw order, so the course tests pin it bit for bit
  mcnemar          scipy.stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue
                   (exact) and scipy.stats.chi2.sf(max(|b01 - b10| - 1, 0)^2 / n, 1)
  holm             adjusted p-values by numpy (sorted, (m - k) p, running max,
                   clipped at 1) and rejections at three alphas

    uv run --offline --python 3.11 --script course/oracle/M07.5/scipy_golden.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

OUT = Path("course/fixtures/M07.5/scipy_golden.json")
spec = importlib.util.spec_from_file_location("pcg", "course/tests/_lib/pcg32.py")
pcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcg)


def f(x) -> float:
    return float(x)


def exact_paired_enum(d: np.ndarray) -> float:
    t = abs(d.sum())
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=d.size):
        if abs(float(np.dot(signs, d))) >= t * (1 - 1e-9):
            hits += 1
    return hits / 2**d.size


def exact_two_enum(x: np.ndarray, y: np.ndarray) -> float:
    z = np.concatenate([x, y])
    t = abs(x.mean() - y.mean())
    hits = total = 0
    for comb in itertools.combinations(range(z.size), x.size):
        mask = np.zeros(z.size, dtype=bool)
        mask[list(comb)] = True
        total += 1
        if abs(z[mask].mean() - z[~mask].mean()) >= t * (1 - 1e-9):
            hits += 1
    return hits / total


def replay_paired(a, b, n_perm, seed) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    g = pcg.PCG32(seed=seed)
    t = abs(float(np.sum(d)))
    c = 0
    for _ in range(n_perm):
        s = np.ones(d.size)
        for i in range(d.size):
            if g.uniform() < 0.5:
                s[i] = -1.0
        c += abs(float(np.sum(s * d))) >= t * (1 - 1e-9)
    return (1 + c) / (1 + n_perm)


def replay_two(a, b, n_perm, seed) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    z = np.concatenate([x, y])
    g = pcg.PCG32(seed=seed)
    t = abs(x.mean() - y.mean())
    c = 0
    for _ in range(n_perm):
        idx = np.arange(z.size)
        for i in range(z.size - 1, 0, -1):
            j = min(int(g.uniform() * (i + 1)), i)
            idx[[i, j]] = idx[[j, i]]
        p = z[idx]
        c += abs(float(np.mean(p[: x.size]) - np.mean(p[x.size :]))) >= t * (1 - 1e-9)
    return (1 + c) / (1 + n_perm)


def main() -> None:
    out: dict = {
        "__meta__": {
            "generator": "course/oracle/M07.5/scipy_golden.py",
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        }
    }
    g = pcg.PCG32(seed=505)

    def draws(n, shift=0.0, scale=1.0):
        return [round(scale * g.normal() + shift, 6) for _ in range(n)]

    paired = []
    for name, n, shift in [
        ("null10", 10, 0.0),
        ("shift10", 10, 0.6),
        ("shift12", 12, 0.9),
        ("tiny4", 4, 1.0),
    ]:
        a = draws(n)
        b = [round(v - shift + 0.3 * g.normal(), 6) for v in a]
        x, y = np.asarray(a), np.asarray(b)
        res = stats.permutation_test(
            (x, y),
            lambda u, v, axis=-1: np.abs(np.sum(u - v, axis=axis)),
            permutation_type="samples",
            n_resamples=np.inf,
            alternative="greater",
            vectorized=True,
        )
        enum = exact_paired_enum(x - y)
        assert abs(res.pvalue - enum) < 1e-12, (name, res.pvalue, enum)
        paired.append([name, a, b, f(res.pvalue)])
    out["exact_paired"] = paired

    two = []
    for name, na, nb, shift in [
        ("null6x6", 6, 6, 0.0),
        ("shift7x5", 7, 5, 1.2),
        ("shift8x8", 8, 8, 0.8),
    ]:
        a, b = draws(na, shift), draws(nb)
        x, y = np.asarray(a), np.asarray(b)
        res = stats.permutation_test(
            (x, y),
            lambda u, v, axis=-1: np.abs(np.mean(u, axis=axis) - np.mean(v, axis=axis)),
            permutation_type="independent",
            n_resamples=np.inf,
            alternative="greater",
            vectorized=True,
        )
        enum = exact_two_enum(x, y)
        assert abs(res.pvalue - enum) < 1e-12, (name, res.pvalue, enum)
        two.append([name, a, b, f(res.pvalue)])
    out["exact_two"] = two

    out["replay_paired"] = [
        [name, n_perm, seed, f(replay_paired(a, b, n_perm, seed))]
        for (name, a, b, _), n_perm, seed in [
            (paired[1], 500, 3),
            (paired[3], 64, 0),
            (paired[2], 300, 11),
        ]
    ]
    out["replay_two"] = [
        [name, n_perm, seed, f(replay_two(a, b, n_perm, seed))]
        for (name, a, b, _), n_perm, seed in [(two[1], 400, 5), (two[0], 200, 0)]
    ]

    mc = []
    for b01, b10 in [
        (1, 7),
        (0, 0),
        (5, 5),
        (0, 9),
        (12, 30),
        (40, 61),
        (3, 4),
        (150, 210),
        (0, 1),
    ]:
        n = b01 + b10
        exact = 1.0 if n == 0 else stats.binomtest(min(b01, b10), n, 0.5).pvalue
        asym = 1.0 if n == 0 else stats.chi2.sf(max(abs(b01 - b10) - 1, 0) ** 2 / n, 1)
        mc.append([b01, b10, f(exact), f(asym)])
    out["mcnemar"] = mc

    holm = []
    for p in [
        [0.01, 0.04, 0.02, 0.005],
        [0.2, 0.001, 0.03, 0.03, 0.04, 0.5],
        [0.0, 1.0, 0.049, 0.012],
        [0.3],
        [0.01, 0.01, 0.01],
    ]:
        arr = np.asarray(p)
        m = arr.size
        order = np.argsort(arr, kind="stable")
        scaled = np.minimum(1.0, (m - np.arange(m)) * arr[order])
        adj_sorted = np.maximum.accumulate(scaled)
        adj = np.empty(m)
        adj[order] = adj_sorted
        holm.append(
            [
                p,
                adj.tolist(),
                {str(al): (adj <= al).tolist() for al in (0.01, 0.05, 0.1)},
            ]
        )
    out["holm"] = holm

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1) + "\n"
    OUT.write_text(text)
    b = text.encode()
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/M07.5/scipy_golden.py\t"
        f"scipy=={scipy.__version__} numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
