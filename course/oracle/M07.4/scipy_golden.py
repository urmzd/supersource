# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy==1.17.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M07.4 golden fixture (scipy).

Records, as JSON with float64 values written by repr:
  normal_cdf, normal_ppf      scipy.stats.norm.cdf / .ppf
  t_cdf, t_ppf                scipy.stats.t.cdf / .ppf over a grid of df
  standard_error, mean_ci     scipy.stats.sem and scipy.stats.t.interval
  wilson                      scipy.stats.binomtest(k, n).proportion_ci(method="wilson")
  quantile                    numpy.quantile(method="linear") (type 7)
  bootstrap                   an independent percentile bootstrap: indices from
                              the frozen PCG32 (course/tests/_lib/pcg32.py), one
                              uniform per index, statistics and quantiles by numpy

    uv run --offline --python 3.12 --script course/oracle/M07.4/scipy_golden.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

OUT = Path("course/fixtures/M07.4/scipy_golden.json")

spec = importlib.util.spec_from_file_location("pcg32", "course/tests/_lib/pcg32.py")
pcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcg)

data_rng = np.random.default_rng(20261012)


def f(x) -> float:
    return float(x)


def main() -> None:
    out: dict = {
        "__meta__": {
            "generator": "course/oracle/M07.4/scipy_golden.py",
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "seed": 20261012,
        }
    }
    zs = [
        -8.0,
        -5.0,
        -3.0,
        -1.959963984540054,
        -1.0,
        -0.25,
        0.0,
        0.5,
        1.0,
        1.6448536269514722,
        2.0,
        3.5,
        6.0,
    ]
    out["normal_cdf"] = [[z, f(stats.norm.cdf(z))] for z in zs]
    ps = [
        1e-12,
        1e-6,
        0.001,
        0.025,
        0.05,
        0.1,
        0.3,
        0.4999,
        0.5,
        0.5001,
        0.7,
        0.9,
        0.95,
        0.975,
        0.995,
        1 - 1e-6,
        1 - 1e-12,
    ]
    out["normal_ppf"] = [[p, f(stats.norm.ppf(p))] for p in ps]
    dfs = [1, 2, 3, 4, 5, 6, 9, 10, 19, 29, 30, 99, 1000]
    ts = [-4.0, -2.0, -0.5, 0.0, 0.3, 1.0, 2.0, 2.5, 6.0, 40.0]
    out["t_cdf"] = [[t, d, f(stats.t.cdf(t, d))] for d in dfs for t in ts]
    tps = [0.0005, 0.025, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95, 0.975, 0.995, 0.9995]
    out["t_ppf"] = [[p, d, f(stats.t.ppf(p, d))] for d in dfs for p in tps]

    samples = {
        "hand": [2.0, 4.0, 4.0, 5.0, 7.0, 8.0],
        "normal20": data_rng.normal(3.0, 2.0, 20).tolist(),
        "expon50": data_rng.exponential(1.5, 50).tolist(),
        "tiny2": [1.0, 2.0],
        "uniform200": data_rng.uniform(-1.0, 1.0, 200).tolist(),
    }
    out["samples"] = samples
    out["standard_error"] = {k: f(stats.sem(v)) for k, v in samples.items()}
    ci = []
    for name, v in samples.items():
        for alpha in (0.05, 0.1, 0.01):
            a = np.asarray(v)
            lo, hi = stats.t.interval(
                1 - alpha, len(a) - 1, loc=a.mean(), scale=stats.sem(a)
            )
            ci.append([name, alpha, f(a.mean()), f(lo), f(hi)])
    out["mean_ci"] = ci

    wil = []
    for k, n in [
        (0, 10),
        (1, 10),
        (5, 10),
        (10, 10),
        (3, 7),
        (37, 120),
        (0, 1),
        (1, 1),
        (999, 1000),
    ]:
        for alpha in (0.05, 0.1, 0.01):
            r = stats.binomtest(k, n).proportion_ci(
                confidence_level=1 - alpha, method="wilson"
            )
            wil.append([k, n, alpha, f(r.low), f(r.high)])
    out["wilson"] = wil

    qs = [0.0, 0.025, 0.1, 0.25, 0.5, 0.77, 0.975, 1.0]
    qarr = {
        "hand": [3.0, 1.0, 4.0, 1.0, 5.0],
        "one": [7.5],
        "normal20": samples["normal20"],
    }
    out["quantile"] = [
        [name, q, f(np.quantile(np.asarray(v), q, method="linear"))]
        for name, v in qarr.items()
        for q in qs
    ]
    out["quantile_samples"] = qarr

    boot = []
    for name, statname, n_boot, alpha, seed in [
        ("hand", "mean", 200, 0.1, 7),
        ("normal20", "mean", 300, 0.05, 0),
        ("expon50", "median", 250, 0.05, 3),
    ]:
        a = np.asarray(samples[name], dtype=np.float64)
        st = np.mean if statname == "mean" else np.median
        g = pcg.PCG32(seed=seed)
        n = a.size
        thetas = []
        for _ in range(n_boot):
            idx = [min(int(g.uniform() * n), n - 1) for _ in range(n)]
            thetas.append(float(st(a[idx])))
        th = np.asarray(thetas)
        boot.append(
            [
                name,
                statname,
                n_boot,
                alpha,
                seed,
                f(st(a)),
                f(np.quantile(th, alpha / 2, method="linear")),
                f(np.quantile(th, 1 - alpha / 2, method="linear")),
            ]
        )
    out["bootstrap"] = boot

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1) + "\n"
    OUT.write_text(text)
    b = text.encode()
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/M07.4/scipy_golden.py\t"
        f"scipy=={scipy.__version__} numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
