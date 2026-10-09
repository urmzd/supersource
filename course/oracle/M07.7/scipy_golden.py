# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy==1.17.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M07.7 golden fixture (scipy).

scikit-learn, the design's oracle for this module, is not available offline
here (DEVIATIONS B71-05), so every value comes from an independent route:

  logistic   the minimizer of the penalized negative log-likelihood of
             contracts/py/tinyllm/prob/metrics.pyi by scipy.optimize.minimize
             (BFGS, analytic gradient, gtol 1e-11), a quasi-Newton method
             that never forms the Hessian the reference inverts; the gradient
             norm at the answer is recorded
  roc        the ROC corners by brute force (for each distinct score t,
             count scores >= t per class) and the AUC as the Mann-Whitney U
             statistic, scipy.stats.mannwhitneyu(pos, neg).statistic / (P N)
  ece        reliability bins and ECE with exact rational bin indices
             (fractions.Fraction), binary and 3-class

Data are drawn from the frozen PCG32 (course/tests/_lib/pcg32.py).

    uv run --offline --python 3.11 --script course/oracle/M07.7/scipy_golden.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy import optimize, stats

OUT = Path("course/fixtures/M07.7/scipy_golden.json")
spec = importlib.util.spec_from_file_location("pcg", "course/tests/_lib/pcg32.py")
pcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcg)
g = pcg.PCG32(seed=707)


def f(x) -> float:
    return float(x)


def logistic_case(name, n, d, l2, fit_intercept, w_true, b_true):
    X = np.array([[g.normal() for _ in range(d)] for _ in range(n)])
    X[:, -1] = 0.5 * X[:, 0] + 0.5 * X[:, -1]  # correlated features
    z = X @ np.asarray(w_true) + b_true
    y = np.array([1.0 if g.uniform() < 1.0 / (1.0 + math.exp(-v)) else 0.0 for v in z])
    Xa = np.c_[X, np.ones(n)] if fit_intercept else X
    pen = np.full(Xa.shape[1], l2)
    if fit_intercept:
        pen[-1] = 0.0

    def loss(w):
        zz = Xa @ w
        return float(np.sum(np.logaddexp(0.0, zz) - y * zz) + 0.5 * np.sum(pen * w * w))

    def grad(w):
        p = 1.0 / (1.0 + np.exp(-(Xa @ w)))
        return Xa.T @ (p - y) + pen * w

    res = optimize.minimize(
        loss,
        np.zeros(Xa.shape[1]),
        jac=grad,
        method="BFGS",
        options={"gtol": 1e-11, "maxiter": 10000},
    )
    gn = float(np.max(np.abs(grad(res.x))))
    assert gn < 1e-8, (name, gn)
    return {
        "name": name,
        "X": X.tolist(),
        "y": y.tolist(),
        "l2": l2,
        "fit_intercept": fit_intercept,
        "w": res.x.tolist(),
        "grad_inf": gn,
    }


def roc_case(name, n, ties):
    scores, labels = [], []
    for _ in range(n):
        lab = 1.0 if g.uniform() < 0.4 else 0.0
        s = g.normal() + (1.0 if lab else 0.0)
        if ties:
            s = round(s, 1)
        scores.append(s)
        labels.append(lab)
    s, t = np.array(scores), np.array(labels)
    P, N = t.sum(), (1 - t).sum()
    thr = sorted(set(scores), reverse=True)
    fpr = [0.0] + [f(np.sum((s >= v) & (t == 0)) / N) for v in thr]
    tpr = [0.0] + [f(np.sum((s >= v) & (t == 1)) / P) for v in thr]
    u = stats.mannwhitneyu(s[t == 1], s[t == 0]).statistic
    return {
        "name": name,
        "scores": scores,
        "labels": labels,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
        "auc": f(u / (P * N)),
    }


def bins_exact(conf, correct, B):
    counts = [0] * B
    csum = [0.0] * B
    asum = [0.0] * B
    for c, ok in zip(conf, correct):
        b = math.ceil(Fraction(c) * B) - 1
        b = min(max(b, 0), B - 1)
        counts[b] += 1
        csum[b] += c
        asum[b] += 1.0 if ok else 0.0
    n = len(conf)
    confm = [csum[b] / counts[b] if counts[b] else 0.0 for b in range(B)]
    accm = [asum[b] / counts[b] if counts[b] else 0.0 for b in range(B)]
    e = sum(counts[b] / n * abs(accm[b] - confm[b]) for b in range(B) if counts[b])
    return counts, confm, accm, e


def ece_cases():
    out = []
    # binary, calibrated-ish and overconfident
    for name, sharpen in [("binary_calibrated", 1.0), ("binary_overconfident", 3.0)]:
        probs, labels = [], []
        for _ in range(400):
            p = g.uniform()
            labels.append(1 if g.uniform() < p else 0)
            z = math.log(max(p, 1e-12) / max(1 - p, 1e-12)) * sharpen
            probs.append(1.0 / (1.0 + math.exp(-z)))
        conf = [max(p, 1 - p) for p in probs]
        correct = [(1 if p >= 0.5 else 0) == y for p, y in zip(probs, labels)]
        for B in (10, 15):
            c, cm, am, e = bins_exact(conf, correct, B)
            out.append(
                {
                    "name": name,
                    "n_bins": B,
                    "probs": probs,
                    "labels": labels,
                    "counts": c,
                    "confidence": cm,
                    "accuracy": am,
                    "ece": e,
                }
            )
    # three classes
    probs, labels = [], []
    for _ in range(300):
        z = [g.normal() * 1.5 for _ in range(3)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        p = [v / s for v in e]
        u, acc, lab = g.uniform(), 0.0, 2
        for k in range(3):
            acc += p[k]
            if u < acc:
                lab = k
                break
        probs.append(p)
        labels.append(lab)
    conf = [max(p) for p in probs]
    correct = [int(np.argmax(p)) == y for p, y in zip(probs, labels)]
    c, cm, am, e = bins_exact(conf, correct, 15)
    out.append(
        {
            "name": "three_class",
            "n_bins": 15,
            "probs": probs,
            "labels": labels,
            "counts": c,
            "confidence": cm,
            "accuracy": am,
            "ece": e,
        }
    )
    return out


def main() -> None:
    out: dict = {
        "__meta__": {
            "generator": "course/oracle/M07.7/scipy_golden.py",
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        }
    }
    out["logistic"] = [
        logistic_case("ridge_d3", 80, 3, 1.0, True, [1.5, -2.0, 0.5], -0.3),
        logistic_case("weak_l2_d2", 120, 2, 0.01, True, [0.8, -0.4], 0.6),
        logistic_case("no_intercept_d4", 60, 4, 0.5, False, [1.0, 0.0, -1.0, 2.0], 0.0),
        logistic_case("mle_d2", 150, 2, 0.0, True, [0.7, 0.3], -0.2),
    ]
    out["roc"] = [roc_case("continuous", 60, False), roc_case("tied", 80, True)]
    out["ece"] = ece_cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1) + "\n"
    OUT.write_text(text)
    b = text.encode()
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/M07.7/scipy_golden.py\t"
        f"scipy=={scipy.__version__} numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
