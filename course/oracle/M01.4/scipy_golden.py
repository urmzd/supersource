# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy==1.17.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M01.4 golden fixture (scipy).

Records, as JSON with float64 values written by repr:
  rules       for named functions on [a, b] with n equal steps:
              scipy.integrate.trapezoid and scipy.integrate.simpson on the
              nodes numpy.linspace(a, b, n + 1), and scipy.integrate.quad's
              value of the exact integral (epsabs 1e-13, epsrel 1e-13)
  samples     scipy.integrate.trapezoid(y, x) on uneven, unsorted, and
              decreasing sample points drawn from the frozen PCG32
              (course/tests/_lib/pcg32.py)

    uv run --offline --python 3.12 --script course/oracle/M01.4/scipy_golden.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import integrate

OUT = Path("course/fixtures/M01.4/scipy_golden.json")
spec = importlib.util.spec_from_file_location("pcg", "course/tests/_lib/pcg32.py")
pcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcg)

# name -> numpy function; the course test holds the same table.
FUNCS = {
    "sin": np.sin,
    "exp": np.exp,
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x * x),
    "quartic": lambda x: x**4 - 3.0 * x**2 + x,
    "sqrt1p": lambda x: np.sqrt(1.0 + x),
    "gauss": lambda x: np.exp(-x * x / 2.0),
}
CASES = [
    ("sin", 0.0, float(np.pi), [2, 4, 8, 16, 64]),
    ("exp", 0.0, 1.0, [2, 6, 10, 32]),
    ("runge", -1.0, 1.0, [2, 8, 20, 100]),
    ("quartic", -1.5, 2.0, [2, 4, 12]),
    ("sqrt1p", 0.0, 3.0, [2, 10, 50]),
    ("gauss", -3.0, 3.0, [4, 16, 40]),
    ("exp", 1.0, -0.5, [2, 8]),
]


def f(x) -> float:
    return float(x)


def main() -> None:
    out: dict = {
        "__meta__": {
            "generator": "course/oracle/M01.4/scipy_golden.py",
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        }
    }
    rules = []
    for name, a, b, ns in CASES:
        fn = FUNCS[name]
        exact, _ = integrate.quad(
            lambda t: float(fn(np.float64(t))),
            a,
            b,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=200,
        )
        for n in ns:
            x = np.linspace(a, b, n + 1)
            y = fn(x)
            rules.append(
                [
                    name,
                    a,
                    b,
                    n,
                    f(integrate.trapezoid(y, x=x)),
                    f(integrate.simpson(y, x=x)),
                    f(exact),
                ]
            )
    out["rules"] = rules

    g = pcg.PCG32(seed=101)
    samples = []
    for kind, m in [
        ("uneven", 9),
        ("uneven", 40),
        ("unsorted", 12),
        ("decreasing", 15),
    ]:
        x = np.array([4.0 * g.uniform() - 1.0 for _ in range(m)])
        if kind in ("uneven", "decreasing"):
            x = np.sort(x)
        if kind == "decreasing":
            x = x[::-1].copy()
        y = np.array([3.0 * g.uniform() - 1.0 for _ in range(m)])
        samples.append([kind, x.tolist(), y.tolist(), f(integrate.trapezoid(y, x=x))])
    out["samples"] = samples

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1) + "\n"
    OUT.write_text(text)
    b = text.encode()
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/M01.4/scipy_golden.py\t"
        f"scipy=={scipy.__version__} numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
