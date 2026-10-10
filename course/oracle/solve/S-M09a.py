"""Recompute every auto-checked answer of S-M09a independently (see _confirm.py).

Float32 fields from numpy's own bit view; gaps with numpy.nextafter and from
the format parameters; bf16 rounding by exact rational nearest-value search
(not the uint32 bit trick M09.1 teaches); cancellation by stepwise numpy
float32 or float64 evaluation of the naive formulas.
"""

from __future__ import annotations

import sys
import warnings
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def fields(x: float) -> str:
    u = int(np.array([x], dtype=np.float32).view(np.uint32)[0])
    return f"[{u >> 31}, {(u >> 23) & 0xFF}, {u & 0x7FFFFF}]"


def frac_str(q: F) -> str:
    return f"{q.numerator}/{q.denominator}"


def round_rne(x: F, frac_bits: int) -> F:
    """Nearest value with `frac_bits` fraction bits in x's binade; ties to an even last bit."""
    e = 0
    while F(2) ** (e + 1) <= x:
        e += 1
    while F(2) ** e > x:
        e -= 1
    step = F(2) ** (e - frac_bits)
    lo = (x // step) * step
    hi = lo + step
    if x - lo < hi - x:
        return lo
    if hi - x < x - lo:
        return hi
    return lo if (lo / step) % 2 == 0 else hi


def main() -> int:
    c: dict[str, str] = {}
    c["q1.a"] = fields(1.0)
    c["q1.b"] = fields(-6.5)
    for qid, dt in (("q2.a", np.float32), ("q2.c", np.float16)):
        one = dt(1)
        gap = F(float(np.nextafter(one, dt(2)) - one))
        c[qid] = frac_str(gap)
    # bf16 is the upper 16 bits of a float32: step those bits by one above 1.0
    hi16 = int(np.array([1.0], dtype=np.float32).view(np.uint32)[0]) >> 16
    nxt = np.array([(hi16 + 1) << 16], dtype=np.uint32).view(np.float32)[0]
    c["q2.b"] = frac_str(F(float(nxt)) - 1)
    c["q3.a"] = str(int(np.finfo(np.float16).max))
    tiny = F(float(np.finfo(np.float32).tiny))
    assert tiny.numerator == 1 and tiny.denominator & (tiny.denominator - 1) == 0
    c["q3.b"] = str(-(tiny.denominator.bit_length() - 1))
    c["q4.a"] = frac_str(round_rne(F(1) + F(1, 2**8), 7))
    c["q4.b"] = frac_str(round_rne(F(1) + F(3, 2**8), 7))
    c["q5"] = str(F(1, 10).denominator & (F(1, 10).denominator - 1) == 0).lower()

    x = np.float32(1e8)
    naive = np.sqrt(x + np.float32(1)) - np.sqrt(x)
    assert naive.dtype == np.float32
    c["q6.a"] = str(int(naive)) if naive == int(naive) else repr(float(naive))
    c["q6.b"] = str(sp.N(sp.sqrt(10**8 + 1) - sp.sqrt(10**8), 30))

    xs = 1e-8
    exact = sp.N(1 - sp.cos(sp.Rational(1, 10**8)), 30)
    err_a = abs(sp.Float(1 - np.cos(xs), 30) - exact) / exact
    err_b = abs(sp.Float(2 * np.sin(xs / 2) ** 2, 30) - exact) / exact
    c["q7"] = "a" if err_a < err_b else "b"

    c["q8.a"] = str(sp.log(2 * sp.exp(1000)).expand(force=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.array([1000.0, 1000.0])
        sm = np.exp(v) / np.sum(np.exp(v))
    c["q8.b"] = "nan" if np.isnan(sm[0]) else str(sm[0])

    data = [10**8 + 1, 10**8 + 2, 10**8 + 3]
    mean = F(sum(data), 3)
    c["q9.a"] = frac_str(sum((F(d) - mean) ** 2 for d in data) / 3)
    a = np.array(data, dtype=np.float64)
    one = np.mean(a * a) - np.mean(a) ** 2
    two = np.mean((a - np.mean(a)) ** 2)
    target = 2 / 3
    c["q9.b"] = "two" if abs(two - target) < abs(one - target) else "one"
    assert abs(two - target) < 1e-15 and abs(one - target) > 1e-3, (one, two)

    s = np.float32(1.0)
    tiny = np.float32(1e-8)
    for _ in range(10**4):
        s = np.float32(s + tiny)
    c["q10.a"] = str(int(s)) if s == int(s) else repr(float(s))
    c["q10.b"] = frac_str(F(1) + 10**4 * F(1, 10**8))
    return confirm("S-M09a", c)


if __name__ == "__main__":
    sys.exit(main())
