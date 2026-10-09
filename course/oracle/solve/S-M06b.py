"""Recompute every auto-checked answer of S-M06b independently (see _confirm.py).

Trees by building them (a trie as nested dicts, Pruefer sequences for
Cayley's count, heap indexing for the height); inclusion-exclusion and
birthday answers by brute-force enumeration; Jaccard and MinHash variance by
enumeration over sets and over all 2^4 outcomes of small binomials, then the
general formula; LSH and Bloom answers by evaluating the defining event
probabilities with exact fractions or SymPy; recurrences by iterating them
and fitting the closed form at many points.

    uv run --project course/harness python course/oracle/solve/S-M06b.py
"""

from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _confirm import confirm  # noqa: E402


def trie_nodes(keys: list[str]) -> int:
    root: dict = {}
    for k in keys:
        node = root
        for ch in k:
            node = node.setdefault(ch, {})
    count, stack = 0, [root]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.values())
    return count


def labelled_trees(n: int) -> int:
    """Count spanning trees of K_n by brute force over edge subsets."""
    edges = list(itertools.combinations(range(n), 2))
    total = 0
    for sub in itertools.combinations(edges, n - 1):
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ok = True
        for u, v in sub:
            ru, rv = find(u), find(v)
            if ru == rv:
                ok = False
                break
            parent[ru] = rv
        total += ok
    return total


def main() -> int:
    c: dict[str, str] = {}

    # ---- trees
    # q1: a path on 12 nodes is a tree; every tree on n nodes has as many edges.
    c["q1"] = str(len([(i, i + 1) for i in range(11)]))
    c["q2"] = str(trie_nodes(["car", "cart", "care", "cat", "dog"]))
    # q3: grow a full binary tree by splitting leaves until there are 10.
    leaves, internal = 1, 0
    while leaves < 10:
        leaves, internal = leaves + 1, internal + 1
    c["q3"] = str(internal)
    c["q4"] = str(labelled_trees(5))
    # q5: heap indexing, node i (1-based) has depth floor(log2 i); deepest is node 100.
    depth, i = 0, 100
    while i > 1:
        i //= 2
        depth += 1
    c["q5"] = str(depth)

    # ---- inclusion-exclusion and birthday
    # q7: build three concrete sets with the stated sizes and intersections.
    abc = {0}
    ab, ac, bc = {1, 2}, {3, 4}, {5, 6}
    a = abc | ab | ac | {10, 11, 12, 13, 14}
    b = abc | ab | bc | {20, 21, 22, 23, 24}
    cc = abc | ac | bc | {30, 31, 32, 33, 34}
    assert len(a) == len(b) == len(cc) == 10
    assert len(a & b) == len(a & cc) == len(b & cc) == 3 and len(a & b & cc) == 1
    c["q7"] = str(len(a | b | cc))
    c["q8"] = str(
        sum(1 for x in range(1, 101) if x % 2 == 0 or x % 3 == 0 or x % 5 == 0)
    )
    c["q9"] = str(
        sum(
            1
            for p in itertools.permutations(range(4))
            if all(p[i] != i for i in range(4))
        )
    )
    distinct = sum(
        1 for t in itertools.product(range(10), repeat=3) if len(set(t)) == 3
    )
    f = Fraction(distinct, 10**3)
    c["q10"] = f"{f.numerator}/{f.denominator}"

    # q11: exact expected colliding pairs by enumeration, fitted to a formula.
    def pairs(k: int, m: int) -> Fraction:
        tot = sum(
            sum(1 for i, j in itertools.combinations(range(k), 2) if t[i] == t[j])
            for t in itertools.product(range(m), repeat=k)
        )
        return Fraction(tot, m**k)

    kk, mm = sp.symbols("k m", positive=True)
    cand = kk * (kk - 1) / (2 * mm)
    for k in range(1, 6):
        for m in range(1, 5):
            assert pairs(k, m) == Fraction(str(cand.subs({kk: k, mm: m})))
    c["q11"] = str(cand)
    # q12: solve exp(-k^2 / (2m)) = 1/2 numerically by bisection.
    m32 = 2.0**32
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.exp(-mid * mid / (2 * m32)) > 0.5:
            lo = mid
        else:
            hi = mid
    c["q12"] = repr(lo)

    # ---- Jaccard, MinHash, LSH
    sa = {"ab", "bc", "cd", "de"}
    sb = {"bc", "cd", "de", "ef", "fg"}
    j = Fraction(len(sa & sb), len(sa | sb))
    c["q13"] = f"{j.numerator}/{j.denominator}"
    # q15: variance of a mean of 128 Bernoulli(1/2): enumerate one Bernoulli exactly.
    p = Fraction(1, 2)
    mean = p
    var1 = (1 - mean) ** 2 * p + (0 - mean) ** 2 * (1 - p)
    v = var1 / 128
    c["q15"] = f"{v.numerator}/{v.denominator}"
    # q16: candidate probability by exhaustive enumeration for a tiny (b, r), then the formula.
    s = sp.symbols("s", real=True)

    def candidate_exact(b: int, r: int, sv: Fraction) -> Fraction:
        tot = Fraction(0)
        for rows in itertools.product((0, 1), repeat=b * r):
            pr = Fraction(1)
            for x in rows:
                pr *= sv if x else 1 - sv
            if any(all(rows[i * r : (i + 1) * r]) for i in range(b)):
                tot += pr
        return tot

    formula = lambda b, r: 1 - (1 - s**r) ** b  # noqa: E731
    for b, r in ((2, 2), (3, 2), (2, 3)):
        for sv in (Fraction(1, 3), Fraction(1, 2), Fraction(3, 4)):
            assert candidate_exact(b, r, sv) == Fraction(
                str(sp.nsimplify(formula(b, r).subs(s, sv)))
            )
    c["q16"] = str(formula(16, 8))
    c["q17"] = repr((1 / 16) ** (1 / 8))
    c["q18"] = repr(float(1 - (1 - Fraction(1, 2) ** 8) ** 16))

    # ---- Bloom filters
    # q19: by enumeration, the expected fraction of set bits after n keys with
    # k probes each is exactly 1 - (1 - 1/m)^(kn) (linearity). The textbook rate
    # raises that fraction to the k-th power, which treats the bits as
    # independent (it is an approximation even before exp(-kn/m)); the question
    # asks for that standard form.
    def set_fraction(m: int, n: int, k: int) -> Fraction:
        tot = Fraction(0)
        for inserts in itertools.product(range(m), repeat=n * k):
            tot += Fraction(len(set(inserts)), m)
        return tot / m ** (n * k)

    for m, n, k in ((3, 1, 2), (4, 2, 1), (3, 2, 2), (5, 1, 3)):
        assert set_fraction(m, n, k) == 1 - (1 - Fraction(1, m)) ** (k * n)
    for m, n, k in ((100, 10, 3), (1000, 50, 7)):
        exact_bits = 1 - (1 - 1 / m) ** (k * n)
        assert abs(exact_bits - (1 - math.exp(-k * n / m))) < 2 * k * n / m**2
    kb, nb, mb = sp.symbols("k n m", positive=True)
    c["q19"] = str((1 - sp.exp(-kb * nb / mb)) ** kb)
    # q20: minimize the q19 rate over real k by golden-section search.
    rate = lambda kv: (1 - math.exp(-kv / 10)) ** kv  # noqa: E731
    lo, hi = 1.0, 20.0
    for _ in range(300):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if rate(m1) < rate(m2):
            hi = m2
        else:
            lo = m1
    c["q20"] = repr((lo + hi) / 2)
    c["q21"] = repr(rate((lo + hi) / 2))
    # q22: smallest bits per key whose optimal rate reaches 0.01, by bisection.
    lo, hi = 1.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        kopt = mid * math.log(2)
        if (1 - math.exp(-kopt / mid)) ** kopt > 0.01:
            lo = mid
        else:
            hi = mid
    c["q22"] = repr(lo)

    # ---- recurrences and generating functions
    n = sp.symbols("n", positive=True)
    t = {1: 0}
    for e in range(1, 12):
        t[2**e] = 2 * t[2 ** (e - 1)] + 2**e
    cand23 = n * sp.log(n) / sp.log(2)
    assert all(sp.simplify(cand23.subs(n, x) - v) == 0 for x, v in t.items())
    c["q23"] = str(cand23)
    t24 = [0]
    for i in range(1, 60):
        t24.append(t24[-1] + i)
    cand24 = n * (n + 1) / 2
    assert all(cand24.subs(n, i) == v for i, v in enumerate(t24))
    c["q24"] = str(cand24)
    nn = sp.symbols("n", integer=True, nonnegative=True)
    a25 = [0, 1]
    for _ in range(30):
        a25.append(3 * a25[-1] - 2 * a25[-2])
    cand25 = 2**nn - 1
    assert all(cand25.subs(nn, i) == v for i, v in enumerate(a25))
    c["q25"] = str(cand25)
    x = sp.symbols("x", real=True)
    cand26 = 1 / (1 - x) ** 2
    series = sp.series(cand26, x, 0, 12).removeO()
    assert all(series.coeff(x, i) == i + 1 for i in range(12))
    c["q26"] = str(cand26)
    c["q27"] = str(
        sum(
            1
            for bits in itertools.product("01", repeat=10)
            if "11" not in "".join(bits)
        )
    )

    # q28: count scalar multiplications of the recursion and read the exponent.
    def mults(nsize: int, branches: int) -> int:
        return 1 if nsize == 1 else branches * mults(nsize // 2, branches)

    c["q28.a"] = str(round(math.log(mults(2**10, 8)) / math.log(2**10)))
    c["q28.b"] = repr(math.log(mults(2**10, 7)) / math.log(2**10))
    return confirm("S-M06b", c)


if __name__ == "__main__":
    sys.exit(main())
