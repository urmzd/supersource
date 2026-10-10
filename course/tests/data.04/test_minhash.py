"""Course tests for data.04: near-duplicate detection (MinHash, LSH,
union-find) and decontamination (corpus/minhash.py).

Rung R0 for these course tests (your own tests for this module are rung R4,
properties: section 4 of the chapter lists them). Each test names why it
exists (WHY), what kind of check it is (KIND), the planted bugs it kills
(CATCHES, mutants in course/mutants/data.04), and the chapter section it
comes from.

Fixtures (course/oracle/data.04/make_fixtures.py, synthetic pseudo-words):
near-dups.jsonl has planted clusters (labels from how each document was
made) and borderline documents whose Jaccard with a base is 0.45 to 0.6;
decontam-docs.jsonl has documents labelled by whether they embed a 13-word
span of protected/val-suite.jsonl or protected/heldout.txt.

The chapter's worked example (section 3):

    A = "the cat sat on the mat today", B = "the cat sat on the mat again"
    3-shingles: A and B share 4 of 6 distinct shingles, Jaccard 2/3
    LSH(16 bands, 8 rows): threshold (1/16)^(1/8) = 0.7071,
    P(candidate | s = 0.8) = 0.9470, P(candidate | s = 0.5) = 0.0607
"""

from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from corpus.minhash import (
    LSH,
    MERSENNE31,
    UnionFind,
    clusters,
    contaminated,
    decontaminate,
    estimate,
    jaccard,
    minhash,
    near_dedup,
    perm_params,
    protected_ngrams,
    shingle_hashes,
    shingles,
    signatures,
    words,
)
from corpus.stage import Doc

FX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "data.04"
A = "the cat sat on the mat today"
B = "the cat sat on the mat again"
P = (1 << 31) - 1


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64 written out here, so the test does not trust M06.3's."""
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return h


def near_dup_rows() -> list[dict]:
    with open(FX / "near-dups.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def planted(rows: list[dict]) -> list[set[str]]:
    groups: dict[str, set[str]] = {}
    for r in rows:
        groups.setdefault(r["group"], set()).add(r["id"])
    return sorted((g for g in groups.values() if len(g) > 1), key=min)


def protected() -> list[Path]:
    return [FX / "protected" / "val-suite.jsonl", FX / "protected" / "heldout.txt"]


def doc(i: str, text: str, **meta) -> Doc:
    return Doc(id=i, source_id="s", text=text, meta=meta)


def fresh_words(rng: PCG32, n: int) -> list[str]:
    return [f"w{rng.next_u32():08x}" for _ in range(n)]


# --- the worked example ------------------------------------------------------------


def test_hand_example():
    # WHY: section 3 by hand. The 3-shingles of A and B, their Jaccard
    #      similarity 4/6, and the two LSH numbers the chapter computes:
    #      the threshold (1/b)^(1/r) and the candidate probability
    #      1 - (1 - s^r)^b at s = 0.8 and s = 0.5.
    # KIND: unit
    # CATCHES: s03, m01
    # CHAPTER: data.04 section 3, Worked example by hand
    sa, sb = shingles(A, 3), shingles(B, 3)
    assert sa == {
        "the cat sat",
        "cat sat on",
        "sat on the",
        "on the mat",
        "the mat today",
    }
    assert sa & sb == {"the cat sat", "cat sat on", "sat on the", "on the mat"}
    assert jaccard(sa, sb) == 4 / 6
    lsh = LSH(16, 8)
    assert_close(lsh.threshold, 0.70710678, rtol=0, atol=1e-8)
    assert_close(lsh.probability(0.8), 0.94704880, rtol=0, atol=1e-8)
    assert_close(lsh.probability(0.5), 0.06070190, rtol=0, atol=1e-8)


def test_signature_is_the_spec():
    # WHY: every implementation must produce the same signature for the
    #      same text and seed, or two runs (or a resumed run) cluster
    #      differently. The test recomputes it from the contract: FNV-1a 64
    #      of the UTF-8 shingle mod P, and (a_i, b_i) drawn interleaved from
    #      PCG32(seed), a_i = 1 + below(P - 1), b_i = below(P).
    # KIND: unit
    # CATCHES: s04, s05, s06, m02, m04
    # CHAPTER: data.04 section 2, Principles
    rng = PCG32(7)
    a, b = [], []
    for _ in range(8):
        a.append(1 + rng.below(P - 1))
        b.append(rng.below(P))
    ga, gb = perm_params(8, 7)
    assert [int(x) for x in ga] == a and [int(x) for x in gb] == b
    sh = sorted(shingles("Naïve café: the café is naïve, the café is open.", 2))
    xs = [fnv1a64(s.encode("utf-8")) % P for s in sh]
    assert [int(x) for x in shingle_hashes(sh)] == xs
    want = [min((a[i] * x + b[i]) % P for x in xs) for i in range(8)]
    got = minhash(sh, 8, 7)
    assert got.dtype == np.uint64 and got.shape == (8,)
    assert [int(v) for v in got] == want


def test_words_and_short_texts():
    # WHY: shingles are over lower-cased Unicode word runs, so "Cat," and
    #      "cat" agree; a text shorter than k words is still one shingle
    #      (a 3-word document must be able to match its copy), and a text
    #      with no words has no shingles at all.
    # KIND: boundary
    # CATCHES: s07, m03
    # CHAPTER: data.04 section 5, Pitfalls, item 1
    assert words("The CAT, sat; naïve-ly!") == ["the", "cat", "sat", "naïve", "ly"]
    assert shingles("Hello, world", 5) == {"hello world"}
    assert shingles("!!! ...", 5) == set()
    with pytest.raises(ValueError):
        shingles("a b", 0)


def test_empty_texts_are_nobodys_duplicate():
    # WHY: every text without words has the same all-P signature, so a
    #      careless index would put every empty document in one giant
    #      cluster and keep one of them. They are not near duplicates of
    #      anything: their Jaccard is 0 and LSH must not index them.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: data.04 section 5, Pitfalls, item 2
    sig = minhash(set(), 16, 0)
    assert [int(v) for v in sig] == [MERSENNE31] * 16
    assert jaccard(set(), set()) == 0.0
    got = clusters({"e1": "", "e2": "!!!", "e3": "...", "x": A}, num_perm=16, bands=4)
    assert got == []


# --- the estimator and the S-curve ------------------------------------------------------


def test_estimator_is_unbiased():
    # WHY: Pr[min-hash equal] = Jaccard, so the share of equal entries is
    #      an unbiased estimate. 40 fresh pairs at J = 1/3 (100 shared of
    #      300 distinct words), 128 entries each, give 5120 Bernoulli(1/3)
    #      trials; their mean must sit within the 1e-3 two-sided z bound
    #      (|z| < 3.29). A biased hash (or comparing max to min) drifts.
    # KIND: statistical
    # CATCHES: s11
    # CHAPTER: data.04 section 2, Principles
    rng = PCG32(101)
    texts = []
    for _ in range(40):
        shared = fresh_words(rng, 100)
        texts.append(" ".join(shared + fresh_words(rng, 100)))
        texts.append(" ".join(shared + fresh_words(rng, 100)))
    sig = signatures(texts, num_perm=128, seed=0, k=1)
    est = [estimate(sig[2 * i], sig[2 * i + 1]) for i in range(40)]
    j = 1 / 3
    z = (np.mean(est) - j) / math.sqrt(j * (1 - j) / (128 * 40))
    assert abs(z) < 3.29, (
        f"mean estimate {np.mean(est):.4f} vs Jaccard {j:.4f} (z = {z:.2f})"
    )


def test_lsh_s_curve():
    # WHY: banding turns the estimate into an S-curve: two documents at
    #      similarity s become candidates with probability 1 - (1 - s^r)^b.
    #      150 fresh pairs at each s in 0.55..0.85 must fit that curve
    #      (chi-square over 13 points, p > 1e-3), and the measured 50%
    #      point must be within 0.05 of (1/b)^(1/r). Bands that collide
    #      with each other, or the wrong slice of rows, move the curve.
    # KIND: statistical
    # CATCHES: s02, s03, m01
    # CHAPTER: data.04 section 2, Principles
    rng = PCG32(11)
    grid = [0.55 + 0.025 * i for i in range(13)]
    trials, union = 150, 400
    texts = []
    for s in grid:
        c = int(round(union * s))
        c -= (union - c) % 2
        u = (union - c) // 2
        for _ in range(trials):
            shared = fresh_words(rng, c)
            texts.append(" ".join(shared + fresh_words(rng, u)))
            texts.append(" ".join(shared + fresh_words(rng, u)))
    sig = signatures(texts, num_perm=128, seed=0, k=1)
    lsh0 = LSH(16, 8)
    rates, chi2 = [], 0.0
    for g, s in enumerate(grid):
        hits = 0
        for t in range(trials):
            i = 2 * (g * trials + t)
            lsh = LSH(16, 8)
            lsh.insert("a", sig[i])
            lsh.insert("b", sig[i + 1])
            hits += bool(lsh.candidates())
        p = lsh0.probability(s)
        rates.append(hits / trials)
        chi2 += (hits - trials * p) ** 2 / (trials * p * (1 - p))
    assert chi2 < 34.53, (
        f"candidate rates {rates} do not fit the S-curve (chi2 {chi2:.1f}, 13 dof)"
    )
    k = next(i for i, r in enumerate(rates) if r >= 0.5)
    s50 = grid[k - 1] + (0.5 - rates[k - 1]) * (grid[k] - grid[k - 1]) / (
        rates[k] - rates[k - 1]
    )
    assert abs(s50 - lsh0.threshold) < 0.05, (
        f"50% point {s50:.3f}, (1/b)^(1/r) = {lsh0.threshold:.3f}"
    )


# --- LSH and union-find ---------------------------------------------------------------


def test_bands_do_not_collide_with_each_other():
    # WHY: a bucket is (band index, the band's rows). Without the index, a
    #      band of one document that happens to equal a different band of
    #      another makes a false candidate: band 0 of x equals band 1 of y
    #      here, and no band is equal at the same position.
    # KIND: boundary
    # CATCHES: s01
    # CHAPTER: data.04 section 5, Pitfalls, item 3
    x = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
    y = np.array([9, 9, 1, 2, 7, 7], dtype=np.uint64)
    lsh = LSH(3, 2)
    lsh.insert("x", x)
    lsh.insert("y", y)
    assert lsh.candidates() == []
    lsh.insert("z", np.array([8, 8, 3, 4, 8, 8], dtype=np.uint64))
    assert lsh.candidates() == [("x", "z")]


def test_candidates_are_sorted_unique_pairs():
    # WHY: two keys equal in every band share all b buckets but are one
    #      candidate pair, written (smaller, larger) and sorted, so the
    #      union order and the output are deterministic.
    # KIND: unit
    # CATCHES: m05
    # CHAPTER: data.04 section 4, The interface
    lsh = LSH(2, 2)
    for key in ("c", "a", "b"):
        lsh.insert(key, np.array([1, 2, 3, 4], dtype=np.uint64))
    lsh.insert("d", np.array([1, 2, 9, 9], dtype=np.uint64))
    assert lsh.candidates() == [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ]
    with pytest.raises(ValueError):
        lsh.insert("a", np.array([1, 2, 3, 4], dtype=np.uint64))
    with pytest.raises(ValueError):
        lsh.insert("e", np.array([1, 2, 3], dtype=np.uint64))
    with pytest.raises(ValueError):
        LSH(0, 4)


def test_union_find_root_is_the_smallest_member():
    # WHY: the cluster id written into every shard row is the root, and the
    #      document kept is the root. Union by size or by argument order
    #      would make both depend on the order pairs arrive in; the
    #      smallest member does not.
    # KIND: property
    # CATCHES: s09, m06
    # CHAPTER: data.04 section 5, Pitfalls, item 4
    pairs = [("d", "c"), ("c", "b"), ("x", "y"), ("b", "a"), ("e", "e")]
    for perm in itertools.permutations(pairs):
        uf = UnionFind()
        for x, y in perm:
            uf.union(x, y)
        assert uf.groups() == [{"a", "b", "c", "d"}, {"x", "y"}]
        assert uf.find("d") == "a" and uf.find("y") == "x" and uf.find("e") == "e"


# --- clusters on the fixture ------------------------------------------------------------


def test_fixture_clusters_match_planted():
    # WHY: every planted cluster (one-word edits and truncations, Jaccard
    #      >= 0.93 with their base) is found exactly, for three seeds, and
    #      nothing else is: the 12 borderline documents (Jaccard 0.45 to
    #      0.6 with a base) stay alone at threshold 0.8. A run that skips
    #      the estimate check after LSH merges some of them (about a
    #      quarter of pairs at 0.6 become candidates).
    # KIND: golden
    # CATCHES: s10, s12
    # CHAPTER: data.04 section 4, What the tests check
    rows = near_dup_rows()
    docs = {r["id"]: r["text"] for r in rows}
    want = planted(rows)
    for seed in (0, 1, 2):
        assert clusters(docs, seed=seed) == want, f"seed {seed}"


def test_chain_is_one_cluster():
    # WHY: near-duplicate is not transitive, but clustering is: x ~ y and
    #      y ~ z put x, y, z in one cluster even when x and z are too far
    #      apart to be confirmed. Keeping "every document not similar to a
    #      kept one" keeps both x and z. Word sets on a line: x = 0..199,
    #      y = 40..239, z = 80..279 (k = 1), so J(x, y) = J(y, z) = 2/3 and
    #      J(x, z) = 0.43; 512 entries put both 5 standard deviations from
    #      the 0.55 threshold.
    # KIND: unit
    # CATCHES: s11, s12
    # CHAPTER: data.04 section 5, Pitfalls, item 5
    t = {
        k: " ".join(f"v{i}" for i in range(lo, lo + 200))
        for k, lo in (("x", 0), ("y", 40), ("z", 80))
    }
    sx, sy, sz = (shingles(t[k], 1) for k in "xyz")
    assert (
        jaccard(sx, sy) == 2 / 3
        and jaccard(sy, sz) == 2 / 3
        and jaccard(sx, sz) == 120 / 280
    )
    opts = dict(k=1, num_perm=512, bands=128, threshold=0.55)
    assert clusters(t, **opts) == [{"x", "y", "z"}]
    kept = [d.id for d in near_dedup([doc(k, v) for k, v in t.items()], **opts)]
    assert kept == ["x"]


def test_invariant_to_worker_count():
    # WHY: the corpus build (data.09) and MS-corpus compare output hashes
    #      across worker counts. Signatures computed on 1 or 3 processes
    #      must be identical, so every worker must use the same hash
    #      parameters (not one seed per chunk), and chunks must come back
    #      in order.
    # KIND: property
    # CATCHES: s13, s14
    # CHAPTER: data.04 section 5, Pitfalls, item 6
    rows = near_dup_rows()
    texts = [r["text"] for r in rows[:61]]
    one = signatures(texts, workers=1)
    three = signatures(texts, workers=3)
    assert one.dtype == np.uint64 and one.shape == (61, 128)
    assert np.array_equal(one, three)
    docs = {r["id"]: r["text"] for r in rows}
    assert clusters(docs, workers=3) == clusters(docs, workers=1)
    with pytest.raises(ValueError):
        signatures(texts, workers=0)


def test_invariant_to_input_order():
    # WHY: the stage runs on whatever order the previous stage yields. The
    #      clusters, and which document survives (the smallest id), must
    #      not depend on it.
    # KIND: property
    # CATCHES: s10, s12, s15
    # CHAPTER: data.04 section 2, Principles
    rows = near_dup_rows()
    fwd = [doc(r["id"], r["text"]) for r in rows]
    rev = list(reversed(fwd))
    kept_fwd = sorted(d.id for d in near_dedup(fwd))
    kept_rev = sorted(d.id for d in near_dedup(rev))
    assert kept_fwd == kept_rev
    roots = {min(g) for g in planted(rows)}
    dropped = {r["id"] for r in rows} - set(kept_fwd)
    assert not (dropped & roots)
    assert len(dropped) == sum(len(g) - 1 for g in planted(rows))


def test_near_dedup_tags_keeps_order_and_meta():
    # WHY: data.06 turns meta["minhash_cluster"] into the shard's
    #      minhash_cluster column, so every document carries its root id
    #      (its own id when alone). drop=False keeps everything for
    #      inspection; drop=True keeps the roots. Order and the other meta
    #      keys pass through.
    # KIND: unit
    # CATCHES: s15, s16, m07, m08
    # CHAPTER: data.04 section 4, The interface
    t = A + " " + " ".join(f"w{i}" for i in range(30))
    docs = [
        doc("b", t, url="ub"),
        doc("c", "something else entirely here"),
        doc("a", t + " x", url="ua"),
    ]
    tagged = list(near_dedup(docs, drop=False, num_perm=64, bands=8))
    assert [d.id for d in tagged] == ["b", "c", "a"]
    assert [d.meta["minhash_cluster"] for d in tagged] == ["a", "c", "a"]
    assert tagged[0].meta["url"] == "ub" and tagged[0].text == t
    kept = list(near_dedup(docs, num_perm=64, bands=8))
    assert [d.id for d in kept] == ["c", "a"]
    with pytest.raises(ValueError):
        list(near_dedup([doc("a", "x"), doc("a", "y")]))
    with pytest.raises(ValueError):
        clusters({"a": "x"}, num_perm=100, bands=16)


# --- decontamination ------------------------------------------------------------------


def test_decontaminate_fixture():
    # WHY: a 13-word span of a protected text anywhere in a document (start,
    #      middle, end; upper-cased; punctuation changed) contaminates it;
    #      a 12-word span does not, and neither do the suite's case ids,
    #      tags, or scorer_args, which are not text the model is tested on.
    # KIND: golden
    # CATCHES: s17, s18, m03, m09
    # CHAPTER: data.04 section 4, What the tests check
    with open(FX / "decontam-docs.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    grams = protected_ngrams(protected())
    wrong = [
        r["id"] + ": " + r["why"]
        for r in rows
        if contaminated(r["text"], grams) != r["contaminated"]
    ]
    assert wrong == []
    docs = [doc(r["id"], r["text"]) for r in rows]
    kept = [d.id for d in decontaminate(docs, protected())]
    assert kept == [r["id"] for r in rows if not r["contaminated"]]


def test_span_at_the_very_end():
    # WHY: the windows of a w-word text start at 0 .. w - n inclusive. A
    #      range that stops one short never looks at the last n-gram, so a
    #      document that ends with the protected span passes.
    # KIND: boundary
    # CATCHES: s18
    # CHAPTER: data.04 section 5, Pitfalls, item 7
    grams = {" ".join(f"p{i}" for i in range(13))}
    tail = " ".join(f"p{i}" for i in range(13))
    assert contaminated("a b c " + tail, grams)
    assert contaminated(tail, grams)
    assert not contaminated(" ".join(f"p{i}" for i in range(12)), grams)
    assert contaminated("q0 q1 " + tail, {" ".join(f"p{i}" for i in range(13))}, n=13)


def test_protected_files(tmp_path):
    # WHY: JSON Lines files contribute every string value except under
    #      case_id, id, tags, and scorer_args (recursively, so chat
    #      messages count); any other file is one text; a line that is not
    #      an object is an error, not silently skipped.
    # KIND: boundary
    # CATCHES: s19, m09, m10
    # CHAPTER: data.04 section 2, Principles
    words13 = " ".join(f"m{i}" for i in range(13))
    p = tmp_path / "suite.jsonl"
    rows = [
        {
            "case_id": words13,
            "tags": [words13],
            "input": [{"role": "user", "content": "c " + words13}],
        },
        {"id": "x", "scorer_args": {"regex": " ".join(f"r{i}" for i in range(13))}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    grams = protected_ngrams([p], 13)
    assert words13 in grams
    assert " ".join(f"r{i}" for i in range(13)) not in grams
    assert "c " + " ".join(f"m{i}" for i in range(12)) in grams
    q = tmp_path / "plain.txt"
    q.write_text('{"case_id": "' + words13 + '"}')
    assert words13 in protected_ngrams([q], 13)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("[1, 2]\n")
    with pytest.raises(ValueError):
        protected_ngrams([bad])


def test_decontaminate_streams():
    # WHY: decontamination runs on the full corpus. After reading the
    #      protected sets once it must pull one document at a time: here it
    #      yields 5 documents of an endless stream through islice, and the
    #      stream fails if more than 50 are pulled.
    # KIND: property
    # CATCHES: s20, m09
    # CHAPTER: data.04 section 4, The interface
    def endless():
        for i in itertools.count():
            if i > 50:
                raise AssertionError(
                    "decontaminate read far past what islice asked for"
                )
            yield doc(f"d{i}", f"plain story number {i}")

    out = list(itertools.islice(decontaminate(endless(), protected()), 5))
    assert [d.id for d in out] == ["d0", "d1", "d2", "d3", "d4"]


def test_after_exact_dedup():
    # WHY: the pipeline runs exact dedup (data.03) first. Exact copies are
    #      gone before near dedup sees the stream, so near_dropped counts
    #      only true near duplicates.
    # KIND: regression
    # CHAPTER: data.04 section 6, Where it's used next
    from corpus.dedup import exact_dedup

    rows = near_dup_rows()[:40]
    docs = [doc(r["id"], r["text"]) for r in rows]
    copies = [doc(d.id + ":copy", d.text) for d in docs[:5]]
    unique = list(exact_dedup(iter(docs + copies)))
    assert sorted(d.text for d in unique) == sorted(d.text for d in docs)
    near = list(near_dedup(iter(unique)))
    assert len(unique) - len(near) == sum(len(g) - 1 for g in planted(rows))
