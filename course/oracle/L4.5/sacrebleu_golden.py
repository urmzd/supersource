# /// script
# requires-python = ">=3.12"
# dependencies = ["sacrebleu==2.5.1", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L4.5 golden fixture (sacreBLEU).

Records, as JSON (floats written by repr):
  tokenize_13a   sacrebleu's Tokenizer13a on lines that exercise every rule
  bleu           corpora (hyps, per-hypothesis reference lists, tokenizer) with
                 sacrebleu.metrics.BLEU(tokenize=...): the corpus score and the
                 per-sentence statistics of _extract_corpus_statistics
  chrf           the same with sacrebleu.metrics.CHRF() (char_order 6, beta 2)
  ci             percentile bootstrap intervals computed independently: indices
                 from the frozen PCG32 (course/tests/_lib/pcg32.py), one
                 uniform() per index, i = min(floor(u n), n - 1); each resample
                 scored by sacreBLEU's corpus_score (exact match by plain
                 Python); quantiles by numpy (type 7, "linear")

sacreBLEU takes reference STREAMS; the fixture stores references per
hypothesis (refs[i] lists the references of hyps[i]) and transposes them here,
padding with None where a hypothesis has fewer references (sacreBLEU drops
None references, so the counts may vary per sentence).

    uv run --python 3.12 --script course/oracle/L4.5/sacrebleu_golden.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

OUT = Path("course/fixtures/L4.5/sacrebleu_golden.json")

spec = importlib.util.spec_from_file_location("pcg32", "course/tests/_lib/pcg32.py")
pcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcg)

SEED = 20261009
data_rng = np.random.default_rng(SEED)

LINES_13A = [
    "Hello, world.",
    "The value is 3.14, not 3,14.",
    "It costs $1,000.50 (about 900 EUR)!",
    "pages 10-12 and 2021-05-03",
    "a&amp;b &quot;quoted&quot; &lt;tag&gt;",
    "<skipped>kept",
    "e-mail-\nlater",
    "line one-\nline two",
    "tabs\tand  double  spaces ",
    "x/y {curly} [square] `tick` ~tilde^ _under_ |pipe|",
    "Wait... what?!",
    "Über naïve café, ça va?",
    "1.5.2023 and 1,5,2023",
    "trailing comma, ",
    "",
]

WORDS = (
    "the a cat dog sat ran on in mat park big small red quick brown fox jumps over "
    "lazy . , ! ? 3.14 1,000 well-known it's don't 2021-05-03"
).split()


def sentence(lo: int, hi: int) -> str:
    n = int(data_rng.integers(lo, hi + 1))
    return " ".join(WORDS[int(i)] for i in data_rng.integers(0, len(WORDS), n))


def perturb(s: str) -> str:
    """A noisy copy: drop, swap, or replace a few tokens."""
    toks = s.split()
    out = []
    for t in toks:
        u = data_rng.random()
        if u < 0.15:
            continue
        if u < 0.3:
            out.append(WORDS[int(data_rng.integers(0, len(WORDS)))])
        else:
            out.append(t)
    if len(out) > 2 and data_rng.random() < 0.3:
        i = int(data_rng.integers(0, len(out) - 1))
        out[i], out[i + 1] = out[i + 1], out[i]
    return " ".join(out)


def random_corpus(n: int, max_refs: int) -> dict:
    hyps, refs = [], []
    for _ in range(n):
        base = sentence(1, 14)
        k = int(data_rng.integers(1, max_refs + 1))
        refs.append([perturb(base) if j else base for j in range(k)])
        hyps.append(perturb(base))
    return {"hyps": hyps, "refs": refs}


HAND = {
    "name": "worked-example",
    "hyps": ["the cat sat on the mat"],
    "refs": [["the cat is on the mat"]],
}

SPECIAL = [
    HAND,
    {"name": "clipping", "hyps": ["the the the the"], "refs": [["the cat"]]},
    {
        "name": "two-sentences",
        "hyps": ["the cat sat on the mat", "a dog ran in the park"],
        "refs": [["the cat is on the mat"], ["the dog ran in a park"]],
    },
    {
        "name": "short-hyps-no-4grams",
        "hyps": ["the cat", "a dog ran"],
        "refs": [["the cat"], ["a dog ran"]],
    },
    {"name": "no-match", "hyps": ["xyz qqq"], "refs": [["the cat sat"]]},
    {
        "name": "empty-hyp",
        "hyps": ["", "the cat sat on the mat"],
        "refs": [["the cat"], ["the cat sat on the mat"]],
    },
    {
        "name": "long-hyp",
        "hyps": ["the cat sat on the mat and then it slept for a long long time"],
        "refs": [["the cat sat on the mat"]],
    },
    {
        "name": "closest-ref-tie-to-shorter",
        "hyps": ["a b c d e f g"],
        "refs": [["a b c d e f g h i", "a b c d e"]],
    },
    {
        "name": "closest-ref",
        "hyps": ["a b c d e f g"],
        "refs": [["a b c d e f g h i j k l", "a b c d e f x y"]],
    },
    {
        "name": "multi-ref-clip",
        "hyps": ["the the the cat cat"],
        "refs": [["the cat the", "the the cat on"]],
    },
    {
        "name": "punctuation-13a",
        "hyps": ["Hello, world. It costs $1,000.50!"],
        "refs": [["Hello , world . It costs $1,000.50 !"]],
    },
    {
        "name": "entities-and-dashes",
        "hyps": ["a&amp;b e-mail-\n", "pages 10-12"],
        "refs": [["a & b e-mail-", "a&b email"], ["pages 10 - 12"]],
    },
    {"name": "trailing-space", "hyps": ["the cat sat   "], "refs": [["the cat sat"]]},
    {
        "name": "case-matters",
        "hyps": ["The Cat Sat On The Mat"],
        "refs": [["the cat sat on the mat"]],
    },
    {
        "name": "variable-refs",
        "hyps": ["the cat sat", "a dog ran in the park today", "big red fox"],
        "refs": [
            ["the cat sat down"],
            ["a dog ran in the park", "the dog ran in a park today", "dogs run"],
            ["the big red fox", "big red foxes"],
        ],
    },
]

CHRF_SPECIAL = [
    {"name": "worked-example", "hyps": ["cat"], "refs": [["cats"]]},
    {"name": "white-space-ignored", "hyps": ["the cat sat"], "refs": [["thecat  sat"]]},
    {"name": "inflection", "hyps": ["the cats sitting"], "refs": [["the cat sat"]]},
    {"name": "short-ref", "hyps": ["abcdefgh", "abc"], "refs": [["abc"], ["abcdefgh"]]},
    {"name": "empty-hyp", "hyps": ["", "abc"], "refs": [["abc"], ["abc"]]},
    {
        "name": "best-ref",
        "hyps": ["the quick fox"],
        "refs": [["a slow dog", "the quick fox!"]],
    },
    # "aa" and "acab" give "abca" the same sentence chrF (41.67) with different
    # statistics: the first reference must win, and the pooled score shows it.
    {
        "name": "best-ref-tie-first",
        "hyps": ["abca", "abc"],
        "refs": [["aa", "acab"], ["ab"]],
    },
    {"name": "no-match", "hyps": ["xyz"], "refs": [["abc"]]},
    {
        "name": "unicode",
        "hyps": ["café crème"],
        "refs": [["cafe creme", "café crèmes"]],
    },
    {
        "name": "variable-refs",
        "hyps": ["the cat sat", "a dog ran in the park today"],
        "refs": [
            ["the cat sat down"],
            ["a dog ran in the park", "the dog ran in a park today", "dogs run"],
        ],
    },
]


def streams(refs: list[list[str]]) -> list[list[str | None]]:
    k = max(len(r) for r in refs)
    return [[r[j] if j < len(r) else None for r in refs] for j in range(k)]


def bleu_case(case: dict, tokenize: str) -> dict:
    m = BLEU(tokenize=tokenize)
    st = streams(case["refs"])
    stats = m._extract_corpus_statistics(case["hyps"], st)
    score = m.corpus_score(case["hyps"], st).score
    return dict(
        case,
        tokenize=tokenize,
        score=float(score),
        stats=[[int(v) for v in s] for s in stats],
    )


def chrf_case(case: dict) -> dict:
    m = CHRF()
    st = streams(case["refs"])
    stats = m._extract_corpus_statistics(case["hyps"], st)
    score = m.corpus_score(case["hyps"], st).score
    return dict(case, score=float(score), stats=[[int(v) for v in s] for s in stats])


def em(hyps, refs) -> float:
    return sum(h.strip() == r.strip() for h, r in zip(hyps, refs)) / len(hyps)


def bootstrap(
    metric: str, hyps, refs, n_boot: int, alpha: float, seed: int
) -> list[float]:
    g = pcg.PCG32(seed=seed)
    n = len(hyps)

    def score(hs, rs) -> float:
        if metric == "exact_match":
            return float(em(hs, rs))
        m = BLEU() if metric == "bleu" else CHRF()
        return float(m.corpus_score(hs, streams(rs)).score)

    thetas = []
    for _ in range(n_boot):
        idx = [min(int(math.floor(g.uniform() * n)), n - 1) for _ in range(n)]
        thetas.append(score([hyps[i] for i in idx], [refs[i] for i in idx]))
    t = np.asarray(thetas, dtype=np.float64)
    return [
        score(hyps, refs),
        float(np.quantile(t, alpha / 2, method="linear")),
        float(np.quantile(t, 1 - alpha / 2, method="linear")),
    ]


def main() -> None:
    tok = Tokenizer13a()
    out: dict = {
        "__meta__": {
            "generator": "course/oracle/L4.5/sacrebleu_golden.py",
            "sacrebleu": sacrebleu.__version__,
            "numpy": np.__version__,
            "seed": SEED,
            "bleu": "BLEU(tokenize=...) defaults: smooth exp, 4-grams, no lowercase",
            "chrf": "CHRF() defaults: char_order 6, word_order 0, beta 2, whitespace removed",
        }
    }
    out["tokenize_13a"] = [[line, tok(line)] for line in LINES_13A]

    bleu = [bleu_case(c, "13a") for c in SPECIAL]
    bleu += [
        bleu_case(dict(c, name="none-" + c["name"]), "none")
        for c in SPECIAL[:3] + SPECIAL[10:12]
    ]
    for i in range(6):
        c = random_corpus(int(data_rng.integers(8, 31)), 3)
        bleu.append(bleu_case(dict(c, name=f"random-{i}"), "13a"))
    out["bleu"] = bleu

    chrf = [chrf_case(c) for c in CHRF_SPECIAL]
    for i in range(6):
        c = random_corpus(int(data_rng.integers(8, 31)), 3)
        chrf.append(chrf_case(dict(c, name=f"random-{i}")))
    out["chrf"] = chrf

    ci = []
    corpus = random_corpus(16, 2)
    em_hyps = [f"2021-0{1 + i % 9}-{10 + i:02d}" for i in range(20)]
    em_refs = [h if i % 3 else h.replace("-", "/") for i, h in enumerate(em_hyps)]
    for metric, hyps, refs in [
        ("exact_match", em_hyps, em_refs),
        ("bleu", corpus["hyps"], corpus["refs"]),
        ("chrf", corpus["hyps"], corpus["refs"]),
    ]:
        for n_boot, alpha, seed in [(200, 0.05, 0), (150, 0.1, 7)]:
            ci.append(
                {
                    "metric": metric,
                    "hyps": hyps,
                    "refs": refs,
                    "n_boot": n_boot,
                    "alpha": alpha,
                    "seed": seed,
                    "want": bootstrap(metric, hyps, refs, n_boot, alpha, seed),
                }
            )
    out["ci"] = ci

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode()
    OUT.write_bytes(data)
    print(
        "\t".join(
            [
                str(OUT),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/L4.5/sacrebleu_golden.py",
                f"sacrebleu=={sacrebleu.__version__} numpy=={np.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
