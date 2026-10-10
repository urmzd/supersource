# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for course/fixtures/L8.1/sampler_golden.json.

An independent, stdlib-only transcription of spec/sampling.md (no numpy,
no course code): float64 lists, math.exp and math.log, sums as explicit
left-to-right loops, orderings by (logit desc, id asc), PCG32 and SplitMix64
from spec/pcg32.md written out again here. Each case fixes float32 logits
(stored as the exact decimal of their float32 value), sampling parameters,
a prompt, and a seed; the expected output is the 12 ids and logprobs that a
request draws when it samples 12 times from those same logits with its
history growing by each sampled id (so the penalties change step to step).
This is also the shared fixture of `parity/sampler` (L10.1 must emit the
same ids and logprobs).

    uv run --script course/oracle/L8.1/sampler_golden.py
Run from the repo root; it prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "course" / "fixtures" / "L8.1" / "sampler_golden.json"
M64, M32 = (1 << 64) - 1, (1 << 32) - 1
GOLDEN = 0x9E3779B97F4A7C15


def mix64(x: int) -> int:
    z = x & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


class Pcg:
    def __init__(self, seed: int, seq: int = 54):
        self.inc = ((seq << 1) | 1) & M64
        self.state = 0
        self.u32()
        self.state = (self.state + seed) & M64
        self.u32()

    def u32(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & M64
        xs = (((old >> 18) ^ old) >> 27) & M32
        rot = old >> 59
        return ((xs >> rot) | (xs << ((32 - rot) & 31))) & M32

    def uniform(self) -> float:
        a = self.u32() >> 5
        b = self.u32() >> 6
        return (a * 67108864 + b) * 2.0**-53


def sample_stream(seed: int) -> Pcg:
    p = 4  # PURPOSES["sample"]
    return Pcg(mix64((seed + p * GOLDEN) & M64), p)


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def seqsum(xs):
    s = 0.0
    for v in xs:
        s += v
    return s


def softmax_on(l, ids):
    ids = sorted(ids)
    m = max(l[i] for i in ids)
    e = [math.exp(l[i] - m) for i in ids]
    z = seqsum(e)
    return {i: v / z for i, v in zip(ids, e)}


def step(x, prm, prompt, out, rng):
    l = [float(v) for v in x]  # step 1
    r = prm["repetition_penalty"]
    if r != 1.0:
        for i in sorted(set(prompt) | set(out)):
            l[i] = l[i] / r if l[i] > 0 else l[i] * r
    if prm["presence_penalty"] != 0.0 or prm["frequency_penalty"] != 0.0:
        for i, c in sorted(Counter(out).items()):
            l[i] = l[i] - prm["frequency_penalty"] * c - prm["presence_penalty"]
    m = max(l)
    z = seqsum([math.exp(v - m) for v in l])
    lp = [v - m - math.log(z) for v in l]
    T = prm["temperature"]
    if T == 0.0:
        best = max(range(len(l)), key=lambda i: (l[i], -i))
        return best, lp[best]
    t = [v / T for v in l]
    order = sorted(
        (i for i in range(len(t)) if t[i] != -math.inf), key=lambda i: (-t[i], i)
    )
    k = prm["top_k"]
    if 0 < k < len(order):
        order = order[:k]
    if prm["top_p"] < 1.0:
        q = softmax_on(t, order)
        s, cut = 0.0, len(order)
        for n, i in enumerate(order):
            s += q[i]
            if s >= prm["top_p"]:
                cut = n + 1
                break
        order = order[:cut]
    if prm["min_p"] > 0.0:
        q = softmax_on(t, order)
        top = max(q.values())
        order = [i for i in order if q[i] >= prm["min_p"] * top]
    q = softmax_on(t, order)
    u = rng.uniform()
    c, last = 0.0, None
    for i in sorted(order):
        c += q[i]
        last = i
        if u < c:
            return i, lp[i]
    return last, lp[last]


BASE = dict(
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
)
CASES = [
    ("plain", {}),
    ("temperature_0.7", {"temperature": 0.7}),
    ("hot", {"temperature": 1.8}),
    ("top_k_5", {"top_k": 5}),
    ("top_p_0.9", {"top_p": 0.9}),
    ("min_p_0.1", {"min_p": 0.1}),
    ("all_filters", {"temperature": 0.8, "top_k": 12, "top_p": 0.85, "min_p": 0.05}),
    ("repetition_1.3", {"repetition_penalty": 1.3}),
    ("presence_frequency", {"presence_penalty": 0.6, "frequency_penalty": 0.4}),
    (
        "penalties_and_filters",
        {
            "temperature": 0.9,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
            "frequency_penalty": 0.3,
        },
    ),
    (
        "greedy_penalized",
        {"temperature": 0.0, "repetition_penalty": 1.5, "presence_penalty": 0.5},
    ),
    ("masked", {"top_k": 40}),
]


def main() -> None:
    gen = Pcg(424242, 7)
    cases = []
    for n, (name, over) in enumerate(CASES):
        V = 48
        x = [f32((gen.uniform() * 2 - 1) * 4.0) for _ in range(V)]
        x[n % V] = x[(n + 7) % V]  # an exact tie somewhere
        if name == "masked":
            for i in range(0, V, 3):
                x[i] = -math.inf
        prompt = [int(gen.uniform() * V) for _ in range(6)]
        prm = dict(BASE, **over)
        seed = n * 1000 + 17
        rng = sample_stream(seed)
        out, ids, lps = [], [], []
        for _ in range(12):
            tok, lp = step(x, prm, prompt, out, rng)
            out.append(tok)
            ids.append(tok)
            lps.append(lp)
        cases.append(
            {
                "name": name,
                "logits": [repr(v) for v in x],
                "params": prm,
                "prompt": prompt,
                "seed": seed,
                "ids": ids,
                "logprobs": lps,
            }
        )
    doc = {
        "__meta__": {
            "generator": "course/oracle/L8.1/sampler_golden.py",
            "spec": "spec/sampling.md",
            "steps": 12,
            "note": "logits are float32 values written as Python reprs ('-inf' for masks)",
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT.relative_to(ROOT)),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/L8.1/sampler_golden.py",
                "python stdlib",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
