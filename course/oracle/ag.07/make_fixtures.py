"""Fixtures for ag.07 (retrieval), computed independently of the Go reference.

    uv run python course/oracle/ag.07/make_fixtures.py

Writes course/fixtures/ag.07/:

  index/meta.json, index/docs.jsonl, index/vectors.f32, index/bm25.idx
      a 40-chunk RAG index in the layout of contracts/formats/rag-index.md;
      vectors come from a feature-hashing embedder (below) so the Go test can
      embed queries the same way without a model
  expected.json
      BM25 scores of every chunk for each query (to compare at 1e-9), the
      lexical, flat, hybrid (RRF), filtered, and MMR rankings, and k-means++
      picks with the frozen PCG32 (course/tests/_lib/pcg32.py)

Everything here is written from the contract's text: terms are maximal runs
of Unicode letters, digits, and underscore (categories L*, N*, and "_"),
lowercased; idf = ln(1 + (N - df + 0.5) / (df + 0.5)); tf part tf * (k1 + 1) /
(tf + k1 * (1 - b + b * dl / avgdl)); a repeated query term counts each time;
ties go to the lower chunk number. Sums run in the same order as the
contract states (query term order, then postings order), so the Go scores
must agree to 1e-9 (in practice they agree bit for bit).

The corpus is written for this file (no third-party text).
"""

from __future__ import annotations

import json
import math
import struct
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve()
COURSE = HERE.parents[2]
sys.path.insert(0, str(COURSE / "tests" / "_lib"))
from pcg32 import PCG32  # noqa: E402  (the frozen generator)

OUT = COURSE / "fixtures" / "ag.07"
K1, B = 1.2, 0.75
DIM = 16
INDEX_ID = "docs"

DOCS = [
    (
        "docs/kv-cache.md",
        "file",
        [
            "The KV cache keeps the keys and values of every past token so decoding does not recompute attention over the prompt.",
            "Eviction frees KV blocks when the pool is full: the least recently used block whose reference count is zero goes first.",
            "A prefix cache reuses KV blocks across requests that share a prompt prefix; block hashes chain FNV-1a over token ids.",
        ],
    ),
    (
        "docs/gateway.md",
        "file",
        [
            "The gateway checks the API key, applies rate limits per tenant, and proxies server-sent events without buffering.",
            "Routing uses consistent hashing with bounded loads so one engine replica never takes more than its share of requests.",
            "A 429 answer carries Retry-After; clients back off instead of hammering the gateway.",
        ],
    ),
    (
        "docs/durable.md",
        "file",
        [
            "The durable engine records every workflow step in a write-ahead log and replays the history after a crash.",
            "Activities are retried with backoff; an idempotency key makes a retried write happen exactly once.",
            "Timers live in a min-heap; a restarted server fires each due timer exactly once, in order.",
        ],
    ),
    (
        "docs/tokenizer.md",
        "file",
        [
            "Byte-pair encoding merges the most frequent adjacent pair until the vocabulary reaches its target size.",
            "The Rust tokenizer must produce the same ids as the Python one; tl_tok_encode is checked against golden ids.",
        ],
    ),
    (
        "docs/sampling.md",
        "file",
        [
            "Sampling applies penalties, temperature, top-k, top-p, and min-p, then one uniform draw picks a token by inverse CDF.",
            "Temperature zero is greedy decoding; ties go to the lowest token id so Python and Rust agree.",
        ],
    ),
    (
        "docs/kernels.md",
        "file",
        [
            "The matmul kernel tiles the loops so each tile fits in cache; a cache miss costs far more than a multiply.",
            "Flash attention computes softmax online, one block of keys at a time, and never materializes the score matrix.",
            "Int4 quantization stores two weights per byte with one fp16 scale per group of 32 columns.",
        ],
    ),
    (
        "docs/eval.md",
        "file",
        [
            "An eval suite runs every case against a subject and scores the output; an errored scorer is counted, never averaged as zero.",
            "Bootstrap confidence intervals resample cases with replacement; a paired design compares two subjects on the same cases.",
        ],
    ),
    (
        "docs/agent.md",
        "file",
        [
            "The agent loop calls the model, runs the tool calls it asks for, and stops at a final answer or the iteration limit.",
            "Tool arguments are validated against the JSON Schema before the tool runs; invalid arguments go back to the model as an error.",
            "A gate decides before dispatch whether a tool call may run; writes need approval.",
        ],
    ),
    (
        "docs/retrieval.md",
        "file",
        [
            "BM25 scores lexical matches with saturating term frequency and length normalization; k1 and b are its two knobs.",
            "Dense retrieval embeds the query and returns the nearest chunks by cosine similarity.",
            "Reciprocal rank fusion adds 1 / (k + rank) over the lists, so it needs ranks, not comparable scores.",
            "Maximal marginal relevance trades relevance against redundancy when it picks the final chunks.",
        ],
    ),
    (
        "web/ivf.html",
        "web",
        [
            "An IVF index clusters the vectors with k-means and scans only the nprobe nearest lists at query time.",
            "k-means++ seeding picks each new centroid with probability proportional to its squared distance from the nearest one already picked.",
            "Recall at 10 measures how many of the exact top 10 neighbours the approximate search returned.",
        ],
    ),
    (
        "web/caching.html",
        "web",
        [
            "A response cache keyed by the request hash returns a stored completion; the eviction policy is LRU with a size cap.",
            "Café-style write-back caches defer writes; überall in distributed systems, a cache is a replica with weaker guarantees.",
        ],
    ),
    (
        "web/observability.html",
        "web",
        [
            "Traces connect the gateway span to the engine span through the traceparent header.",
            "SLO burn rate alerts fire when the error budget is consumed faster than the window allows.",
            "Time to first token, TTFT, and inter-token latency, ITL, are the two latencies users feel.",
        ],
    ),
    (
        "docs/release.md",
        "file",
        [
            "ModelRelease gates a model on its eval thresholds, rolls out a canary, and rolls back when the burn rate is too high.",
            "A model card states intended use, evaluation results with confidence intervals, and known limitations.",
        ],
    ),
    (
        "docs/data.md",
        "file",
        [
            "The corpus pipeline fetches, filters, deduplicates with MinHash, removes PII, and tokenizes into token streams.",
            "Decontamination drops documents that share a long n-gram with a protected eval set.",
        ],
    ),
    (
        "docs/numbers.md",
        "file",
        [
            "Port 30080 serves the gateway in kind; port 9464 exposes /metrics; block_tokens is 16 by default.",
            "cache cache cache: a chunk that repeats a term saturates, it does not win by repetition alone.",
        ],
    ),
]

QUERIES = [
    "kv cache eviction",
    "cache",
    "exactly once retries with an idempotency key",
    "how does reciprocal rank fusion work",
    "k-means++ seeding for an IVF index",
    "TTFT and ITL latency",
    "café überall",
    "tl_tok_encode golden ids",
    "port 30080",
    "cache cache",
    "zebra quantum",
]


def is_term_char(c: str) -> bool:
    return c == "_" or unicodedata.category(c)[0] in ("L", "N")


def terms(text: str) -> list[str]:
    out, cur = [], []
    for c in text:
        if is_term_char(c):
            cur.append(c)
        elif cur:
            out.append("".join(cur).lower())
            cur = []
    if cur:
        out.append("".join(cur).lower())
    return out


def fnv1a64(b: bytes) -> int:
    h = 0xCBF29CE484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001B3) & ((1 << 64) - 1)
    return h


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def hash_embed_raw(text: str) -> list[float]:
    """Feature hashing: each term adds +1 or -1 to coordinate h % DIM, sign
    from the top bit of h = fnv1a64(term bytes). Integers, exact in float32."""
    v = [0.0] * DIM
    for t in terms(text):
        h = fnv1a64(t.encode())
        v[h % DIM] += -1.0 if h >> 63 else 1.0
    return v


def normalize(v: list[float]) -> list[float]:
    """The contract's Normalize: float64 sum of squares in order, scale by
    1/sqrt, round each coordinate to float32."""
    s = 0.0
    for x in v:
        s += x * x
    if s == 0.0:
        return [0.0] * len(v)
    inv = 1.0 / math.sqrt(s)
    return [f32(x * inv) for x in v]


def dot(a, b) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def dist2(a, b) -> float:
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def topk(scored: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
    s = sorted(scored, key=lambda p: (-p[1], p[0]))
    return s[:k] if k > 0 else s


class BM25:
    def __init__(self, texts: list[str]):
        self.doc_len = []
        self.post: dict[str, list[tuple[int, int]]] = {}
        for d, text in enumerate(texts):
            ts = terms(text)
            self.doc_len.append(len(ts))
            tf: dict[str, int] = {}
            for t in ts:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                self.post.setdefault(t, []).append((d, c))
        self.n = len(texts)
        self.avgdl = sum(self.doc_len) / self.n

    def idf(self, t: str) -> float:
        df = len(self.post.get(t, []))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, q: str) -> list[float]:
        s = [0.0] * self.n
        for t in terms(q):
            pl = self.post.get(t)
            if not pl:
                continue
            idf = self.idf(t)
            for d, tf in pl:
                dl = self.doc_len[d]
                s[d] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / self.avgdl))
        return s

    def search(self, q: str, k: int, keep=lambda d: True) -> list[tuple[int, float]]:
        matched = set()
        for t in terms(q):
            matched |= {d for d, _ in self.post.get(t, [])}
        s = self.scores(q)
        return topk([(d, s[d]) for d in sorted(matched) if keep(d)], k)

    def to_bytes(self) -> bytes:
        def uvarint(v: int) -> bytes:
            out = bytearray()
            while v >= 0x80:
                out.append((v & 0x7F) | 0x80)
                v >>= 7
            out.append(v)
            return bytes(out)

        b = bytearray(b"TLBM")
        b += struct.pack("<IIId", 1, self.n, len(self.post), self.avgdl)
        for dl in self.doc_len:
            b += struct.pack("<I", dl)
        for t in sorted(self.post, key=lambda s: s.encode()):
            pl = self.post[t]
            post = bytearray()
            prev = 0
            for d, tf in pl:
                post += uvarint(d - prev) + uvarint(tf)
                prev = d
            tb = t.encode()
            b += (
                struct.pack("<H", len(tb))
                + tb
                + struct.pack("<II", len(pl), len(post))
                + post
            )
        return bytes(b)


def rrf(lists: list[list[int]], k: float, n: int) -> list[tuple[int, float]]:
    score: dict[int, float] = {}
    for lst in lists:
        for r, d in enumerate(lst):
            score[d] = score.get(d, 0.0) + 1 / (k + (r + 1))
    return topk(list(score.items()), n)


def mmr(q, cands: list[int], vecs, lam: float, k: int) -> list[tuple[int, float]]:
    rel = [dot(q, vecs[c]) for c in cands]
    maxsim = [0.0] * len(cands)
    used = [False] * len(cands)
    out = []
    while len(out) < min(k, len(cands)):
        best, bv = -1, 0.0
        for i in range(len(cands)):
            if used[i]:
                continue
            v = lam * rel[i] - (1 - lam) * maxsim[i]
            if best < 0 or v > bv:
                best, bv = i, v
        used[best] = True
        out.append((cands[best], bv))
        pv = vecs[cands[best]]
        for i in range(len(cands)):
            if used[i]:
                continue
            s = dot(vecs[cands[i]], pv)
            if len(out) == 1 or s > maxsim[i]:
                maxsim[i] = s
    return out


def kmeanspp(vecs, k: int, r: PCG32) -> list[int]:
    n = len(vecs)
    picked = [r.below(n)]
    d = [dist2(v, vecs[picked[0]]) for v in vecs]
    while len(picked) < k:
        total = 0.0
        for x in d:
            total += x
        if total == 0.0:
            nxt = next(i for i in range(n) if i not in picked)
        else:
            u = r.uniform() * total
            run, nxt = 0.0, -1
            for i, x in enumerate(d):
                run += x
                if run > u:
                    nxt = i
                    break
            if nxt < 0:
                nxt = max(i for i in range(n) if d[i] > 0)
        picked.append(nxt)
        for i, v in enumerate(vecs):
            x = dist2(v, vecs[nxt])
            if x < d[i]:
                d[i] = x
    return picked


def main() -> None:
    chunks = []
    for doc_id, source, texts in DOCS:
        for k, text in enumerate(texts):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}#{k}",
                    "doc_id": doc_id,
                    "source": source,
                    "uri": doc_id,
                    "text": text,
                    "n_tokens": len(text.encode()) // 4,
                    "fingerprint": __import__("hashlib")
                    .sha256(text.encode())
                    .hexdigest(),
                }
            )
    texts = [c["text"] for c in chunks]
    bm = BM25(texts)
    vecs = [normalize(hash_embed_raw(t)) for t in texts]
    n = len(chunks)

    idx = OUT / "index"
    idx.mkdir(parents=True, exist_ok=True)
    meta = {
        "index_id": INDEX_ID,
        "created_at": "2026-10-09T12:00:00Z",
        "n_chunks": n,
        "dim": DIM,
        "embedding_model": "hash16",
        "tokenizer": "bytes",
        "chunker": {"max_tokens": 256, "overlap": 32},
        "bm25": {"k1": K1, "b": B},
    }
    (idx / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (idx / "docs.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    )
    (idx / "vectors.f32").write_bytes(
        b"".join(struct.pack(f"<{DIM}f", *v) for v in vecs)
    )
    (idx / "bm25.idx").write_bytes(bm.to_bytes())

    queries = []
    cand = 50
    for q in QUERIES:
        qv = normalize(hash_embed_raw(q))
        lex = bm.search(q, cand)
        flat = topk([(i, dot(qv, vecs[i])) for i in range(n)], cand)
        fused = rrf([[d for d, _ in lex], [d for d, _ in flat]], 60, cand)
        web = {i for i, c in enumerate(chunks) if c["source"] == "web"}
        lex_f = bm.search(q, cand, lambda d: d in web)
        flat_f = topk([(i, dot(qv, vecs[i])) for i in range(n) if i in web], cand)
        fused_f = rrf([[d for d, _ in lex_f], [d for d, _ in flat_f]], 60, cand)
        mm = mmr(qv, [d for d, _ in fused], vecs, 0.5, 4)
        queries.append(
            {
                "q": q,
                "terms": terms(q),
                "bm25": bm.scores(q),
                "lexical_top": lex[:10],
                "flat_top": flat[:10],
                "hybrid5": [[chunks[d]["chunk_id"], s] for d, s in fused[:5]],
                "hybrid5_web": [[chunks[d]["chunk_id"], s] for d, s in fused_f[:5]],
                "mmr4": [[chunks[d]["chunk_id"], s] for d, s in mm],
            }
        )
    expected = {
        "k1": K1,
        "b": B,
        "dim": DIM,
        "avgdl": bm.avgdl,
        "n_terms": len(bm.post),
        "queries": queries,
        "kmeanspp": [
            {
                "seed": fnv1a64(INDEX_ID.encode()),
                "k": 4,
                "picks": kmeanspp(vecs, 4, PCG32(fnv1a64(INDEX_ID.encode()))),
            },
            {"seed": 7, "k": 6, "picks": kmeanspp(vecs, 6, PCG32(7))},
        ],
    }
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(f"wrote {n} chunks, {len(bm.post)} terms, {len(QUERIES)} queries to {OUT}")


if __name__ == "__main__":
    main()
