# RAG index

<!-- modules: ag.06 (ingest writes it), ag.07 (retrieval reads it), ag.08 (citations resolve against docs.jsonl)
     conformance: formats/ -->

One index is a directory `rag/<index_id>/` under `/artifacts`. Its four files are written by `{ctl} rag ingest` and read by retrieval; the agent's citations name `chunk_id`s that must resolve in `docs.jsonl`. All integers little-endian.

## `meta.json`

```json
{"index_id": "docs", "created_at": "2026-10-09T12:00:00Z", "n_chunks": 812, "dim": 576,
 "embedding_model": "smol-135m", "tokenizer": "smol-135m",
 "chunker": {"max_tokens": 256, "overlap": 32}, "bm25": {"k1": 1.2, "b": 0.75}}
```

`dim` is the length of every embedding; `embedding_model` is the model whose `/v1/embeddings` produced them (queries must use the same one); `tokenizer` is the model whose `/v1/tokenize` counted chunk tokens.

## `docs.jsonl`

One chunk per line, in chunk order (line `i` is chunk index `i`, the row of `vectors.f32` and the doc number in `bm25.idx`):

```json
{"chunk_id": "docs/c4/containers.d2#3", "doc_id": "docs/c4/containers.d2", "source": "file", "uri": "docs/c4/containers.d2", "text": "...", "n_tokens": 241, "fingerprint": "<sha256 of text>"}
```

`chunk_id` is `<doc_id>#<k>` with `k` the chunk's position in its document. Re-ingesting a document whose fingerprints are all unchanged writes nothing (ag.06).

## `vectors.f32`

`n_chunks x dim` float32, row-major, no header: the size is exactly `n_chunks * dim * 4` bytes. Each row is L2-normalized, so cosine similarity is the dot product.

## `bm25.idx`

```
offset  size   field
0       4      magic "TLBM"
4       4      u32 version = 1
8       4      u32 n_docs           (= n_chunks)
12      4      u32 n_terms
16      8      f64 avgdl            mean document length in terms
24      4*n    u32 doc_len[n_docs]
then n_terms entries, sorted by term bytes:
        2      u16 term_len, then term_len bytes of UTF-8
        4      u32 df               documents containing the term
        4      u32 postings_bytes
        ...    postings: df pairs (doc gap, tf) as unsigned LEB128 varints; the first gap is the doc number itself
```

Terms are the maximal runs of `[\p{L}\p{N}_]` (Unicode letters, digits, and `_`, in Go RE2 syntax) of the text, lowercased with `strings.ToLower`. Scores use `idf = ln(1 + (N - df + 0.5) / (df + 0.5))` and the usual `tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`; ag.07's course test compares scores with the reference to 1e-9.

IVF centroids are not stored: ag.07 builds them at load with k-means++, seeding PCG32 (spec/pcg32.md) with `fnv1a64(index_id)`, so the file set stays the same for flat and IVF retrieval.
