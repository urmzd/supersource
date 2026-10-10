# Corpus files: raw documents, clean shards, the shard manifest

<!-- modules: data.01 (raw), data.02 to data.05 (stages), data.06 (shards and manifest), data.07 (reads shards), data.08 (ledger), data.09 (CorpusBuild)
     conformance: formats/ -->

The corpus pipeline (`python/corpus/`) turns fetched sources into clean, deduplicated, PII-scrubbed Parquet shards that the tokenizer step (data.07) packs into [token streams](tokens-bin.md). Paths are relative to `/artifacts` (`[paths].artifacts`). The pipeline's inputs are fixed by a corpus config ([`corpus-config.schema.json`](corpus-config.schema.json)) and its sources by the [data ledger](ledger.schema.json).

## Raw documents (data.01)

`corpus/raw/<source_id>/<yyyymmdd>/part-<nnnnn>.jsonl.zst`: zstd-compressed JSON Lines, one document per line, UTF-8:

```json
{"url": "https://example.org/story/1", "fetched_at": "2026-10-09T12:00:00Z", "license_spdx": "CDLA-Sharing-1.0", "text": "Once upon a time..."}
```

| Field | Type | Meaning |
|---|---|---|
| `url` | string | where the document came from (for a single-file source, the file URL plus `#<line>`) |
| `fetched_at` | RFC 3339 UTC | fetch time |
| `license_spdx` | string | SPDX identifier from the ledger row of the source |
| `text` | string | the extracted text, before normalization |

`<yyyymmdd>` is the fetch date. A rerun of fetch downloads nothing whose checksum already matches the ledger.

## Clean shards (data.06)

`corpus/<dataset>/<version>/shard-<iiiii>-of-<nnnnn>.parquet`, for example `shard-00000-of-00064.parquet`. Parquet with zstd compression, at most 64 MiB per row group, one row per kept document, columns in this order:

| Column | Arrow type | Meaning |
|---|---|---|
| `id` | `string` | `<source_id>:<k>`, where `k` is the document's 0-based position in its source's raw files read in path order |
| `text` | `string` | the final text: normalized (Unicode NFC), filtered, PII scrubbed |
| `source_id` | `string` | the ledger's source id |
| `url` | `string` | from the raw document |
| `license_spdx` | `string` | from the raw document |
| `lang` | `string` | ISO 639-1 code from the language filter (`und` when undetermined) |
| `n_chars` | `int32` | Unicode code points in `text` |
| `sha256` | `fixed_size_binary(32)` | SHA-256 of the UTF-8 bytes of `text` |
| `minhash_cluster` | `int64` | the near-duplicate cluster id (union-find root, data.04): the smallest row index of the cluster in `id` order; equal to the row's own index when it has no near duplicate |
| `pii_redactions` | `int32` | placeholders the PII scrub inserted (data.05) |
| `split` | `string` | `train` or `val` |

**Order and determinism.** Rows are sorted by `id` (byte order) across the whole dataset and cut into shards of `shard_rows` rows (the corpus config); `nnnnn` is the shard count. The output (every shard and the manifest) is byte-identical across runs and across worker counts.

**Split.** A document is `val` when the first 8 bytes of its `sha256`, read as a big-endian unsigned integer, modulo 1000 are below the config's `val_permille`; otherwise `train`. Splitting by content hash keeps exact duplicates on one side.

**Placeholders.** The PII scrub replaces each match with a typed placeholder: `<EMAIL>`, `<PHONE>`, `<CARD>`, `<IP>`, `<KEY>`.

## Shard manifest

`corpus/<dataset>/<version>/_MANIFEST.json`, schema [`corpus-manifest.schema.json`](corpus-manifest.schema.json). It records what was kept, what was dropped and why, and which ledger and config produced it:

```json
{"dataset": "tinystories", "version": "v1", "n_docs": 2119719, "n_shards": 64,
 "shards": [{"name": "shard-00000-of-00064.parquet", "sha256": "<hex>", "rows": 33121}],
 "filters": {"lang": 1203, "gopher": 8812, "repetition": 77, "ppl": 0},
 "dedup": {"exact_dropped": 5120, "near_dropped": 18044, "decontaminated": 12,
           "jaccard_threshold": 0.8, "num_perm": 128, "bands": 16, "ngram": 13},
 "pii": {"emails": 3, "phones": 0, "cards": 0, "ips": 1, "keys": 0},
 "ledger_ref": "corpus/LEDGER.jsonl", "config_sha256": "<hex>"}
```

`n_docs` equals the sum of `rows`; `filters` counts documents each quality filter dropped; `dedup.decontaminated` counts documents dropped for sharing an `ngram`-token n-gram with a protected eval or validation set (data.04).
