# Token streams: llm.c `.bin` files and their manifest

<!-- modules: data.07 (writer, python/corpus/tokenize.py), L0.6 (reader, TokenStream), C1 (training data)
     conformance: formats/ -->

The training loader memory-maps a flat stream of token ids (D17). The layout is the `.bin` format of Karpathy's llm.c, so llm.c tools read the course's GPT-2-vocabulary files unchanged.

Path: `/artifacts/tokens/<tokenizer_id>/<dataset>/<split>-<nnnnn>.bin`, for example `tokens/smollm2/tinystories/train-00000.bin`, with `split` in `{train, val}`.

## Layout

All integers little-endian.

```
offset  size                 field
0       1024                 header: 256 x int32
1024    n_tokens * w         the token ids, w = 2 (uint16) for version 1, 4 (uint32) for version 2
```

| Header int | Value |
|---|---|
| `[0]` | magic `20240520` |
| `[1]` | version: `1` (uint16 ids, so `vocab_size <= 65536`) or `2` (uint32 ids) |
| `[2]` | `n_tokens`, the number of ids that follow (at most `2^31 - 1` per file) |
| `[3]` | `vocab_size` of the tokenizer (llm.c leaves it 0; the course writes it and readers check it against the model) |
| `[4..255]` | 0 |

The file size is exactly `1024 + n_tokens * w`. A reader rejects a wrong magic, an unknown version, a size that disagrees with `n_tokens`, or an id `>= vocab_size` when `[3]` is non-zero. Version 2 is the course's extension for vocabularies above 65536; llm.c's own Llama 3 files use a different magic (`20240801`, version 7) and are not part of this contract.

Documents are concatenated in shard order. When the tokenizer defines a document separator (the `eos_token_id` of its `generation_config.json`, for example `<|endoftext|>`), each document is **preceded** by it, as llm.c does; the byte tokenizer has none. A file holds whole documents: the writer starts a new file (`-00001`, ...) before a document that would push the file past 100,000,000 tokens. Train and val come from the shard's `split` column (`formats/corpus-shard.md`), so no document is in both.

## Manifest

`tokens/<tokenizer_id>/<dataset>/_MANIFEST.json`, schema [`tokens-manifest.schema.json`](tokens-manifest.schema.json):

```json
{"tokenizer_id": "smollm2", "tokenizer_sha256": "<sha256 of tokenizer.json>", "vocab_size": 49152,
 "doc_sep_id": 0, "dataset": "tinystories", "version": "v1", "corpus_manifest_sha256": "<sha256 of the corpus _MANIFEST.json>",
 "files": [{"name": "train-00000.bin", "split": "train", "n_tokens": 1234567, "n_docs": 2100, "sha256": "<hex>"}]}
```

`n_tokens` summed over a split equals the tokens the writer produced for that split (token conservation, data.07), and the manifest is byte-identical across two runs (deterministic output).

## Worked example

The ids `[1, 2, 3]`, version 1, `vocab_size` 256 (the byte tokenizer, so no separator): 1030 bytes.

```
offset 0     88 d8 34 01     20240520 = 0x0134D888
offset 4     01 00 00 00     version 1
offset 8     03 00 00 00     n_tokens 3
offset 12    00 01 00 00     vocab_size 256
offset 16    00 ... 00       1008 zero bytes up to offset 1024
offset 1024  01 00 02 00 03 00
```
