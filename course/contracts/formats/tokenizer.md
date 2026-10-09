# Tokenizers

A tokenizer turns text into a list of integer ids and back. A model directory names its tokenizer in [`config.json`](config.schema.json) with `tl_tokenizer`:

| `tl_tokenizer` | File in the model directory | Introduced by |
|---|---|---|
| `bytes` | none | the tracer (L0.0, L10.0): the identity byte tokenizer below |
| `file` | `tokenizer.json` (HF subset), or `tinyllm_char.json` for the char tokenizer | L1.1 to L1.6 |

## The byte tokenizer (`bytes`)

The vocabulary is the 256 byte values. The id of a byte is its value, so `vocab_size` is exactly 256 and there are no special tokens: no BOS, no EOS, no padding.

- **Encode.** `encode(text)` is the list of the UTF-8 bytes of `text`, each as an integer in `0..=255`.
- **Decode.** `decode(ids)` is those bytes decoded as UTF-8 **with replacement**: each maximal ill-formed subsequence becomes one U+FFFD (the Unicode "maximal subpart" practice). Python `bytes(ids).decode("utf-8", "replace")` and Rust `String::from_utf8_lossy(&bytes)` both do exactly this.
- **Decode rejects** an id outside `0..=255` (Python `ValueError`, Rust an error, HTTP 400 with `param` naming the field).

Worked example by hand:

| Text | Bytes (hex) | Ids |
|---|---|---|
| `hi` | `68 69` | `[104, 105]` |
| `héllo` | `68 c3 a9 6c 6c 6f` | `[104, 195, 169, 108, 108, 111]` |

`é` is U+00E9, which UTF-8 encodes in two bytes, `c3 a9`. So one character can be two tokens, and a model that samples bytes can emit sequences that are not valid UTF-8:

| Ids | Decoded | Why |
|---|---|---|
| `[255]` | `"�"` | `ff` never appears in UTF-8 |
| `[226, 130]` | `"�"` | `e2 82` is the start of a three-byte sequence that never finishes: one maximal subpart, one U+FFFD |
| `[195, 169]` | `"é"` | complete |

### Incremental decoding (streaming)

A server that streams one chunk per token cannot decode each byte on its own. It keeps a buffer of pending bytes and, for each new token:

1. Append the token's byte to the buffer.
2. Repeat: decode the longest valid UTF-8 prefix of the buffer and emit it. If the next bytes are an invalid sequence, emit one U+FFFD, drop that maximal subpart, and continue. If they are the start of a sequence that could still complete, stop and keep them pending.
3. The chunk's text is everything emitted in step 2 (possibly `""`).

At the end of generation, decode whatever is still pending with replacement and append it to the last chunk. The concatenated chunk texts then equal `decode(all generated ids)`, which is what conformance case `v0.stream.equals_nonstream` checks. Python's `codecs.getincrementaldecoder("utf-8")("replace")` implements exactly these steps; in Rust, `std::str::from_utf8` reports `valid_up_to()` and `error_len()` (`None` means "incomplete, keep pending").

Streaming the ids `[195, 169, 255]`: chunk texts `""`, `"é"`, `"�"`.

## The char tokenizer (L1.1)

Saved as `tinyllm_char.json`: `{"type": "char", "vocab": [...], "specials": {...}}`. `vocab[i]` is the one-character string of id `i`; `specials` maps names such as `"unk"` to ids. L1.1 defines it in full.

## `tokenizer.json`, the HF subset (L1.1 to L1.6)

A subset of the Hugging Face `tokenizers` JSON format: `model.type` in `{BPE, WordPiece, Unigram}`, `pre_tokenizer.type = ByteLevel` (the GPT-2 split), `added_tokens`, and a `post_processor` limited to `TemplateProcessing`. SmolLM2's file must load unchanged. The tokenizer modules (B4) define the subset field by field.
