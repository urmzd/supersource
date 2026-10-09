# `model.safetensors`

A safetensors file holds named tensors and a small string-to-string metadata map. The course reads and writes it in Python (`tinyllm/io/safetensors.py`: L0.0 writes and reads F32 only, L0.6 adds every dtype), and the Rust engine memory-maps it (L10.0 reads the bigram, L10.1 the rest). A writer that follows this page produces the **same bytes** as the `safetensors` library pinned by the course oracle, which is what the L0.0 and L0.6 golden tests compare.

## Layout

```
offset 0        8 bytes   N, the header length: unsigned 64-bit, little-endian
offset 8        N bytes   the header: UTF-8 JSON, padded with spaces (0x20) so N % 8 == 0
offset 8 + N    the data buffer: every tensor's bytes, back to back
```

The header is one JSON object. Each key except `__metadata__` is a tensor name; its value is

```json
{"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
```

- `dtype` is one of the names in the dtype table below.
- `shape` lists the dimensions, outermost first. `[]` is a scalar (one element).
- `data_offsets` is `[begin, end)` in bytes, **relative to the start of the data buffer**, not to the file.

`__metadata__`, when present, maps strings to strings. Course writers always set `"format": "tinyllm"`; quantized files add `"quant"` (below).

Tensor bytes are little-endian and row-major (C order: the last index varies fastest). There is no padding between tensors.

## Rules a reader enforces

A reader rejects the file (Python `ValueError`, Rust an error, C `TL_EFORMAT`) when any of these fail:

1. The file is at least 8 bytes and `8 + N` is at most the file size. `N` is at most 100,000,000.
2. The header is valid UTF-8 JSON, an object, with no duplicate keys.
3. `__metadata__`, if present, has only string values.
4. Every `dtype` is known, every dimension is a non-negative integer, and `end - begin` equals the product of `shape` times the dtype's size in bytes.
5. Sorted by `begin`, the tensors tile the data buffer exactly: the first begins at 0, each begins where the previous ended, and the last ends at the end of the file. No gaps, no overlap, no trailing bytes.

## Rules a canonical writer follows

These are the choices the pinned library makes. Following them makes a writer byte-identical to it.

1. **Tensor order.** Sort the tensors by dtype, in this order (the reverse of the library's dtype enumeration, so larger alignment comes first):
   `U64, I64, F64, F32, U32, I32, BF16, F16, U16, I16, F8_E4M3, F8_E5M2, I8, U8, BOOL`.
   Within one dtype, sort by name, comparing the UTF-8 bytes. Lay the data out in this order, with no gaps. (A single-dtype file, such as every v0 file, is simply sorted by name.)
2. **Header text.** Compact JSON, no whitespace between tokens. `__metadata__` comes first, then the tensors in the order of rule 1. Inside a tensor entry the keys are `dtype`, `shape`, `data_offsets`, in that order. Non-ASCII characters are written as raw UTF-8, not as `\u` escapes. In Python: `json.dumps(obj, separators=(",", ":"), ensure_ascii=False)` on a `dict` built in that order.
3. **Metadata.** Omit `__metadata__` when the map is empty. With several keys, write them in ascending byte order. (Golden tests compare against the library with exactly one metadata key, because the library keeps several keys in a hash map: their order is the one thing this page cannot pin.)
4. **Padding.** Append spaces until the header length is a multiple of 8. `N` counts the spaces.

## Worked example

One F32 tensor `w = [[1, 2], [3, 4]]` with metadata `{"format": "tinyllm"}`.

The header text is 93 bytes:

```
{"__metadata__":{"format":"tinyllm"},"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}
```

93 is not a multiple of 8, so three spaces follow and N = 96. The file is:

```
60 00 00 00 00 00 00 00                            N = 96, little-endian u64
7b 22 5f 5f 6d 65 ... 7d 7d 20 20 20               96 header bytes, the last three are spaces
00 00 80 3f 00 00 00 40 00 00 40 40 00 00 80 40    1.0f 2.0f 3.0f 4.0f, little-endian
```

Total size: 8 + 96 + 16 = 120 bytes.

## Dtypes

| Name | Bytes | numpy | `tl_dtype` | Course use |
|---|---|---|---|---|
| `F32` | 4 | `float32` | `TL_F32` | weights and checkpoints (v0: the only dtype) |
| `F16` | 2 | `float16` | `TL_F16` | weights, int4 scales, KV format v1 |
| `BF16` | 2 | (raw `uint16`) | `TL_BF16` | mixed precision (L11.1), SmolLM2 weights |
| `F8_E4M3` | 1 | (raw `uint8`) | `TL_F8_E4M3` | fp8 weights (M09.4) |
| `U8` | 1 | `uint8` | `TL_U8` | packed int4 `qweight` |
| `I8` | 1 | `int8` | `TL_I8` | int8 weights |
| `I32` | 4 | `int32` | `TL_I32` | integer buffers |

Contract v0 (L0.0) writes and the tracer engine reads `F32` only; any other dtype is an unsupported-format error until L0.6 and L10.1.

## Tensor names

| `tl_arch` | Tensors |
|---|---|
| `bigram` (the tracer) | `bigram.weight`, F32, shape `[vocab_size, vocab_size]`. Row `i` holds the next-token logits after token `i`: `logits = onehot(ids) @ W`. The count model of L0.0 stores the log of its add-one-smoothed probabilities, so each row's softmax is the model's distribution; L0.5 stores trained logits under the same name |
| `llama` | Llama/HF names: `model.embed_tokens.weight`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `model.layers.{i}.mlp.{gate,up,down}_proj.weight`, `model.layers.{i}.{input,post_attention}_layernorm.weight`, `model.norm.weight`, optional `lm_head.weight` (absent when `tie_word_embeddings` is true) |

Other architectures add their names with the module that introduces them.

## Int4 weights (L8.5, L9.5)

A quantized linear weight `<name>` of shape `[out, in]` becomes two tensors:

- `<name>.qweight`, U8, shape `[out, in / 2]`. Byte `b` of row `r` holds columns `2b` (low nibble) and `2b + 1` (high nibble), each a signed 4-bit value in two's complement.
- `<name>.scales`, F16, shape `[out, in / group]`. Weight `(r, c)` is `q * scales[r, c / group]`.

The file's metadata is `{"format": "tinyllm", "quant": "int4-g<group>-sym"}`, for example `int4-g32-sym`.

## Model directory

A model directory (the engine's `--model-dir`, a checkpoint step) holds `model.safetensors` and [`config.json`](config.schema.json). It holds `tokenizer.json` only when `config.json` says `"tl_tokenizer": "file"` ([tokenizer.md](tokenizer.md)).
