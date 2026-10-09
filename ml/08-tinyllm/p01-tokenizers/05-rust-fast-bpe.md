<!-- ss:module L1.5 -->
# Rust fast BPE (tl-tok), byte tokenizer, streaming decoder, PyO3 binding

## Overview

| | |
|---|---|
| **Module** | `L1.5` · build · Rust · Pass 3 · 10 to 14 h |
| **You build** | `rust/crates/tl-tok/src/`: `lib.rs` (the `Tokenizer` trait, `ByteTokenizer`, `SpecialSet`, errors), `pretok.rs` (the GPT-2 split and `Digits` by hand over general-category tables), `bpe.rs` (`ByteBpe`: merges from a lazy heap over Robin Hood maps, `encode_batch` on threads), `hfjson.rs` (the `tokenizer.json` and `vocab.json` + `merges.txt` loaders), `stream.rs` (`StreamDecoder`) · `rust/crates/tl-py/src/lib.rs` and `tok.rs`: the PyO3 module `tinyllm_rs` with `Bpe` · your binary `rust/crates/tl-tok/src/main.rs` (`tl-tok encode`, `tl-tok bench`) and the `tok` verbs of your Python CLI |
| **Contract** | Python surface: [`py/tinyllm_rs.pyi`](../../../course/contracts/py/tinyllm_rs.pyi) (`Bpe`, and the build contract) · the files: [`formats/tokenizer.md`](../../../course/contracts/formats/tokenizer.md) (bytes tokenizer, BPE subset, incremental decoding) · the roles: [`spec/cli-roles.md`](../../../course/contracts/spec/cli-roles.md) (`tl-tok`) · the Rust interface in section 4 |
| **Tests** | `course/tests/rust/l1_5.rs` (16 tests) and `course/tests/L1.5/test_tinyllm_rs.py` (6 tests, one of them run for both tokenizers); what they check: section 4 · parity suite `tokenizer.bpe` · your own tests in `rust/crates/tl-tok/tests/l1_5_props.rs`, rung R4 (proptest), graded by mutation (threshold 0.80) |
| **Needs** | `L1.2` your Python BPE, the specification every id is checked against ([chapter](02-byte-level-bpe.md)) · `ds.05` the Robin Hood map ([chapter](../../../algorithms/16-systems-data-structures/05-robin-hood-hash-map.md)) · `ds.06` the lazy heap ([chapter](../../../algorithms/16-systems-data-structures/06-binary-heap-lazy-deletion.md)) · reading: `lang.04` the [Rust primer](../../../software-craftsmanship/12-language-and-tool-primers/04-rust.md), `M05.2` the [GPT-2 byte map](../../../math/05-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md) · or `--ref-deps` |
| **Used by** | `ds.08` adds `tinyllm_rs.Bloom` to the `tl-py` crate this module creates · `data.07` packs the corpus into token streams with `encode_batch` · later: `L10.1` and `L10.5` tokenize every request and stream with `StreamDecoder`, `C1` serves the tokenizer it trains |
| **Milestone** | `MS-L1` (`tl-tok encode` and `tinyllm tok encode` against the GPT-2 and SmolLM2 oracles; the speedups are local perf steps) |
| **Optional depth** | the [PyO3 user guide](https://pyo3.rs/) (free); [Unicode Standard Annex #44](https://www.unicode.org/reports/tr44/) (general categories, free); Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) source |

## Key Takeaways

- Python is the specification (P6): your Rust tokenizer is proven id for id against your own `L1.2` and, on the same strings, against Hugging Face, so two equally wrong implementations still fail (`test_py_bpe_matches_python_l12`, `gpt2_golden_10k`, `smollm2_golden_10k`).
- BPE encoding with a lazy heap applies exactly the pair the definition names (lowest rank, then leftmost) in $O(n \log n)$ instead of $O(n^2)$; a merge must remove the two pairs it destroys (`hand_example_merges_from_the_heap`, `heap_bpe_equals_naive_bpe`).
- The GPT-2 pre-tokenizer is a regular expression you can write as a loop over character classes, if the classes are Unicode general categories, not Rust's `is_alphabetic` (`gpt2_split_hand_cases`).
- A streaming decoder holds back the bytes of an unfinished character and emits them when it completes; the concatenated chunks equal `decode(ids)` (`stream_decoder_holds_back_partial_characters`, `stream_equals_decode_on_golden`).
- One binding, `tinyllm_rs`, carries the Rust tokenizer into Python: errors become `ValueError` and `OSError`, never a panic, and `encode_batch` releases the GIL and gives the same ids on any number of threads (`test_py_decode_contract`, `test_py_encode_batch_matches_encode`).

## How to work this chapter

```bash
ss start L1.5               # stubs tl-tok and tl-py; writes their manifests if absent
ss tests L1.5               # read the test catalog first
ss check L1.5               # Rust tests, then Python tests through tinyllm_rs; then grades your tests
ss check L1.5 --ref-deps    # only if your L1.2, ds.05, or ds.06 is not passing yet
ss parity tokenizer.bpe     # your Rust against your Python and the oracle, with --fuzz for live text
ss milestone MS-L1          # your entry points: tok train, tok encode, tl-tok encode
```

`ss start` writes `rust/crates/tl-tok/Cargo.toml` (`tl-ds`, `serde_json`; `proptest` for your tests) and `rust/crates/tl-py/Cargo.toml` with `build.rs` (the PyO3 build contract below) only if they are absent. The harness builds the extension for the Python tests itself; to use it from your own Python:

```bash
PYO3_PYTHON=$(uv run --project python python -c 'import sys; print(sys.executable)') \
  cargo rustc --release --manifest-path rust/Cargo.toml -p tl-py --lib --crate-type cdylib \
  -- -C link-arg=-undefined -C link-arg=dynamic_lookup          # the last line on macOS only
cp rust/target/release/libtl_py.dylib artifacts/pyext/tinyllm_rs.so   # .so on Linux
```

Work in this order: `ByteTokenizer` and `bytes_to_unicode`; `pretok.rs` until `gpt2_split_hand_cases` passes; `ByteBpe::new` and `encode_piece` on the hand example; the loaders; then the golden tests; then `StreamDecoder`; `encode_batch`; and last the binding.

---

## 1. Why now

Your Python BPE (`L1.2`) matches GPT-2 and SmolLM2 exactly, and it is too slow for the two places that need it most. The corpus pipeline (`data.07`) must turn hundreds of megabytes of text into token streams, and the Rust engine (`L10.1`, `L10.5`) must tokenize every request on the hot path and stream text back one token at a time, where a token can end in the middle of a UTF-8 character. Neither can call Python. This module ports the tokenizer to Rust (`tl-tok`), the language the engine is written in, adds the byte tokenizer the tracer engine has served since Pass 1, a streaming decoder for the SSE path, and a PyO3 binding (`tinyllm_rs`) so Python reaches the same code. The speed comes from three things you built: Robin Hood maps for the tables (`ds.05`), a lazy heap for the merges (`ds.06`), and threads.

## 2. Principles

### 2.1 Bytes, the byte map, and the trait

| Symbol | Meaning | Type |
|---|---|---|
| $V$ | vocabulary size: ids $0 .. V-1$ | `u32` |
| $b$ | a byte, $0 \le b \le 255$ | `u8` |
| $\beta(b)$ | GPT-2's byte-to-character map (`M05.2`) | `[char; 256]` |
| $x$ | a pre-token: a run of bytes merged in isolation | `&[u8]` |
| $s_0 \dots s_{n-1}$ | the symbols of $x$ while merging: token ids | `u32` |
| $r(a, b)$ | the rank of merging ids $a$ and $b$ (lower merges first), if any | `Option<u32>` |
| $T$ | threads of `encode_batch` | `usize` |

Every tokenizer implements one trait: `encode`, `encode_with_special` (which added tokens may match), `token_bytes(id)`, `vocab_size`, and, defined once from `token_bytes`, `decode_bytes` and `decode` (UTF-8 with replacement: one U+FFFD per maximal ill-formed subsequence, `formats/tokenizer.md`). The **byte tokenizer** of the tracer (D32) is the simplest implementation: the id of a byte is its value, 256 ids, nothing to load.

A byte-level BPE file writes its tokens as strings in GPT-2's byte alphabet $\beta$: the 188 printable bytes stand for themselves and the other 68 map to U+0100, U+0101, ... in byte order, so the space 0x20 is `Ġ` (U+0120) and 0xAD, the last non-printable, is U+0143. `tl-tok` turns every token string back into bytes once, at load, with $\beta^{-1}$ (a bijection: `M05.2`), and works on bytes from then on. `ByteBpe` keeps the vocabulary as a `RobinHoodMap<Vec<u8>, u32>` (looked up by `&[u8]`, `ds.05`), the id-to-bytes table as a `Vec<Vec<u8>>`, and the merge table as `RobinHoodMap<(u32, u32), Merge>` with `Merge { rank, id }`.

### 2.2 The pre-tokenizer, by hand

Merges never cross a pre-token boundary. GPT-2's boundaries come from one regular expression, tried left to right at each position:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

`Gpt2Split` is that expression as a loop. At the current position: (1) an apostrophe followed by one of the lowercase suffixes `s t re ve m ll d` is a contraction (`'S` is not: it splits into `'` and `S`); (2 to 4) otherwise an optional single U+0020 followed by a maximal run of one class, letters (`\p{L}`, the general categories Lu Ll Lt Lm Lo), numbers (`\p{N}`: Nd Nl No), or other non-space characters; (5) a white-space run gives back its last character when a non-space follows, because that character belongs to the next pre-token (`\s+(?!\S)`); (6) else the white space itself. `\s` is the Unicode White_Space property (25 code points; Rust's `char::is_whitespace` is the same set, Python's `str.isspace` is not).

The classes must be the **general categories your Python uses**, or the ids drift apart on rare characters. Rust's `char::is_alphabetic` is the Alphabetic property, which also includes marks such as the Devanagari vowel signs (Mc), so `हिन्दी` would stay one run in Rust and split into six pieces in Python. `tl-tok` therefore embeds two tables of code-point ranges, `LETTER_RANGES` and `NUMBER_RANGES`, generated from Python's `unicodedata` by `course/oracle/L1.5/unicode_tables.py` (write your own copy) and searched by bisection. The Unicode version matters too: each Python version ships one (3.11 has 14.0, 3.12 has 15.0, 3.14 has 16.0), and a character added later is unassigned, so not a letter, to an older one. Generate the tables with the Python your `L1.2` runs under.

SmolLM2 adds one step before ByteLevel: `Digits` with `individual_digits = true` cuts every `\p{N}` code point into its own piece, so `12345` is five pre-tokens and five tokens. A `PreTokenizer` is a list of steps, each applied to every piece the previous step produced.

### 2.3 Merging with a lazy heap

Inside one pre-token, BPE starts from one symbol per byte and repeatedly merges the adjacent pair $(s_i, s_{i+1})$ with the lowest $r$, the **leftmost** among equal ranks, until no adjacent pair has a rank. The rescan written from the definition costs $O(n)$ per merge and $O(n^2)$ per pre-token. `encode_piece` keeps the symbols in a doubly linked list over an array (`prev`, `next`) and every mergeable pair in a `LazyHeap` (`ds.06`) keyed by the tuple (rank, position of the left symbol), so the tuple order is exactly "lowest rank, then leftmost". Popping $(r, i)$ merges $s_i$ and its successor $s_j$:

1. the pairs $(s_{prev}, s_i)$ and $(s_j, s_{next})$ no longer exist: remove them **by handle** from the heap (this is what lazy deletion is for);
2. $s_i$ becomes the merged id and absorbs $s_j$'s link;
3. the new pairs $(s_{prev}, s_i)$ and $(s_i, s_{next})$ are pushed if they have a rank.

Each merge is $O(\log n)$, and the ids equal the rescan's, because both always apply the pair the definition names (`heap_bpe_equals_naive_bpe` checks it on 3,000 random pieces). One `LazyHeap`, cleared between pieces, serves a whole text.

**Added tokens** (`<|endoftext|>`, SmolLM2's `<|im_start|>` and 15 more) are split out first, leftmost-longest, as the Hugging Face library does, and never pre-tokenized or merged. `encode` allows all of them; `encode_with_special(text, &SpecialSet::none())` allows none, so a server can keep user text from forging control tokens.

### 2.4 Loading `tokenizer.json`

`ByteBpe::from_hf_json` reads the `formats/tokenizer.md` BPE subset that GPT-2's and SmolLM2's files use: `model.type` `BPE` (or absent, when `model.merges` is present), `model.vocab`, `model.merges` as `"left right"` strings or `[left, right]` pairs, a `ByteLevel` pre-tokenizer or a `Sequence` of `Digits` and one final `ByteLevel`, `added_tokens`. Anything else (a normalizer, another model, `byte_fallback`, `add_prefix_space: true`) is refused with an error that names the field, so a file that loads is a file this code encodes exactly. A merge whose parts or result are not in the vocabulary is a `Format` error, and a repeated merge keeps its first rank. `from_gpt2_files` reads the classic `vocab.json` and `merges.txt` (skipping the `#version` line); the tokenizer your Python trains and saves (`L1.2`, `save`) loads here unchanged, which is how `C1` serves the vocabulary it trains.

### 2.5 Streaming decode

A token's bytes need not end on a character boundary: GPT-2 splits the emoji U+1F600 (`f0 9f 98 80`) into tokens 47249 (`f0 9f 98`) and 222 (`80`). An engine that streams one SSE chunk per token must neither emit half a character nor stall on bytes that can never form one. `StreamDecoder` keeps the pending bytes and, after each token, emits the longest valid UTF-8 prefix; an invalid sequence becomes one U+FFFD for its maximal subpart, and an incomplete tail that could still complete stays pending (at most 3 bytes). `std::str::from_utf8` gives exactly the two numbers needed: `valid_up_to()` and `error_len()`, which is `None` for "incomplete, wait". `finish` decodes what is left with replacement. Concatenating every chunk and `finish` gives `decode(ids)`, the property the conformance case `chat.stream.equals_nonstream` relies on.

### 2.6 Threads, a cache, and the binding

`encode_batch(texts, T)` cuts the texts into $T$ contiguous chunks, encodes each on its own thread with `std::thread::scope` (which lets the threads borrow the tokenizer and the texts), and joins the results **in chunk order**, so the output does not depend on $T$. $T = 0$ means the hardware concurrency. Each worker keeps its own piece cache, another `RobinHoodMap` from pre-token bytes to ids, cleared at 65,536 entries: real text repeats its words, so most pieces are hits, and no lock is shared between threads.

`tl-py` is a `cdylib` built with PyO3's `abi3-py311` and `extension-module` features. A Python extension does not link against libpython: the interpreter provides those symbols when it loads the module, so on macOS the library links with `-undefined dynamic_lookup` (`build.rs`), and `PYO3_PYTHON` names your uv interpreter. The crate root declares one `#[pymodule] fn tinyllm_rs` that adds `tok::Bpe` and `bloom::Bloom` (`ds.08`). `Bpe.encode_batch` runs inside `Python::detach`, which releases the GIL while the Rust threads work, so other Python threads keep going. A Rust panic must never cross the boundary (PyO3 would raise `PanicException`): every bad input is checked and returned as `ValueError` (bad ids, negative thread counts, files outside the subset) or `OSError` (a file that cannot be read).

## 3. Worked example by hand

Take the `L1.2` worked example: the 256 byte tokens in your Python's order (ids by the code point of their byte-map character, so `g` is 70, `s` is 82, `Ġ` is 220) and three merges: rank 0 `u g` makes id 256 (`ug`), rank 1 `h ug` makes 257 (`hug`), rank 2 `p ug` makes 258 (`pug`). Encode `hugs pug`.

**Pre-tokens.** `hugs` is a letter run; the space joins the next run: `hugs`, ` pug`.

**`hugs`.** Symbols `h u g s` at positions 0 to 3. Mergeable pairs: only `(u, g)` at position 1, rank 0. Heap: $\{(0, 1)\}$.

| Pop | Merge | Removed pairs | New pairs pushed | Symbols |
|---|---|---|---|---|
| (0, 1) | `u g` to 256 at position 1 | none queued at 0 or 2 | `(h, ug)` at 0, rank 1; `(ug, s)` has no rank | `h ug s` |
| (1, 0) | `h ug` to 257 at position 0 | none | `(hug, s)` has no rank | `hug s` |

Ids: 257, 82.

**` pug`.** Symbols `Ġ p u g`; the only ranked pair is `(u, g)` at position 2. Pop (0, 2): `ug` at 2, push `(p, ug)` at 1 with rank 2. Pop (2, 1): `pug` at 1; `(Ġ, pug)` has no rank. Ids: 220, 258.

Result `[257, 82, 220, 258]`, exactly your Python's. **Lazy removal** shows on `aaaa` with merges rank 0 `a a` (256) and rank 1 `aa aa` (257): the heap holds $(0,0), (0,1), (0,2)$. Pop $(0,0)$: positions 0 and 1 merge, and the queued pair at position 1 (`a` at 1 with `a` at 2) no longer exists: it is removed by its handle. Pop $(0, 2)$: positions 2 and 3 merge, and the new pair `(aa, aa)` at 0 is pushed with rank 1. Pop $(1, 0)$: one token, 257. Without the removal, the stale $(0, 1)$ would pop second and merge a symbol that is already gone. This is `hand_example_merges_from_the_heap`.

**Streaming.** Push 47249: pending `f0 9f 98`, a 4-byte lead and two continuations, incomplete, so the chunk is empty. Push 222: `80` completes the character and the chunk is `😀`.

## 4. The interface

```rust
// rust/crates/tl-tok/src/lib.rs
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;                          // every added token matches
    fn encode_with_special(&self, text: &str, allowed: &SpecialSet) -> Vec<u32>;
    fn token_bytes(&self, id: u32) -> Option<&[u8]>;
    fn vocab_size(&self) -> u32;
    fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, DecodeError>;   // provided
    fn decode(&self, ids: &[u32]) -> Result<String, DecodeError>;          // provided: UTF-8 with replacement
}
pub struct SpecialSet;  impl SpecialSet { pub fn all() -> Self; pub fn none() -> Self; pub fn only(t: &[&str]) -> Self; pub fn allows(&self, t: &str) -> bool; }
pub enum DecodeError { UnknownId { id: u32, vocab_size: u32 } }
pub enum LoadError { Io { path: PathBuf, source: io::Error }, Json(String), Unsupported { field: String, value: String }, Format(String) }
pub struct ByteTokenizer;                                              // impl Tokenizer: id = byte

// pretok.rs
pub const LETTER_RANGES: &[(u32, u32)];  pub const NUMBER_RANGES: &[(u32, u32)];  pub const WHITE_SPACE: &[char];
pub fn is_letter(c: char) -> bool;  pub fn is_number(c: char) -> bool;  pub fn is_space(c: char) -> bool;
pub fn pretokenize_gpt2(text: &str) -> Vec<&str>;  pub fn split_digits(text: &str, individual: bool) -> Vec<&str>;
pub enum Step { Digits { individual: bool }, ByteLevel }
pub struct PreTokenizer { pub steps: Vec<Step> }  impl PreTokenizer { pub fn gpt2() -> Self; pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str>; }

// bpe.rs
pub fn bytes_to_unicode() -> [char; 256];  pub fn unicode_to_bytes() -> Vec<Option<u8>>;
pub struct AddedToken { pub content: String, pub id: u32, pub special: bool }
pub struct Merge { pub rank: u32, pub id: u32 }
pub fn by_rank(a: &(u32, usize), b: &(u32, usize)) -> Ordering;     // the merge queue's order
pub struct ByteBpe;                                                    // impl Tokenizer
impl ByteBpe {
    pub fn new(vocab: Vec<(Vec<u8>, u32)>, merges: Vec<(Vec<u8>, Vec<u8>)>, added: Vec<AddedToken>, pretok: PreTokenizer) -> Result<Self, LoadError>;
    pub fn from_hf_json(path: &Path) -> Result<Self, LoadError>;  pub fn from_hf_json_str(text: &str) -> Result<Self, LoadError>;
    pub fn from_gpt2_files(vocab_json: &Path, merges_txt: &Path) -> Result<Self, LoadError>;
    pub fn token_to_id(&self, bytes: &[u8]) -> Option<u32>;  pub fn merge(&self, left: u32, right: u32) -> Option<Merge>;
    pub fn merge_count(&self) -> usize;  pub fn added_tokens(&self) -> &[AddedToken];  pub fn pre_tokenizer(&self) -> &PreTokenizer;
    pub fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>);
    pub fn encode_piece_with(&self, piece: &[u8], out: &mut Vec<u32>, scratch: &mut MergeScratch);
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32>;
    pub fn encode_cached(&self, text: &str, cache: &mut RobinHoodMap<Vec<u8>, Vec<u32>>) -> Vec<u32>;
    pub fn encode_batch(&self, texts: &[&str], threads: usize) -> Vec<Vec<u32>>;
}

// stream.rs
pub struct StreamDecoder<'a, T: Tokenizer + ?Sized>;
impl<'a, T: Tokenizer + ?Sized> StreamDecoder<'a, T> {
    pub fn new(tok: &'a T) -> Self;
    pub fn push(&mut self, id: u32) -> Result<Option<String>, DecodeError>;   // None: nothing completed yet
    pub fn pending(&self) -> &[u8];  pub fn finish(&mut self) -> String;
}
```

```python
# tinyllm_rs (contracts/py/tinyllm_rs.pyi), from rust/crates/tl-py/src/{lib,tok}.rs
class Bpe:
    @staticmethod
    def from_hf_json(path: str) -> "Bpe": ...        # OSError: unreadable; ValueError: outside the subset
    def encode(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: list[str], threads: int) -> list[list[int]]: ...   # releases the GIL
    def decode(self, ids: list[int]) -> str: ...     # ValueError for an id outside 0 .. vocab_size - 1
    def vocab_size(self) -> int: ...
```

The binary is yours (D16): `tl-tok encode --tokenizer <file> --in <texts>` and `tl-tok bench`, with the final JSON lines that `MS-L1` fixes; declare it in `system.toml` as the `tl-tok` role (the reference uses `cargo run --release -p tl-tok --`). The Python CLI gains `tok train`, `tok encode`, and `tok bench` over `L1.2` and this binding.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_merges_from_the_heap` | unit | section 3: `hugs pug` is `[257, 82, 220, 258]`; `aaaa`, `aaaaa`, `aaaaaa` with lazy removals | you, your Python, and the tests agree on the rule |
| `heap_bpe_equals_naive_bpe` | differential | 3,000 random pieces: heap ids equal the $O(n^2)$ rescan | the heap is only a faster route to the same ids |
| `byte_tokenizer_is_the_identity` | unit | 256 ids, id = byte, `token_bytes(256)` is `None`, `decode([0xFF])` is U+FFFD | the tracer engine keeps serving bytes |
| `byte_map_matches_gpt2` | unit | $\beta$ at 0x20, 0x21, 0x7E, 0x00, 0x7F, 0xAD, 0xAE, 0xFF; a bijection; vocabulary keys are bytes | every token string loads to the right bytes |
| `gpt2_split_hand_cases` | unit | contractions in both cases, space runs, `\s+(?!\S)`, digits, marks, Devanagari | the boundaries your Python draws |
| `digits_split_for_smollm2` | unit | `Digits` individual and not; `12345` is five SmolLM2 tokens | SmolLM2's pre-tokenizer sequence |
| `gpt2_golden_10k` | golden | 10,000 strings equal Hugging Face's and tiktoken's GPT-2 ids | exact against the oracle |
| `smollm2_golden_10k` | golden | the same strings under SmolLM2's file | exact against the oracle with Digits and 17 added tokens |
| `decode_inverts_encode` | property | `decode_bytes(encode(x))` is $x$'s UTF-8 for every golden string and both files | byte-level BPE is lossless |
| `added_tokens_are_matched_first` | unit | SmolLM2's chat markers; `SpecialSet::none` and `only` | servers can refuse forged control tokens |
| `stream_decoder_holds_back_partial_characters` | boundary | the emoji split over two GPT-2 tokens; `C3 A9 FF E2 82` byte by byte; `finish` | the SSE path never emits half a character |
| `stream_equals_decode_on_golden` | property | 2,000 golden strings: chunks plus `finish` equal `decode`; at most 3 pending bytes | streamed text equals non-streamed text |
| `encode_batch_equals_serial_encode` | property | 997 texts on 0, 1, 2, 3, 4, 8 threads equal serial `encode` | `data.07` output does not depend on core count |
| `loader_rejects_files_outside_the_subset` | boundary | a normalizer, `WordPiece`, `add_prefix_space`, `byte_fallback`, bad JSON, a missing file | a file that loads is one you encode exactly |
| `gpt2_classic_files_load_identically` | differential | `vocab.json` + `merges.txt` written from `tokenizer.json` encode 1,000 strings identically | GPT-2's other distribution |
| `vocab_and_merge_tables` | unit | GPT-2's first merge `(Ġ, t)` is rank 0, id 256; sizes 50,257 and 50,000 | the Robin Hood tables hold what the file says |
| `test_py_bpe_matches_python_l12` | differential, golden | 1,500 golden strings per file: Rust == your Python == the oracle | the port is proven against your own specification |
| `test_py_random_text_matches_python` | differential | 800 seeded random strings over contractions, spaces, marks, CJK, emoji, digits | agreement beyond the fixtures |
| `test_py_trained_tokenizer_loads_in_rust` | differential | a tokenizer your Python trains and saves loads in Rust and encodes identically | `C1` serves the vocabulary it trains |
| `test_py_encode_batch_matches_encode` | property, boundary | thread counts 0, 1, 3 give `encode`'s ids; `threads = -1` is ValueError | the GIL-free batch path |
| `test_py_decode_contract` | boundary | the split emoji, a lone partial token gives U+FFFD, bad and negative ids are ValueError | errors cross as exceptions, not panics |
| `test_py_load_errors` | boundary | a missing file is OSError; bad JSON and a normalizer are ValueError naming the field | the contract's exceptions |

**Your tests (rung R4).** Write `rust/crates/tl-tok/tests/l1_5_props.rs` with proptest (`craft.04` teaches it), reading the two tokenizer files from `$TINYLLM_FIXTURES` (`ss check` sets it). Properties: `decode(encode(s)) == s` for any `String`, both files; the heap equals a naive rescan you write in the test; pre-token pieces concatenate to the input; streamed chunks equal `decode`; `encode_batch` equals serial `encode` for any thread count; the byte tokenizer round-trips any bytes. Add a few pinned examples (known GPT-2 and SmolLM2 ids, `IT'S`, a normalizer that must be refused, `SpecialSet::none`). Use `Config { rng_seed: RngSeed::Fixed(..), failure_persistence: None, .. }`. At least 80% of the planted bugs, and every planted bug outside the binding, must make one fail.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Breaking rank ties toward the rightmost pair | `aaa` encodes as `a aa`, not `aa a`; long runs of one letter disagree with Python | `hand_example_merges_from_the_heap`, `heap_bpe_equals_naive_bpe`, `gpt2_golden_10k` (mutant `s01`) |
| Not removing the pair a merge destroys | a stale pair pops later and merges a symbol that is gone: wrong ids or a panic | `heap_bpe_equals_naive_bpe`, `test_py_trained_tokenizer_loads_in_rust` (mutant `s02`) |
| An off-by-one in the byte tokenizer's id range | id 256 is accepted and indexes past the table | `byte_tokenizer_is_the_identity` (mutant `s03`) |
| Treating 0xAD as printable in the byte map | two bytes share a character; a whole class of tokens loads to the wrong bytes | `byte_map_matches_gpt2`, `decode_inverts_encode` (mutant `s04`) |
| Forgetting the `\s+(?!\S)` back-off | `a  b` encodes as `a`, `  `, `b`, not `a`, ` `, ` b` | `gpt2_split_hand_cases` (mutant `s05`) |
| Matching contractions in any case | `IT'S` splits `'S` off; GPT-2 is case-sensitive here | `gpt2_split_hand_cases` (mutant `s06`) |
| Ignoring `individual_digits` | the pre-tokens differ from SmolLM2's; the ids agree only by luck of its vocabulary | `digits_split_for_smollm2` (mutant `s07`) |
| Using `char::is_alphabetic` for `\p{L}` | marks join letter runs: Devanagari, Thai, and combining accents tokenize differently | `gpt2_split_hand_cases`, `gpt2_golden_10k` (mutant `s08`) |
| Matching added tokens whatever the `SpecialSet` says | user text containing `<|endoftext|>` ends a document | `added_tokens_are_matched_first` (mutant `s09`) |
| Flushing an incomplete character as U+FFFD | every emoji split over two tokens streams as two replacement characters | `stream_decoder_holds_back_partial_characters` (mutant `s10`) |
| Not draining the emitted bytes | every chunk repeats the text before it | `stream_equals_decode_on_golden` (mutant `s11`) |
| Joining thread results out of order | `encode_batch` reorders documents when $T > 1$ | `encode_batch_equals_serial_encode` (mutant `s12`) |
| Accepting a normalizer and ignoring it | a file with NFC or lowercasing loads and encodes differently from Hugging Face | `loader_rejects_files_outside_the_subset` (mutant `s13`) |
| Casting a negative thread count to `usize` | Python callers pass `-1` and get no error | `test_py_encode_batch_matches_encode` (mutant `s14`) |
| Mapping a missing file to `ValueError` | callers cannot tell a typo from a broken file | `test_py_load_errors` (mutant `s15`) |
| Decoding strictly instead of with replacement | a partial character decodes to an empty string, and text is lost | `test_py_decode_contract`, `byte_tokenizer_is_the_identity` (mutant `s16`) |
| Generating the category tables with a different Python than your `L1.2` | a handful of characters assigned in a newer Unicode version split differently | the parity suite tokenizer.bpe in fuzz mode (the golden strings avoid those characters) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.2` | the specification: every id here is checked against your Python on the fixtures and on random text |
| Back | `ds.05` | the vocabulary, the merge table, and the piece cache are `RobinHoodMap`s |
| Back | `ds.06` | every merge comes out of a `LazyHeap` and removes destroyed pairs by handle |
| Back | `lang.04` | traits, threads with `std::thread::scope`, `extern` linkage for the extension |
| Back | `M05.2` | the byte map $\beta$ and its inverse |
| Forward | `ds.08` | adds `bloom::Bloom` to the `tl-py` module object this crate root defines |
| Forward | `L10.1` | the engine's request path encodes prompts with `ByteBpe` and keeps serving `ByteTokenizer` for the tracer model |
| Forward | `L10.5` | the SSE server streams each token through `StreamDecoder` |
| Forward | `data.07` | the corpus tokenize stage calls `tinyllm_rs.Bpe.encode_batch` on every core |
| Forward | `C1` | the capstone trains its tokenizer in Python and serves it from Rust |

If you skip this module, `ss check ds.08` stops with `needs L1.5: build it, or pass --ref-deps`, and `MS-L1` has no `tl-tok` role to run.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ByteBpe` merges | [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) | the same lazy merge queue, dropout, `byte_fallback`, normalizers, offsets for every token | `tokenizers/src/models/bpe/word.rs`, `model.rs` |
| `Gpt2Split` by hand | [tiktoken](https://github.com/openai/tiktoken) | the regex compiled with `fancy-regex`, and a rank-based merge over byte pairs with no merges table | `src/lib.rs` (`byte_pair_merge`) |
| the piece cache | tokenizers' `Cache` | one bounded cache shared by every thread behind a read-write lock | `tokenizers/src/utils/cache.rs` |
| `StreamDecoder` | [vLLM's detokenizer](https://github.com/vllm-project/vllm) | incremental detokenization with prefix offsets, for any tokenizer | `vllm/transformers_utils/detokenizer_utils.py` |
| `tinyllm_rs` | [maturin](https://github.com/PyO3/maturin) | builds and packages the same cdylib as a wheel | `maturin build --release` |
