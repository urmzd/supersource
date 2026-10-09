# contracts/py/tinyllm_rs.pyi: the PyO3 module built from rust/crates/tl-py
# (DESIGN 2.5, D8). Python reaches Rust only through this module.
#
# modules: L1.5 (crate root tl-py/src/lib.rs and tok.rs: Bpe), ds.08 (bloom.rs: Bloom)
# conformance: parity/tokenizer.bpe, parity/bloom
#
# Build contract (checked by the contract pre-check): tl-py has
# crate-type = ["cdylib"], PyO3 features "abi3-py311" and "extension-module",
# and on macOS the link argument "-undefined dynamic_lookup" (in
# rust/.cargo/config.toml or build.rs). The harness sets PYO3_PYTHON to the
# learner's uv interpreter, builds with `cargo build -p tl-py`, copies
# libtl_py.dylib (libtl_py.so on Linux) to tinyllm_rs.so in TINYLLM_PYEXT_DIR,
# and prepends that directory to PYTHONPATH. maturin is optional.
#
# Errors cross as Python exceptions: ValueError for bad input, OSError for
# I/O. A Rust panic must never cross the boundary (PyO3 turns one into
# pyo3_runtime.PanicException, which the course tests treat as a failure).

class Bpe:
    """Byte-level BPE, exact against GPT-2 and the HF `tokenizers` library
    on the formats/tokenizer.md subset (model.type BPE with the ByteLevel
    pre-tokenizer)."""

    @staticmethod
    def from_hf_json(path: str) -> "Bpe":
        """Load a tokenizer.json. OSError when the file cannot be read;
        ValueError when it is not valid JSON or uses a model, pre-tokenizer,
        or post-processor outside the formats/tokenizer.md subset (the
        message names the field)."""

    def encode(self, text: str) -> list[int]:
        """Ids of `text` without special tokens added by a post-processor;
        added tokens that appear literally in `text` are matched first, as
        the HF library does. Equal to tokenizers.Tokenizer.encode(text,
        add_special_tokens=False).ids."""

    def encode_batch(self, texts: list[str], threads: int) -> list[list[int]]:
        """encode() of each text, in order. Releases the GIL and runs on
        `threads` OS threads (0 means the hardware concurrency); the result
        does not depend on `threads`. ValueError for threads < 0."""

    def decode(self, ids: list[int]) -> str:
        """The bytes of the ids concatenated and decoded as UTF-8 with
        replacement (one U+FFFD per maximal ill-formed subsequence, as
        formats/tokenizer.md). ValueError for an id outside the vocabulary."""

    def vocab_size(self) -> int:
        """Number of ids, added tokens included: one more than the largest id."""

class Bloom:
    """A Bloom filter over byte strings. Sizing, hashing, and the byte
    layout of to_bytes are fixed by formats/bloom.md, so a filter built in
    one implementation answers identically in another."""

    @staticmethod
    def with_rate(n: int, p: float) -> "Bloom":
        """An empty filter for n expected items at false-positive rate p:
        m = ceil(-n ln p / (ln 2)^2) bits and k = max(1, round(m / n * ln 2))
        hashes. ValueError unless n >= 1 and 0 < p < 1."""

    def insert(self, item: bytes) -> None: ...
    def contains(self, item: bytes) -> bool:
        """True for every inserted item (no false negatives)."""

    def union(self, other: "Bloom") -> None:
        """In place: bitwise OR of the bit arrays; the item counts add.
        ValueError when m or k differ."""

    def to_bytes(self) -> bytes:
        """The formats/bloom.md serialization."""

    @staticmethod
    def from_bytes(b: bytes) -> "Bloom":
        """Inverse of to_bytes. ValueError for a bad magic, version, length,
        or an m or k of 0."""
