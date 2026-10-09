"""The Tokenizer protocol and the byte tokenizer (L1.1).

Every tokenizer in the course (bytes, char, BPE, WordPiece, Unigram, and the
Rust `tl-tok` behind PyO3) has the same shape: `encode` turns text into ids,
`decode` turns ids back into text, and `save`/`load` move it through a model
directory. Code that consumes tokens (the corpus pipeline, the trainers, the
metrics of L1.6, the engine) is written against `Tokenizer` only.

`ByteTokenizer` is the tracer's tokenizer (D32, formats/tokenizer.md): the 256
byte values are the vocabulary, so it needs no training and no file. Its
token strings are the GPT-2 byte map of M05.2 (byte 0x20 is "Ġ"), the same
alphabet byte-level BPE (L1.2) starts from: bytes are BPE with no merges.

Contract: contracts/py/tinyllm/tok/base.pyi.
"""

from __future__ import annotations

import os
from typing import Optional, Protocol, Sequence, runtime_checkable

from tinyllm.tok.bytes_unicode import bytes_to_unicode, unicode_to_bytes


@runtime_checkable
class Tokenizer(Protocol):
    """What every tokenizer provides. Ids are integers in [0, vocab_size)."""

    vocab_size: int
    special_ids: dict[str, int]
    unk_id: Optional[int]

    def encode(self, text: str, add_special: bool = False) -> list[int]: ...

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str: ...

    def token_to_id(self, s: str) -> Optional[int]: ...

    def id_to_token(self, i: int) -> str: ...

    def save(self, dir: str) -> None: ...

    @classmethod
    def load(cls, dir: str) -> "Tokenizer": ...


def check_ids(ids: Sequence[int], vocab_size: int) -> list[int]:
    """ids as a list of Python ints, each checked against [0, vocab_size).
    ValueError for a bool, a non-integer, or an id out of range."""
    # SOLUTION-BEGIN L1.1
    out: list[int] = []
    for i in ids:
        if isinstance(i, bool) or not hasattr(i, "__index__"):
            raise ValueError(f"token ids must be integers, got {i!r}")
        v = int(i)
        if not 0 <= v < vocab_size:
            raise ValueError(f"token id {v} is outside [0, {vocab_size})")
        out.append(v)
    return out
    # SOLUTION-END


class ByteTokenizer:
    """The identity byte tokenizer (D32): id = byte value, no specials."""

    vocab_size = 256

    def __init__(self) -> None:
        # SOLUTION-BEGIN L1.1
        self.special_ids: dict[str, int] = {}
        self.unk_id: Optional[int] = None
        self._b2u = bytes_to_unicode()
        self._u2b = unicode_to_bytes()
        # SOLUTION-END

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        # SOLUTION-BEGIN L1.1
        return list(text.encode("utf-8"))
        # SOLUTION-END

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        # SOLUTION-BEGIN L1.1
        return bytes(check_ids(ids, 256)).decode("utf-8", "replace")
        # SOLUTION-END

    def token_to_id(self, s: str) -> Optional[int]:
        # SOLUTION-BEGIN L1.1
        return self._u2b.get(s)
        # SOLUTION-END

    def id_to_token(self, i: int) -> str:
        # SOLUTION-BEGIN L1.1
        (v,) = check_ids([i], 256)
        return self._b2u[v]
        # SOLUTION-END

    def save(self, dir: str) -> None:
        # SOLUTION-BEGIN L1.1
        # Nothing to write: config.json declares tl_tokenizer = "bytes".
        os.makedirs(dir, exist_ok=True)
        # SOLUTION-END

    @classmethod
    def load(cls, dir: str) -> "ByteTokenizer":
        # SOLUTION-BEGIN L1.1
        return cls()
        # SOLUTION-END
