# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer fetcher for course/fixtures/small-corpora/tinyshakespeare.txt.

Tiny Shakespeare (Karpathy, char-rnn, 2015): 1,115,394 bytes of
Shakespeare's plays, the classic character-level language-modeling corpus.
The text is in the public domain; this script downloads it from the char-rnn
repository at a pinned commit and checks its sha256, so the committed file
is byte for byte the published one.

    uv run --python 3.12 --script course/oracle/MS-L3/tinyshakespeare.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
URL = f"https://raw.githubusercontent.com/karpathy/char-rnn/{COMMIT}/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
OUT = Path("course/fixtures/small-corpora/tinyshakespeare.txt")


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        raise SystemExit(f"{URL}: sha256 {got}, want {SHA256}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print("\t".join([str(OUT), got, str(len(data)), "course/oracle/MS-L3/tinyshakespeare.py", "-",
                     f"github:karpathy/char-rnn@{COMMIT}:data/tinyshakespeare/input.txt", "public-domain"]))


if __name__ == "__main__":
    main()
