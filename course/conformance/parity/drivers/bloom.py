"""Parity driver for the independent Python Bloom screen in data.03.

Reads one case per stdin line and prints the version-1 Bloom bytes and probe
answers. The screen algorithm comes from the corpus dedup reference; this
adapter serializes its bits according to contracts/formats/bloom.md.
"""

from __future__ import annotations

import json
import struct
import sys

from corpus.dedup import _Bloom

for line in sys.stdin:
    if not line.strip():
        continue
    case = json.loads(line)
    bloom = _Bloom(int(case["n"]), float(case["p"]))
    inserted = case["insert"]
    for item in inserted:
        bloom.insert(bytes.fromhex(item))
    header = struct.pack(
        "<4sIQIIQ",
        b"TLBF",
        1,
        bloom.m,
        bloom.k,
        0,
        len(inserted),
    )
    raw = header + bytes(bloom.bits)
    print(
        json.dumps(
            {
                "m": bloom.m,
                "k": bloom.k,
                "bytes": raw.hex(),
                "contains": [bloom.contains(bytes.fromhex(x)) for x in case["probe"]],
            }
        ),
        flush=True,
    )
