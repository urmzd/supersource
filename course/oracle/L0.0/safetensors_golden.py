# /// script
# requires-python = ">=3.11"
# dependencies = ["safetensors==0.8.0", "numpy==2.2.6"]
# ///
"""Maintainer generator for the L0.0 safetensors golden fixtures.

Writes course/fixtures/L0.0/safetensors/<case>.safetensors with the pinned
`safetensors` library and cases.json with the inputs, so the course test can
rebuild each input, call the learner's save_safetensors, and compare bytes.

    uv run --script course/oracle/L0.0/safetensors_golden.py

Run from the repo root. Update course/fixtures/MANIFEST.tsv with the new
sha256 and sizes afterwards (the script prints the rows).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import safetensors
from safetensors.numpy import save

OUT = Path("course/fixtures/L0.0/safetensors")


def ramp(n: int, scale: float = 0.5, shift: float = -1.0) -> list[float]:
    return [float(np.float32(i * scale + shift)) for i in range(n)]


def aligned_name() -> str:
    """A tensor name that makes the header length an exact multiple of 8, so
    the writer must add no padding at all."""
    for k in range(1, 16):
        name = "x" * k
        hdr = json.dumps(
            {name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
            separators=(",", ":"),
        )
        if len(hdr.encode()) % 8 == 0:
            return name
    raise AssertionError("no aligned name found")


CASES = [
    # The 120-byte worked example of formats/safetensors.md.
    {
        "name": "worked_example",
        "meta": {"format": "tinyllm"},
        "tensors": [{"name": "w", "shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0]}],
    },
    # Inserted out of order; a scalar, an empty tensor, a non-ASCII name,
    # -0.0 and the smallest float32 subnormal must survive byte for byte.
    {
        "name": "several_tensors",
        "meta": {"format": "tinyllm"},
        "tensors": [
            {"name": "b", "shape": [3], "values": [-0.0, 1.401298464324817e-45, 3.5]},
            {"name": "é", "shape": [1], "values": [7.0]},
            {"name": "a.weight", "shape": [2, 3], "values": ramp(6)},
            {"name": "z", "shape": [0, 3], "values": []},
            {"name": "A", "shape": [], "values": [42.0]},
        ],
    },
    # An empty metadata map is omitted, not written as {}.
    {
        "name": "no_metadata",
        "meta": {},
        "tensors": [{"name": "w", "shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0]}],
    },
    # Header already a multiple of 8: no padding byte at all.
    {
        "name": "no_padding",
        "meta": {},
        "tensors": [{"name": aligned_name(), "shape": [2], "values": [0.25, -0.25]}],
    },
    # Non-ASCII metadata is raw UTF-8, never \u escapes.
    {
        "name": "unicode_metadata",
        "meta": {"note": "héllo ✓"},
        "tensors": [{"name": "w", "shape": [1], "values": [1.0]}],
    },
    # The tracer tensor name and shape family, small: a log-probability table.
    {
        "name": "bigram_weight",
        "meta": {"format": "tinyllm"},
        "tensors": [
            {
                "name": "bigram.weight",
                "shape": [3, 3],
                "values": [
                    float(np.float32(np.log(p)))
                    for p in (
                        1 / 6,
                        3 / 6,
                        2 / 6,
                        2 / 5,
                        2 / 5,
                        1 / 5,
                        2 / 4,
                        1 / 4,
                        1 / 4,
                    )
                ],
            }
        ],
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        tensors = {
            t["name"]: np.array(t["values"], dtype=np.float32).reshape(t["shape"])
            for t in case["tensors"]
        }
        data = save(tensors, metadata=case["meta"] or None)
        p = OUT / f"{case['name']}.safetensors"
        p.write_bytes(data)
        rows.append((p, data))
    doc = {
        "generator": "course/oracle/L0.0/safetensors_golden.py",
        "library": f"safetensors=={safetensors.__version__}",
        "cases": CASES,
    }
    cj = OUT / "cases.json"
    cj.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    rows.append((cj, cj.read_bytes()))
    for p, data in rows:
        print(
            "\t".join(
                [
                    p.as_posix(),
                    hashlib.sha256(data).hexdigest(),
                    str(len(data)),
                    "course/oracle/L0.0/safetensors_golden.py",
                    f"safetensors=={safetensors.__version__}",
                    "-",
                    "Apache-2.0",
                ]
            )
        )


if __name__ == "__main__":
    main()
