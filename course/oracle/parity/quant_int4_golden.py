# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6"]
# ///
"""Maintainer generator for the `quant.int4` parity suite (DESIGN 5.8).

Each case is a float32 weight W [rows, cols] drawn from the frozen PCG32
(seed 2026, one child stream per case) and its int4 group quantization by
L8.5's rule (contracts/py/tinyllm/infer/quant.pyi) in the byte layout of
course/contracts/formats/safetensors.md:

    s = float16(amax(group) / 7)                       one per (row, group)
    q = clip(rint(float32(w) / float32(s)), -8, 7)     round half to even; q = 0 when s = 0
    byte b of row r holds columns 2b (low nibble) and 2b + 1 (high nibble)

The expected output of every implementation is {"packed": [bytes, row-major],
"scales_f16": [f16 bit patterns]}. The Python implementation (L8.5)
quantizes `w`. The C implementation (L9.5) cannot quantize: its driver reads
`packed` and `scales_f16` from the input, dequantizes every weight with
tl_matmul_q4_f32 on identity activations (exact: one nonzero product per
output), recovers q = w / s, and packs again, so it reproduces the bytes
only if the kernel decodes the layout exactly as L8.5 encodes it.

    uv run --script course/oracle/parity/quant_int4_golden.py

Run from the repo root. It rewrites course/fixtures/parity/quant_int4.json
and prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "course" / "tests"))
from _lib.pcg32 import PCG32  # noqa: E402

OUT = ROOT / "course" / "fixtures" / "parity" / "quant_int4.json"
SHAPES = [(1, 2, 2), (2, 8, 4), (3, 32, 32), (4, 64, 32), (2, 64, 64), (5, 48, 16)]


def quantize(w: np.ndarray, group: int):
    n, k = w.shape
    g = w.reshape(n, k // group, group)
    scale = (np.abs(g).max(axis=2) / np.float32(7)).astype(np.float16)
    s32 = scale.astype(np.float32)
    q = np.clip(
        np.rint(g / np.where(s32 == 0, np.float32(1), s32)[:, :, None]), -8, 7
    ).astype(np.int8)
    q = q.reshape(n, k)
    u = (q.astype(np.int16) & 0xF).astype(np.uint8)
    packed = u[:, 0::2] | (u[:, 1::2] << 4)
    deq = (q.astype(np.float32) * np.repeat(s32, group, axis=1)).astype(np.float32)
    return packed, scale, deq


def main() -> None:
    root = PCG32(2026, 54)
    cases = []
    for i, (n, k, group) in enumerate(SHAPES):
        g = root.split(i)
        w = np.array(
            [g.normal() * 0.1 for _ in range(n * k)], dtype=np.float32
        ).reshape(n, k)
        packed, scale, _ = quantize(w, group)
        cases.append(
            {
                "name": f"{n}x{k}_g{group}",
                "input": {
                    "rows": n,
                    "cols": k,
                    "group": group,
                    "w": [float(v) for v in w.reshape(-1)],
                    "packed": [int(b) for b in packed.reshape(-1)],
                    "scales_f16": [int(b) for b in scale.view(np.uint16).reshape(-1)],
                },
                "output": {
                    "packed": [int(b) for b in packed.reshape(-1)],
                    "scales_f16": [int(b) for b in scale.view(np.uint16).reshape(-1)],
                },
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generator": "course/oracle/parity/quant_int4_golden.py",
                "numpy": np.__version__,
                "cases": cases,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.relative_to(ROOT).as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/parity/quant_int4_golden.py",
                f"numpy=={np.__version__}",
                "formats/safetensors.md",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
