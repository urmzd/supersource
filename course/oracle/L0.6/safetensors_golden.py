# /// script
# requires-python = ">=3.12"
# dependencies = ["safetensors==0.8.0", "torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L0.6 safetensors golden fixtures.

Writes course/fixtures/L0.6/safetensors/<case>.safetensors with the pinned
`safetensors` library (its torch front end: numpy has no bfloat16 or float8)
and cases.json with, per tensor, what the learner passes in (float32 values,
or raw ints, plus the dtype name) and the float32 values the file decodes to.
torch's float32 -> bfloat16 and float32 -> float8_e4m3fn casts round to
nearest, ties to even, which is the rule the contract fixes.

    uv run --offline --python 3.12 --script course/oracle/L0.6/safetensors_golden.py

Run from the repo root, then update the MANIFEST.tsv rows it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

OUT = Path("course/fixtures/L0.6/safetensors")
TORCH = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
}


def f32(xs):
    return [float(np.float32(x)) for x in xs]


# name -> (dtype, shape, values). Float dtypes take float32 values (rounded by
# torch's cast for F16, BF16, F8_E4M3); integer dtypes take the ints.
CASES = {
    # The chapter's worked example: formats/safetensors.md's tensor, as BF16.
    "worked_bf16": (
        {"format": "tinyllm"},
        {"w": ("BF16", [2, 2], [1.0, 2.0, 3.0, 4.0])},
    ),
    # Every dtype in one file. Names run against the dtype order, so name
    # order alone gets the layout wrong.
    "mixed": (
        {"format": "tinyllm"},
        {
            "a_u8": ("U8", [3], [0, 7, 255]),
            "b_i8": ("I8", [2], [-128, 127]),
            "c_f8": ("F8_E4M3", [4], f32([1.0, -0.5, 448.0, 0.001953125])),
            "d_f16": ("F16", [2, 2], f32([1.0, -2.5, 65504.0, 5.960464477539063e-08])),
            "e_bf16": ("BF16", [3], f32([1.0, -0.15625, 3.0e38])),
            "f_i32": ("I32", [1, 2], [-2147483648, 2147483647]),
            "g_f32": ("F32", [2], f32([1.5, -2.25])),
            "h_scalar": ("F32", [], f32([0.5])),
            "i_empty": ("F16", [0], []),
        },
    ),
    # Float32 values that are not BF16 values: round to nearest, ties to even.
    "bf16_rounding": (
        {"format": "tinyllm"},
        {
            "x": (
                "BF16",
                [8],
                f32([1.00390625, 1.01171875, 0.1, -0.1, 3.14159, 1e-40, 65504.0, -0.0]),
            )
        },
    ),
    # Float32 values that are not F8_E4M3 values, subnormal ties included.
    "f8_rounding": (
        {"format": "tinyllm"},
        {
            "q": (
                "F8_E4M3",
                [10],
                f32(
                    [
                        0.3,
                        1.0625,
                        1.1875,
                        0.0009765625,
                        0.0029296875,
                        240.0,
                        250.0,
                        -17.0,
                        -0.0,
                        0.0,
                    ]
                ),
            )
        },
    ),
    # No metadata: __metadata__ is omitted entirely.
    "no_metadata": (
        {},
        {"k": ("I32", [3], [1, 2, 3]), "h": ("F16", [1], f32([0.333251953125]))},
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "generator": "course/oracle/L0.6/safetensors_golden.py",
        "safetensors": safetensors.__version__,
        "torch": torch.__version__,
        "cases": {},
    }
    for case, (meta, spec) in CASES.items():
        tensors, rows = {}, []
        for name, (dt, shape, values) in spec.items():
            if dt in ("I8", "U8", "I32"):
                t = torch.tensor(values, dtype=TORCH[dt]).reshape(shape)
            else:
                t = (
                    torch.tensor(values, dtype=torch.float32)
                    .reshape(shape)
                    .to(TORCH[dt])
                )
            tensors[name] = t
            decoded = (
                t.to(torch.float32).reshape(-1).tolist()
                if dt not in ("I8", "U8", "I32")
                else values
            )
            rows.append(
                {
                    "name": name,
                    "dtype": dt,
                    "shape": shape,
                    "values": values,
                    "decoded": [float(x) for x in decoded]
                    if dt not in ("I8", "U8", "I32")
                    else values,
                }
            )
        path = OUT / f"{case}.safetensors"
        save_file(tensors, str(path), metadata=meta or None)
        doc["cases"][case] = {"file": path.name, "meta": meta, "tensors": rows}
    (OUT / "cases.json").write_text(json.dumps(doc, indent=1) + "\n")
    for p in sorted(OUT.iterdir()):
        b = p.read_bytes()
        print(f"{p}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}")


if __name__ == "__main__":
    main()
