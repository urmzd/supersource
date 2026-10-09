# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.2.6"]
# ///
"""Maintainer generator for the M00.2 golden rotations.

The oracle is torch's complex arithmetic, the way Meta's Llama reference code
applies RoPE: view each interleaved pair (x[2i], x[2i+1]) as a complex number
(torch.view_as_complex), multiply by torch.polar(1, theta) = e^{i theta}, and
read the pairs back (torch.view_as_real). Inputs come from a seeded torch
generator and are stored with the outputs.

    uv run --script course/oracle/M00.2/rotation_golden.py

Run from the repo root and paste the printed row into course/fixtures/MANIFEST.tsv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

OUT = Path("course/fixtures/M00.2/rotation.json")


def rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    w = torch.polar(torch.ones_like(theta), theta)
    return torch.view_as_real(z * w).reshape(x.shape)


def case(
    name: str, shape: tuple[int, ...], theta_shape: tuple[int, ...], dtype, seed: int
) -> dict:
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(shape, generator=g, dtype=torch.float64) * 4 - 2).to(dtype)
    theta = (torch.rand(theta_shape, generator=g, dtype=torch.float64) * 20 - 10).to(
        dtype
    )
    out = rotate(x, theta)
    return {
        "name": name,
        "dtype": str(dtype).removeprefix("torch."),
        "x": x.tolist(),
        "theta": theta.tolist(),
        "out": out.tolist(),
    }


def main() -> None:
    cases = [
        case("rows_own_angles_f64", (3, 8), (3, 4), torch.float64, 1),
        case("one_angle_per_pair_f64", (2, 3, 6), (2, 3, 3), torch.float64, 2),
        case("rows_own_angles_f32", (4, 16), (4, 8), torch.float32, 3),
    ]
    doc = {
        "generator": "course/oracle/M00.2/rotation_golden.py",
        "library": f"torch=={torch.__version__}",
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M00.2/rotation_golden.py",
                f"torch=={torch.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
