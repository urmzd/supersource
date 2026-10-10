# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1"]
# ///
"""Maintainer generator for the M07.3 goldens: torch's gain table and fan
rules, and the standard deviations torch's initializers target.

    uv run --offline --script course/oracle/M07.3/init_golden.py

Run from the repo root. It prints the MANIFEST.tsv row.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import init

OUT = Path("course/fixtures/M07.3/init_torch.json")


def main() -> None:
    gains = []
    for nl, param in (
        ("linear", None),
        ("conv1d", None),
        ("conv2d", None),
        ("conv3d", None),
        ("sigmoid", None),
        ("tanh", None),
        ("relu", None),
        ("leaky_relu", None),
        ("leaky_relu", 0.2),
        ("leaky_relu", 0.0),
        ("selu", None),
    ):
        gains.append(
            {"nonlinearity": nl, "param": param, "gain": init.calculate_gain(nl, param)}
        )
    fans = []
    for shape in (
        (4, 3),
        (3, 4),
        (512, 128),
        (1, 7),
        (8, 4, 3, 3),
        (16, 3, 5),
        (2, 6, 1, 1, 4),
    ):
        fi, fo = init._calculate_fan_in_and_fan_out(torch.empty(shape))
        t = torch.empty(shape)
        g = 1.7
        fans.append(
            {
                "shape": list(shape),
                "fan_in": fi,
                "fan_out": fo,
                # the bound and standard deviations torch's initializers use
                "xavier_uniform_bound": g * math.sqrt(6.0 / (fi + fo)),
                "xavier_gain": g,
                "xavier_normal_std": g * math.sqrt(2.0 / (fi + fo)),
                "kaiming_relu_fan_in_std": init.calculate_gain("relu") / math.sqrt(fi),
                "kaiming_relu_fan_out_std": init.calculate_gain("relu") / math.sqrt(fo),
                "kaiming_tanh_fan_in_std": init.calculate_gain("tanh") / math.sqrt(fi),
            }
        )
        del t
    doc = {
        "generator": "course/oracle/M07.3/init_golden.py",
        "torch": torch.__version__,
        "gains": gains,
        "fans": fans,
    }
    data = (json.dumps(doc, indent=1) + "\n").encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M07.3/init_golden.py\ttorch=={torch.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
