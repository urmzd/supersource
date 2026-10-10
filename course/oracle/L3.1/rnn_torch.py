# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L3.1 torch golden fixture (vanilla RNN).

torch.nn.RNN(nonlinearity="tanh") in float64 with random weights: the
outputs for a random input and initial state, and the gradients of
sum(out * g) + sum(h_n * gn) for a random upstream g, gn, with respect to the
input, the initial state, and every parameter. torch stores W_ih [H, D] and
two biases; the course's manual RNN uses Wxh = W_ih^T, Whh = W_hh^T, and
bh = b_ih + b_hh, so the test maps between the two.

    uv run --offline --python 3.12 --script course/oracle/L3.1/rnn_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/L3.1/rnn_torch.npz")
torch.manual_seed(20261013)
T, B, D, H = 5, 2, 3, 4


def main() -> None:
    rnn = torch.nn.RNN(D, H, nonlinearity="tanh", dtype=torch.float64)
    with torch.no_grad():
        for p in rnn.parameters():
            p.uniform_(-0.8, 0.8)
    x = torch.randn(T, B, D, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(1, B, H, dtype=torch.float64, requires_grad=True) * 0.5
    h0.retain_grad()
    out, hn = rnn(x, h0)
    g = torch.randn(T, B, H, dtype=torch.float64)
    gn = torch.randn(1, B, H, dtype=torch.float64)
    (out * g).sum().add((hn * gn).sum()).backward()
    arrs = {
        "x": x.detach().numpy(),
        "h0": h0.detach().numpy()[0],
        "out": out.detach().numpy(),
        "g": g.numpy(),
        "gn": gn.numpy()[0],
        "gx": x.grad.numpy(),
        "gh0": h0.grad.numpy()[0],
    }
    for name, p in rnn.named_parameters():
        arrs[f"p_{name}"] = p.detach().numpy()
        arrs[f"gp_{name}"] = p.grad.numpy()
    meta = {
        "generator": "course/oracle/L3.1/rnn_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": 20261013,
        "shapes": {"T": T, "B": B, "D": D, "H": H},
    }
    buf = io.BytesIO()
    np.savez(buf, __meta__=np.array(json.dumps(meta)), **arrs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(buf.getvalue())
    b = buf.getvalue()
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L3.1/rnn_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
