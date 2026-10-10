# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L2.2 torch golden fixture (the NPLM).

Bengio's NPLM written directly in torch from the paper's equation,
    y = b + W x + U tanh(d + H x),   x = [C[w_1], ..., C[w_{n-1}]] (oldest first),
with the course's parameter names (emb, hidden, out, direct). For each case:
the parameters (float32, from a seeded numpy generator; copied into the
learner's model by state_dict name), a batch of windows (ids repeat inside
a window and across rows, so embedding gradients must accumulate), the
targets, torch's logits, its mean cross-entropy, and the gradient of that
loss for every parameter. Cases: with and without the direct path.

    uv run --offline --python 3.12 --script course/oracle/L2.2/nplm_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path("course/fixtures/L2.2/nplm_torch.npz")
V, CTX, M, H, B = 11, 3, 4, 5, 6


class NPLM(nn.Module):
    def __init__(self, direct: bool) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, M)
        self.hidden = nn.Linear(CTX * M, H)
        self.out = nn.Linear(H, V)
        self.direct = nn.Linear(CTX * M, V, bias=False) if direct else None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids).reshape(ids.shape[0], CTX * M)
        y = self.out(torch.tanh(self.hidden(x)))
        return y + self.direct(x) if self.direct is not None else y


def case(direct: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
    model = NPLM(direct)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(
                torch.from_numpy(
                    rng.normal(0.0, 0.5, size=tuple(p.shape)).astype(np.float32)
                )
            )
    ids = rng.integers(0, V, size=(B, CTX))
    ids[0] = [3, 3, 7]  # a repeated id inside one window
    ids[1, 0] = 3  # and across rows
    targets = rng.integers(0, V, size=B)
    logits = model(torch.from_numpy(ids))
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(targets))
    loss.backward()
    out = {
        "ids": ids.astype(np.int64),
        "targets": targets.astype(np.int64),
        "logits": logits.detach().numpy().astype(np.float32),
        "loss": np.array(loss.item(), dtype=np.float32),
    }
    for name, p in model.named_parameters():
        out[f"param.{name}"] = p.detach().numpy().astype(np.float32)
        out[f"grad.{name}"] = p.grad.numpy().astype(np.float32)
    return out


def main() -> None:
    rng = np.random.default_rng(20261009)
    arrays: dict[str, np.ndarray] = {}
    for tag, direct in (("direct", True), ("nodirect", False)):
        for k, v in case(direct, rng).items():
            arrays[f"{tag}/{k}"] = v
    meta = {
        "generator": "course/oracle/L2.2/nplm_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": 20261009,
        "V": V,
        "context": CTX,
        "d_emb": M,
        "d_hidden": H,
        "batch": B,
        "state_dict_order": [n for n, _ in NPLM(True).named_parameters()],
    }
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **arrays)
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/L2.2/nplm_torch.py",
                f"torch=={torch.__version__},numpy=={np.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
