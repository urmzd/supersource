# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "numpy==2.4.2"]
# ///
"""Maintainer generator for the M08.3 golden fixture.

Writes course/fixtures/M08.3/torch_vjp.npz: for each op, float64 inputs,
an upstream gradient g, the saved forward values the VJP takes (y, xhat,
rstd), and the input gradients torch.autograd computes for sum(g * f(x)).

    uv run --script course/oracle/M08.3/torch_vjp_golden.py

Run from the repo root, then update the fixture's row in
course/fixtures/MANIFEST.tsv (the script prints it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path("course/fixtures/M08.3/torch_vjp.npz")
torch.set_default_dtype(torch.float64)


def t(a: np.ndarray, grad: bool = True) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


def main() -> None:
    rng = np.random.default_rng(803)
    d: dict[str, np.ndarray] = {}

    # matmul, 2-D and with a broadcast batch axis on A only.
    for name, sa, sb in (
        ("matmul", (3, 4), (4, 5)),
        ("matmul_batched", (2, 3, 4), (4, 5)),
    ):
        A, B = rng.normal(size=sa), rng.normal(size=sb)
        TA, TB = t(A), t(B)
        Y = TA @ TB
        g = rng.normal(size=tuple(Y.shape))
        Y.backward(torch.tensor(g))
        d |= {
            f"{name}/A": A,
            f"{name}/B": B,
            f"{name}/g": g,
            f"{name}/dA": TA.grad.numpy(),
            f"{name}/dB": TB.grad.numpy(),
        }

    # softmax and log_softmax along the last axis and along axis 0.
    for name, shape, axis in (
        ("softmax_last", (3, 6), -1),
        ("softmax_axis0", (4, 3), 0),
    ):
        x = rng.normal(scale=2.0, size=shape)
        g = rng.normal(size=shape)
        for fn, key in ((F.softmax, "softmax"), (F.log_softmax, "log_softmax")):
            tx = t(x)
            y = fn(tx, dim=axis)
            y.backward(torch.tensor(g))
            d |= {
                f"{name}/{key}/y": y.detach().numpy(),
                f"{name}/{key}/dx": tx.grad.numpy(),
            }
        d |= {f"{name}/x": x, f"{name}/g": g, f"{name}/axis": np.array(axis)}

    # LayerNorm over the last axis, eps 1e-5.
    x = rng.normal(loc=0.5, scale=1.5, size=(2, 3, 8))
    gamma, beta = rng.normal(size=8), rng.normal(size=8)
    g = rng.normal(size=x.shape)
    tx, tg, tb = t(x), t(gamma), t(beta)
    y = F.layer_norm(tx, (8,), tg, tb, eps=1e-5)
    y.backward(torch.tensor(g))
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
    d |= {
        "layernorm/x": x,
        "layernorm/gamma": gamma,
        "layernorm/beta": beta,
        "layernorm/g": g,
        "layernorm/xhat": (x - mu) * rstd,
        "layernorm/rstd": rstd[..., 0],
        "layernorm/dx": tx.grad.numpy(),
        "layernorm/dgamma": tg.grad.numpy(),
        "layernorm/dbeta": tb.grad.numpy(),
    }

    # RMSNorm over the last axis, eps 1e-6.
    x = rng.normal(scale=1.5, size=(4, 8))
    w = rng.normal(size=8)
    g = rng.normal(size=x.shape)
    tx, tw = t(x), t(w)
    y = F.rms_norm(tx, (8,), tw, eps=1e-6)
    y.backward(torch.tensor(g))
    rstd = 1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)
    d |= {
        "rmsnorm/x": x,
        "rmsnorm/w": w,
        "rmsnorm/g": g,
        "rmsnorm/rstd": rstd,
        "rmsnorm/dx": tx.grad.numpy(),
        "rmsnorm/dw": tw.grad.numpy(),
    }

    # Cross-entropy with ignored rows (mean over the valid rows).
    logits = rng.normal(scale=3.0, size=(7, 5))
    targets = np.array([0, 4, -100, 2, 2, -100, 1], dtype=np.int64)
    tz = t(logits)
    loss = F.cross_entropy(
        tz, torch.tensor(targets), ignore_index=-100, reduction="mean"
    )
    loss.backward()
    d |= {
        "cross_entropy/logits": logits,
        "cross_entropy/targets": targets,
        "cross_entropy/dlogits": tz.grad.numpy(),
    }

    meta = {
        "generator": "course/oracle/M08.3/torch_vjp_golden.py",
        "oracle_versions": f"torch=={torch.__version__} numpy=={np.__version__}",
        "seed": 803,
    }
    d["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        np.savez(f, **d)
    raw = OUT.read_bytes()
    print(
        "\t".join(
            [
                str(OUT),
                hashlib.sha256(raw).hexdigest(),
                str(len(raw)),
                "course/oracle/M08.3/torch_vjp_golden.py",
                meta["oracle_versions"],
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
