# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L5.1 torch golden fixture (scaled dot-product attention).

For each case, float64 inputs q [.., Tq, d], k [.., Tk, d], v [.., Tk, dv], an
optional bool mask (True = may attend), an optional scale, and an upstream
gradient g; records torch.nn.functional.scaled_dot_product_attention's output
and the gradients of sum(out * g) for q, k, v (float64, autograd), and the
attention weights softmax(q k^T * scale + mask) computed by torch ops. Cases:

  plain      [2, 2, 4, 8] self-shaped, no mask
  causal     [2, 2, 5, 4], is_causal (upper left)
  cross      Tq 3, Tk 6, dv 5, key padding (lengths 6 and 4) as a [2, 1, 1, 6] mask
  scaled     [1, 3, 4, 6], scale 0.5, a random mask with every row open somewhere
  three_d    [2, 5, 4] (no head axis), causal

    uv run --offline --python 3.12 --script course/oracle/L5.1/sdpa_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

OUT = Path("course/fixtures/L5.1/sdpa_torch.npz")
gen = np.random.default_rng(20261051)


def case(arrs, name, qs, ks, vs, mask=None, scale=None, is_causal=False):
    q, k, v = (gen.standard_normal(s) for s in (qs, ks, vs))
    qt, kt, vt = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
    tm = None if mask is None else torch.from_numpy(mask)
    out = Fn.scaled_dot_product_attention(
        qt, kt, vt, attn_mask=tm, scale=scale, is_causal=is_causal
    )
    g = gen.standard_normal(out.shape)
    (out * torch.from_numpy(g)).sum().backward()
    sc = (1.0 / np.sqrt(qs[-1])) if scale is None else scale
    s = (qt.detach() @ kt.detach().transpose(-1, -2)) * sc
    if is_causal:
        mask = np.tril(np.ones((qs[-2], ks[-2]), dtype=bool))
    if mask is not None:
        s = s.masked_fill(~torch.from_numpy(mask), float("-inf"))
    w = torch.softmax(s, dim=-1)
    arrs.update(
        {
            f"{name}.q": q,
            f"{name}.k": k,
            f"{name}.v": v,
            f"{name}.g": g,
            f"{name}.out": out.detach().numpy(),
            f"{name}.weights": w.numpy(),
            f"{name}.grad.q": qt.grad.numpy(),
            f"{name}.grad.k": kt.grad.numpy(),
            f"{name}.grad.v": vt.grad.numpy(),
            f"{name}.scale": np.array(np.nan if scale is None else scale),
        }
    )
    if mask is not None:
        arrs[f"{name}.mask"] = mask


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    case(arrs, "plain", (2, 2, 4, 8), (2, 2, 4, 8), (2, 2, 4, 8))
    case(arrs, "causal", (2, 2, 5, 4), (2, 2, 5, 4), (2, 2, 5, 4), is_causal=True)
    pad = np.arange(6)[None, :] < np.array([6, 4])[:, None]
    case(
        arrs,
        "cross",
        (2, 2, 3, 4),
        (2, 2, 6, 4),
        (2, 2, 6, 5),
        mask=pad[:, None, None, :],
    )
    rm = gen.uniform(size=(1, 3, 4, 6)) < 0.6
    rm[..., 0] = True
    case(arrs, "scaled", (1, 3, 4, 6), (1, 3, 6, 6), (1, 3, 6, 6), mask=rm, scale=0.5)
    case(arrs, "three_d", (2, 5, 4), (2, 5, 4), (2, 5, 4), is_causal=True)
    meta = {
        "generator": "course/oracle/L5.1/sdpa_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L5.1/sdpa_torch.py\t"
        f"torch=={torch.__version__.split('+')[0]},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
