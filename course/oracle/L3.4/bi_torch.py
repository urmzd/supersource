# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L3.4 torch golden fixture (bidirectional RNNs).

torch.nn.GRU(3, 4, bidirectional=True) and torch.nn.LSTM(3, 4,
bidirectional=True), float32, weights U(-0.5, 0.5), over a padded batch with
lengths [5, 2, 4] (and a second GRU run with every length 5), through
pack_padded_sequence(enforce_sorted=False) and pad_packed_sequence. Records
the [T, B, 8] output (forward half, then backward half, padding 0) and the
gradients of sum(out * g) with respect to the input and every parameter. The
course test loads the *_l0 weights into the forward module and the
*_l0_reverse weights into the backward one.

    uv run --offline --python 3.12 --script course/oracle/L3.4/bi_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

OUT = Path("course/fixtures/L3.4/bi_torch.npz")
torch.manual_seed(20261016)
D, H, T, B = 3, 4, 5, 3


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    cases = []
    for key, cls, lengths in [
        ("gru", torch.nn.GRU, [5, 2, 4]),
        ("lstm", torch.nn.LSTM, [5, 2, 4]),
        ("gru_full", torch.nn.GRU, [5, 5, 5]),
    ]:
        m = cls(D, H, bidirectional=True)
        with torch.no_grad():
            for p in m.parameters():
                p.uniform_(-0.5, 0.5)
        x = (torch.randn(T, B, D) * 0.7).requires_grad_(True)
        packed = pack_padded_sequence(x, torch.tensor(lengths), enforce_sorted=False)
        po, _ = m(packed)
        out, _ = pad_packed_sequence(po, total_length=T)
        g = torch.randn(T, B, 2 * H)
        (out * g).sum().backward()
        arrs[f"{key}_x"], arrs[f"{key}_gx"] = x.detach().numpy(), x.grad.numpy()
        arrs[f"{key}_out"], arrs[f"{key}_g"] = out.detach().numpy(), g.numpy()
        arrs[f"{key}_lengths"] = np.asarray(lengths, dtype=np.int64)
        names = []
        for name, p in m.named_parameters():
            arrs[f"{key}_p_{name}"] = p.detach().numpy().copy()
            arrs[f"{key}_gp_{name}"] = p.grad.numpy().copy()
            names.append(name)
        cases.append({"name": key, "cell": cls.__name__, "params": names})
    meta = {
        "generator": "course/oracle/L3.4/bi_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": 20261016,
        "cases": cases,
    }
    buf = io.BytesIO()
    np.savez(buf, __meta__=np.array(json.dumps(meta)), **arrs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L3.4/bi_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
