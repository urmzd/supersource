# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the L3.3 torch golden fixture (GRU).

float32 throughout, weights drawn U(-0.5, 0.5). Cases:
  cell     torch.nn.GRUCell(3, 4) on a batch of 2 with a given h
  one      torch.nn.GRU(3, 4) over T = 5, B = 3 with a given h0
  two      torch.nn.GRU(3, 4, num_layers=2), zero initial state
  packed   torch.nn.GRU(3, 4) over sequences of lengths [5, 2, 3], through
           pack_padded_sequence(enforce_sorted=False) and pad_packed_sequence
Each records the outputs and the gradients of sum(out * g) + sum(h_n * gh)
(cell: sum(h' g)) for random upstream gradients, with respect to the input,
the state, and every parameter. `__meta__` also records torch's state_dict
key order for a 2-layer GRU.

    uv run --offline --python 3.12 --script course/oracle/L3.3/gru_torch.py

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

OUT = Path("course/fixtures/L3.3/gru_torch.npz")
torch.manual_seed(20261015)
D, H = 3, 4


def init(m: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in m.parameters():
            p.uniform_(-0.5, 0.5)


def leaf(*shape) -> torch.Tensor:
    return (torch.randn(*shape) * 0.5).requires_grad_(True)


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    cases = []

    def save_params(key: str, m: torch.nn.Module) -> list[str]:
        names = []
        for name, p in m.named_parameters():
            arrs[f"{key}_p_{name}"] = p.detach().numpy().copy()
            arrs[f"{key}_gp_{name}"] = p.grad.numpy().copy()
            names.append(name)
        return names

    cell = torch.nn.GRUCell(D, H)
    init(cell)
    x, h = leaf(2, D), leaf(2, H)
    h2 = cell(x, h)
    gh = torch.randn(2, H)
    (h2 * gh).sum().backward()
    arrs["cell_x"], arrs["cell_gx"] = x.detach().numpy(), x.grad.numpy()
    arrs["cell_h"], arrs["cell_gh"] = h.detach().numpy(), h.grad.numpy()
    arrs["cell_h2"], arrs["cell_uh"] = h2.detach().numpy(), gh.numpy()
    cases.append({"name": "cell", "params": save_params("cell", cell)})

    def run(key: str, layers: int, T: int, B: int, lengths, with_state: bool) -> None:
        m = torch.nn.GRU(D, H, num_layers=layers)
        init(m)
        x = leaf(T, B, D)
        h0 = leaf(layers, B, H) if with_state else None
        if lengths is None:
            out, hn = m(x, h0)
        else:
            packed = pack_padded_sequence(
                x, torch.tensor(lengths), enforce_sorted=False
            )
            po, hn = m(packed, h0)
            out, _ = pad_packed_sequence(po, total_length=T)
        g, gh = torch.randn(T, B, H), torch.randn(layers, B, H)
        ((out * g).sum() + (hn * gh).sum()).backward()
        arrs[f"{key}_x"], arrs[f"{key}_gx"] = x.detach().numpy(), x.grad.numpy()
        if with_state:
            arrs[f"{key}_h0"], arrs[f"{key}_gh0"] = h0.detach().numpy(), h0.grad.numpy()
        arrs[f"{key}_out"], arrs[f"{key}_hn"] = (
            out.detach().numpy(),
            hn.detach().numpy(),
        )
        arrs[f"{key}_g"], arrs[f"{key}_uh"] = g.numpy(), gh.numpy()
        if lengths is not None:
            arrs[f"{key}_lengths"] = np.asarray(lengths, dtype=np.int64)
        cases.append(
            {
                "name": key,
                "layers": layers,
                "state": with_state,
                "packed": lengths is not None,
                "params": save_params(key, m),
            }
        )

    run("one", 1, 5, 3, None, True)
    run("two", 2, 4, 2, None, False)
    run("packed", 1, 5, 3, [5, 2, 3], True)

    meta = {
        "generator": "course/oracle/L3.3/gru_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": 20261015,
        "cases": cases,
        "gru2_state_dict_keys": list(torch.nn.GRU(D, H, num_layers=2).state_dict()),
        "cell_state_dict_keys": list(torch.nn.GRUCell(D, H).state_dict()),
    }
    buf = io.BytesIO()
    np.savez(buf, __meta__=np.array(json.dumps(meta)), **arrs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L3.3/gru_torch.py\t"
        f"torch=={torch.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
