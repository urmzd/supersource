# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6"]
# ///
"""Maintainer generator for the `flash.fwd` parity suite (DESIGN 5.8).

Each case is q [B, H, Tq, D], k and v [B, Hkv, Tk, D] in float32 drawn from
the frozen PCG32 (seed 2027, one child stream per case) as uniforms in
[-1, 1), with the flags of tinyllm/attention.h (scale, q_offset, causal,
window). The oracle is naive attention in float64 from those float32
values: query i at position q_offset + i sees key j when (not causal or
j <= p) and (window <= 0 or j > p - window); o = softmax(scale q.k) v over
the visible keys and lse = log(sum exp(scale q.k)). Every row sees at least
one key, so every value is finite (JSON has no infinities). Sinks are not
part of this suite (the L9.3 course tests cover them).

    uv run --script course/oracle/parity/flash_fwd_golden.py

Run from the repo root. It rewrites course/fixtures/parity/flash_fwd.json
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

OUT = ROOT / "course" / "fixtures" / "parity" / "flash_fwd.json"
# B, H, Hkv, Tq, Tk, D, q_offset, causal, window
CASES = [
    (1, 1, 1, 1, 1, 4, 0, 1, 0),
    (1, 2, 1, 5, 5, 8, 0, 1, 0),
    (2, 4, 2, 3, 11, 8, 8, 1, 0),
    (1, 3, 3, 9, 9, 16, 0, 1, 4),
    (1, 2, 2, 4, 7, 8, 0, 0, 0),
    (1, 6, 2, 6, 20, 12, 14, 1, 5),
]


def main() -> None:
    root = PCG32(2027, 54)
    cases = []
    for c, (B, H, Hkv, Tq, Tk, D, q_offset, causal, window) in enumerate(CASES):
        g = root.split(c)

        def draw(n):
            return np.array([g.uniform() * 2 - 1 for _ in range(n)], dtype=np.float32)

        q = draw(B * H * Tq * D).reshape(B, H, Tq, D)
        k = draw(B * Hkv * Tk * D).reshape(B, Hkv, Tk, D)
        v = draw(B * Hkv * Tk * D).reshape(B, Hkv, Tk, D)
        scale = float(np.float32(D**-0.5))
        o = np.zeros((B, H, Tq, D))
        lse = np.zeros((B, H, Tq))
        for b in range(B):
            for h in range(H):
                kv = h // (H // Hkv)
                for i in range(Tq):
                    p = q_offset + i
                    vis = [
                        j
                        for j in range(Tk)
                        if (not causal or j <= p) and (window <= 0 or j > p - window)
                    ]
                    assert vis, "every row must see a key"
                    s = scale * (
                        k[b, kv, vis].astype(np.float64) @ q[b, h, i].astype(np.float64)
                    )
                    m = s.max()
                    e = np.exp(s - m)
                    o[b, h, i] = (e / e.sum()) @ v[b, kv, vis].astype(np.float64)
                    lse[b, h, i] = m + np.log(e.sum())
        cases.append(
            {
                "name": f"B{B}H{H}kv{Hkv}_Tq{Tq}Tk{Tk}D{D}_off{q_offset}_c{causal}_w{window}",
                "input": {
                    "B": B,
                    "H": H,
                    "Hkv": Hkv,
                    "Tq": Tq,
                    "Tk": Tk,
                    "D": D,
                    "scale": scale,
                    "q_offset": q_offset,
                    "causal": causal,
                    "window": window,
                    "q": [float(x) for x in q.reshape(-1)],
                    "k": [float(x) for x in k.reshape(-1)],
                    "v": [float(x) for x in v.reshape(-1)],
                },
                "output": {
                    "o": [float(x) for x in o.reshape(-1)],
                    "lse": [float(x) for x in lse.reshape(-1)],
                },
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generator": "course/oracle/parity/flash_fwd_golden.py",
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
                "course/oracle/parity/flash_fwd_golden.py",
                f"numpy=={np.__version__}",
                "-",
                "Apache-2.0",
            ]
        )
    )


if __name__ == "__main__":
    main()
