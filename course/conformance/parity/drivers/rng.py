"""Parity driver for `rng` (M06.3, Python): reads one JSON case per stdin line
({seed, seq, n_u32, n_uniform, n_normal}) and prints one JSON object per line:
{"u32": [...], "uniform": [...], "normal": [...]}, each list drawn from a
fresh tinyllm.num.rng.PCG32(seed, seq).

Normals follow spec/pcg32.md: Box-Muller over two consecutive uniforms,
r = sqrt(-2 ln(1 - u1)), cosine first, the sine kept as the spare for the
next call. M06.3 owns the uniforms; the transform is written here so the
suite needs no other module.
"""

import json
import math
import sys

from tinyllm.num.rng import PCG32


def normals(g: PCG32, n: int) -> list[float]:
    out: list[float] = []
    spare = None
    while len(out) < n:
        if spare is not None:
            out.append(spare)
            spare = None
            continue
        u1, u2 = g.uniform(), g.uniform()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        out.append(r * math.cos(2.0 * math.pi * u2))
        spare = r * math.sin(2.0 * math.pi * u2)
    return out


for line in sys.stdin:
    if not line.strip():
        continue
    c = json.loads(line)
    seed, seq = int(c["seed"]), int(c.get("seq", 54))
    g = PCG32(seed, seq)
    u32 = [g.next_u32() for _ in range(int(c.get("n_u32", 0)))]
    g = PCG32(seed, seq)
    uni = [g.uniform() for _ in range(int(c.get("n_uniform", 0)))]
    nor = normals(PCG32(seed, seq), int(c.get("n_normal", 0)))
    print(json.dumps({"u32": u32, "uniform": uni, "normal": nor}), flush=True)
