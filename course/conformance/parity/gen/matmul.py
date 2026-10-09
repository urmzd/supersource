"""Fuzz inputs for `matmul`: gen/matmul.py <seed> <n> prints n JSON cases.
Shapes up to 9 x 40 x 9, float32-exact values in [-4, 4)."""

import json
import random
import struct
import sys


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


rng = random.Random(int(sys.argv[1]))
for _ in range(int(sys.argv[2])):
    m, k, n = rng.randint(1, 9), rng.randint(1, 40), rng.randint(1, 9)
    a = [f32(rng.uniform(-4, 4)) for _ in range(m * k)]
    b = [f32(rng.uniform(-4, 4)) for _ in range(k * n)]
    print(json.dumps({"m": m, "k": k, "n": n, "a": a, "b": b}))
