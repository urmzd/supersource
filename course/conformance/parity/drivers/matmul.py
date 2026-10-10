"""Parity driver for `matmul` (M03.1, Python): reads one JSON case per stdin
line ({m, k, n, a, b}: row-major float32 operands) and prints the row-major
product from tinyllm.linalg.matmul.matmul_f32 (alpha 1, beta 0), as the C
driver does for L9.1."""

import json
import sys

import numpy as np

from tinyllm.linalg.matmul import matmul_f32

for line in sys.stdin:
    if not line.strip():
        continue
    c = json.loads(line)
    m, k, n = int(c["m"]), int(c["k"]), int(c["n"])
    a = np.asarray(c["a"], dtype=np.float32).reshape(m, k)
    b = np.asarray(c["b"], dtype=np.float32).reshape(k, n)
    out = np.asarray(matmul_f32(a, b), dtype=np.float32).reshape(-1)
    print(json.dumps([float(x) for x in out]), flush=True)
