"""The float64 oracle for `matmul` fuzz cases: exact-rounded dot products
(math.fsum) of the float32 inputs, one JSON line of M*N values per case."""

import json
import math
import sys

for line in sys.stdin:
    c = json.loads(line)
    m, k, n, a, b = c["m"], c["k"], c["n"], c["a"], c["b"]
    print(
        json.dumps(
            [
                math.fsum(a[i * k + p] * b[p * n + j] for p in range(k))
                for i in range(m)
                for j in range(n)
            ]
        )
    )
