"""Course benchmark for M90.1: scale() on 1,000 floats."""
import json
import time

from tinyllm.demo.scale import scale

xs = [float(i) for i in range(1000)]
best = float("inf")
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(100):
        scale(xs, 2.0)
    best = min(best, (time.perf_counter() - t0) / 100)
print(json.dumps({"calls_per_s": 1.0 / best}))
