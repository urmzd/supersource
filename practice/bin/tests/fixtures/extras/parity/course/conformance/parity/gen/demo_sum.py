"""Fuzz inputs for demo.sum: gen.py <seed> <n> prints n JSON cases."""
import json
import random
import sys

rng = random.Random(int(sys.argv[1]))
for _ in range(int(sys.argv[2])):
    xs = [round(rng.uniform(-100, 100), 3) for _ in range(rng.randrange(0, 20))]
    print(json.dumps({"xs": xs}))
