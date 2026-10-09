"""Fuzz inputs for `rng`: gen/rng.py <seed> <n> prints n JSON cases."""

import json
import random
import sys

rng = random.Random(int(sys.argv[1]))
for _ in range(int(sys.argv[2])):
    print(
        json.dumps(
            {
                "seed": rng.getrandbits(64),
                "seq": rng.getrandbits(63),
                "n_u32": 64,
                "n_uniform": 8,
                "n_normal": 8,
            }
        )
    )
