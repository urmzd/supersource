"""Parity driver: the sum through M90.1's scale (k = 1), then Python's fsum."""
import json
import math
import sys

from tinyllm.demo.scale import scale

for line in sys.stdin:
    case = json.loads(line)
    print(json.dumps(math.fsum(scale(case["xs"], 1.0))), flush=True)
