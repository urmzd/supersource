"""Parity driver for `quant.int4` (L8.5, Python): reads one JSON case per
stdin line ({rows, cols, group, w: row-major floats}) and prints
{"packed": [bytes, row-major], "scales_f16": [f16 bit patterns]} from
tinyllm.infer.quant.quantize_int4_group. The C side (L9.5) consumes exactly
these bytes, so the suite compares them exactly."""

import json
import sys

import numpy as np

from tinyllm.infer.quant import quantize_int4_group

for line in sys.stdin:
    if not line.strip():
        continue
    c = json.loads(line)
    w = np.asarray(c["w"], dtype=np.float32).reshape(int(c["rows"]), int(c["cols"]))
    q = quantize_int4_group(w, int(c["group"]))
    out = {
        "packed": [int(b) for b in np.asarray(q.packed, dtype=np.uint8).reshape(-1)],
        "scales_f16": [
            int(s)
            for s in np.asarray(q.scales, dtype=np.float16).view(np.uint16).reshape(-1)
        ],
    }
    print(json.dumps(out, separators=(",", ":")), flush=True)
