"""Parity driver for `quant.fp8` (M09.4, Python): reads one JSON case per
stdin line ({fmt: 4 | 5, x_bits: float32 bit patterns, codes: uint8}) and
prints {"codes": encoded x, "values_bits": float32 bits of each decoded code,
-1 for NaN} from tinyllm.num.lowp."""

import json
import sys

import numpy as np

from tinyllm.num.lowp import f32_to_fp8_bits, fp8_bits_to_f32

for line in sys.stdin:
    if not line.strip():
        continue
    c = json.loads(line)
    fmt = {4: "e4m3", 5: "e5m2"}[int(c["fmt"])]
    x = np.asarray(c["x_bits"], dtype=np.uint32).view(np.float32)
    codes = f32_to_fp8_bits(x, fmt)
    vals = fp8_bits_to_f32(np.asarray(c["codes"], dtype=np.uint8), fmt).astype(
        np.float32
    )
    bits = vals.view(np.uint32).astype(np.int64)
    bits[np.isnan(vals)] = -1
    out = {
        "codes": [int(v) for v in np.asarray(codes).reshape(-1)],
        "values_bits": [int(v) for v in bits.reshape(-1)],
    }
    print(json.dumps(out, separators=(",", ":")), flush=True)
