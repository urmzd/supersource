"""Parity driver for `bloom`, implementation `python-via-pyo3` (ds.08): reads
one case per stdin line ({"n", "p", "insert": [hex], "probe": [hex]}) and
prints {"m", "k", "bytes", "contains"} for the filter tinyllm_rs.Bloom builds:
m and k read back from the formats/bloom.md header of to_bytes().
"""

import json
import sys

import tinyllm_rs

for line in sys.stdin:
    if not line.strip():
        continue
    c = json.loads(line)
    b = tinyllm_rs.Bloom.with_rate(int(c["n"]), float(c["p"]))
    for it in c["insert"]:
        b.insert(bytes.fromhex(it))
    raw = b.to_bytes()
    print(
        json.dumps(
            {
                "m": int.from_bytes(raw[8:16], "little"),
                "k": int.from_bytes(raw[16:20], "little"),
                "bytes": raw.hex(),
                "contains": [b.contains(bytes.fromhex(p)) for p in c["probe"]],
            }
        )
    )
sys.stdout.flush()
