"""Maintainer generator for course/fixtures/L10.7/engine_metrics.json.

The engine's /metrics must carry exactly the instruments that
course/contracts/otel/metrics.yaml says an engine emits: the Prometheus
name, type, buckets, and label keys (attribute keys with "." replaced by
"_"), plus the allowed values and caps. This script extracts that list so
the Rust course test (L10.7) compares the exposition with the contract
without a YAML parser.

    uv run --project course/harness python course/oracle/L10.7/engine_metrics.py
Run from the repo root; it prints the MANIFEST.tsv row. Rerun whenever
metrics.yaml changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "course" / "contracts" / "otel" / "metrics.yaml"
OUT = ROOT / "course" / "fixtures" / "L10.7" / "engine_metrics.json"


def main() -> None:
    doc = yaml.safe_load(SRC.read_text())
    out = []
    for ins in doc["instruments"]:
        if "engine" not in ins.get("emitted_by", []):
            continue
        attrs = {}
        for k, spec in (ins.get("attributes") or {}).items():
            spec = spec or {}
            attrs[k.replace(".", "_")] = {
                "values": spec.get("values"),
                "capped": bool(spec.get("capped", False)),
            }
        out.append(
            {
                "name": ins["name"],
                "prometheus": ins["prometheus"],
                "type": ins["type"],
                "unit": ins.get("unit", ""),
                "buckets": ins.get("buckets"),
                "labels": attrs,
            }
        )
    body = {
        "generator": "course/oracle/L10.7/engine_metrics.py (contracts/otel/metrics.yaml, emitted_by engine)",
        "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
        "instruments": out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(body, indent=1) + "\n").encode()
    OUT.write_bytes(data)
    rel = OUT.relative_to(ROOT).as_posix()
    print(
        f"{rel}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L10.7/engine_metrics.py\tPyYAML\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
