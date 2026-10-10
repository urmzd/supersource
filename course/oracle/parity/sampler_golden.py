"""Golden cases of the parity suite `sampler` (spec/sampling.md):
course/fixtures/parity/sampler.json, in the parity format
{"cases": [{"name", "input", "output"}]}.

The oracle is the L8.1 golden, course/fixtures/L8.1/sampler_golden.json
(course/oracle/L8.1/sampler_golden.py, Python stdlib only); this script only
repackages it. Input: {logits (float32 values as Python repr strings, "-inf"
for masks), params, prompt, seed, n}; output: {ids, logprobs}, the n tokens
drawn one after another from stream(seed, sample), each fed back as history.

    uv run python course/oracle/parity/sampler_golden.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "fixtures" / "L8.1" / "sampler_golden.json"
OUT = ROOT / "fixtures" / "parity" / "sampler.json"


def main() -> None:
    src = json.loads(SRC.read_text())
    cases = []
    for c in src["cases"]:
        cases.append(
            {
                "name": c["name"],
                "input": {
                    "logits": c["logits"],
                    "params": c["params"],
                    "prompt": c["prompt"],
                    "seed": c["seed"],
                    "n": len(c["ids"]),
                },
                "output": {"ids": c["ids"], "logprobs": c["logprobs"]},
            }
        )
    doc = {
        "generator": "course/oracle/parity/sampler_golden.py",
        "source": "course/fixtures/L8.1/sampler_golden.json",
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
