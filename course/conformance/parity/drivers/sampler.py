"""Parity driver for `sampler` (L8.1, Python): reads one JSON case per stdin
line ({logits: [repr strings], params, prompt, seed, n}) and prints one JSON
object per line: {"ids": [...], "logprobs": [...]}, n tokens drawn one after
another with tinyllm.infer.sample.sample from request_rng(seed) =
stream(seed, sample), each appended to the history (spec/sampling.md).
"""

import json
import sys

import numpy as np

from tinyllm.infer.sample import SamplingParams, request_rng, sample

for line in sys.stdin:
    if not line.strip():
        continue
    c = json.loads(line)
    logits = np.array([float(x) for x in c["logits"]], dtype=np.float32)
    pr = c["params"]
    p = SamplingParams(
        temperature=float(pr["temperature"]),
        top_k=int(pr["top_k"]),
        top_p=float(pr["top_p"]),
        min_p=float(pr["min_p"]),
        repetition_penalty=float(pr["repetition_penalty"]),
        presence_penalty=float(pr["presence_penalty"]),
        frequency_penalty=float(pr["frequency_penalty"]),
    )
    rng = request_rng(int(c["seed"]))
    ids, lps = [], []
    for _ in range(int(c["n"])):
        tok, lp = sample(logits, p, ids, rng, prompt=c["prompt"])
        ids.append(int(tok))
        lps.append(float(lp))
    print(json.dumps({"ids": ids, "logprobs": lps}), flush=True)
