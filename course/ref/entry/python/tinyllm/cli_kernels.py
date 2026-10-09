"""The Pass 6 kernel verbs (course/milestones/MS-L9.toml fixes their flags and
final lines).

    generate --model <llama dir> (--prompt TEXT | --prompt-file FILE) --backend {numpy,c} [--check]
             [--max-tokens N] [--greedy | --temperature T] [--seed S] [--out FILE]
    logits   --model <llama dir> --prompt TEXT [--prefix-ids a,b,...] --backend {numpy,c} [--check]
    bench decode --model <llama dir> --backend numpy,c [--tokens N] [--prompt TEXT]

`intercept(argv)` claims these forms only when `--backend` is on the command
line, and returns None for everything else, so the Pass 5 verbs (cli_modern)
and the other `bench` forms keep their meaning. `--backend numpy` is L7.9's
forward; `--backend c` is L9.7's CBackend over libtinyllm ($TINYLLM_LIB, else
c/build/libtinyllm.*). `--check` runs L9.7's load-time op check first: its
report goes to stderr and its ratios into the final line ("check"), and a
failed op exits 1 before any token.

Entry-point territory (D16): this file is yours. It is glue over L9.7
(CBackend), L7.9 and cli_modern (loading, the tokenizer, the numpy decode
loop), and M06.3 (PCG32 for sampling).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tinyllm.cli_modern import UsageError, _Parser, is_llama_dir, load


def _backend(m, a):
    from tinyllm.backend.c import CBackend, OpCheckError

    try:
        be = CBackend(m, check=bool(getattr(a, "check", False)))
    except OpCheckError as e:
        for r in e.report:
            print(
                f"  {r.op:<10} ratio {r.ratio:.3g}  {'ok' if r.ok else 'FAIL'}  {r.detail}",
                file=sys.stderr,
            )
        e.result = {"check": {r.op: r.ratio for r in e.report}, "ok": False}  # type: ignore[attr-defined]
        raise
    if be.report is not None:
        for r in be.report:
            print(f"  {r.op:<10} ratio {r.ratio:.3g}  ok  {r.detail}", file=sys.stderr)
    return be


def _next_c(be, cache, ids: list[int]) -> np.ndarray:
    return np.asarray(be.forward(ids, cache=cache)[-1], dtype=np.float64)


def generate_c(
    be, prompt: list[int], n: int, temperature: float, seed: int
) -> list[int]:
    from tinyllm.backend.c import argmax
    from tinyllm.num.rng import PCG32

    if not prompt:
        raise UsageError("generate: the prompt has no tokens")
    cache = be.new_cache()
    rng = PCG32(seed).substream("sample")
    row = _next_c(be, cache, prompt)
    out: list[int] = []
    for _ in range(n):
        if temperature == 0:
            nxt = argmax(row.astype(np.float32))  # ties to the lowest id, in C
        else:
            z = row / temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            nxt = min(
                int(np.searchsorted(np.cumsum(p), rng.uniform(), side="right")),
                len(p) - 1,
            )
        out.append(nxt)
        if len(out) < n:
            row = _next_c(be, cache, [nxt])
    return out


def cmd_generate(a) -> dict:
    from tinyllm.cli_modern import generate_ids

    if bool(a.prompt) == bool(a.prompt_file):
        raise UsageError("generate needs exactly one of --prompt and --prompt-file")
    m, text = load(a.model)
    t = 0.0 if a.greedy else a.temperature
    prompts = (
        [a.prompt]
        if a.prompt
        else [x for x in Path(a.prompt_file).read_text().splitlines() if x]
    )
    result: dict = {}
    if a.backend == "c":
        be = _backend(m, a)
        per = [generate_c(be, text.encode(p), a.max_tokens, t, a.seed) for p in prompts]
        if be.report is not None:
            result["check"] = {r.op: r.ratio for r in be.report}
            result["check_max_ratio"] = max(r.ratio for r in be.report)
    else:
        per = [
            generate_ids(m, text.encode(p), a.max_tokens, t, a.seed) for p in prompts
        ]
    ids = [i for row in per for i in row]
    result = {
        "ids": ids,
        "text": "\n".join(text.decode(r) for r in per),
        "backend": a.backend,
        **result,
    }
    if a.prompt_file:
        result["per_prompt"] = per
    if a.out:
        Path(a.out).write_text(json.dumps(result) + "\n")
    return result


def cmd_logits(a) -> dict:
    m, text = load(a.model)
    prefix = (
        [int(x) for x in a.prefix_ids.split(",") if x.strip()] if a.prefix_ids else []
    )
    ids = text.encode(a.prompt) + prefix
    if not ids:
        raise UsageError("logits: the prompt has no tokens")
    if a.backend == "c":
        row = _backend(m, a).forward(ids)[-1]
    else:
        row = m(np.array([ids])).data[0, -1]
    return {"logits": [float(x) for x in np.asarray(row, dtype=np.float64)]}


def cmd_bench(a) -> dict:
    """Decode `tokens` greedy steps after the prompt with each backend, best
    of three timed runs each (after one warm-up of 4 steps), and report
    tokens per second and speedup_c = c / numpy."""
    from tinyllm.cli_modern import generate_ids

    if a.what != "decode":
        raise UsageError("bench --backend: the one form is `bench decode`")
    backends = [b for b in a.backend.split(",") if b]
    if not backends or any(b not in ("numpy", "c") for b in backends):
        raise UsageError("--backend takes numpy, c, or numpy,c")
    m, text = load(a.model)
    prompt = text.encode(a.prompt)
    be = _backend(m, a) if "c" in backends else None
    runs = {
        "numpy": lambda n: generate_ids(m, prompt, n, 0.0, 0),
        "c": lambda n: generate_c(be, prompt, n, 0.0, 0),
    }
    out: dict = {"tokens": a.tokens}
    for b in backends:
        runs[b](4)
        best = min(_timed(runs[b], a.tokens) for _ in range(3))
        out[f"tok_s_{b}"] = round(a.tokens / best, 3)
    if len(backends) == 2:
        out["speedup_c"] = round(out["tok_s_c"] / out["tok_s_numpy"], 3)
    return out


def _timed(fn, n: int) -> float:
    t0 = time.perf_counter()
    fn(n)
    return time.perf_counter() - t0


def parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="tinyllm")
    sub = ap.add_subparsers(dest="verb", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", default="")
    g.add_argument("--prompt-file")
    g.add_argument("--max-tokens", type=int, default=16)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out")
    g.add_argument("--backend", choices=["numpy", "c"], required=True)
    g.add_argument("--check", action="store_true")
    g.set_defaults(fn=cmd_generate)
    lg = sub.add_parser("logits")
    lg.add_argument("--model", required=True)
    lg.add_argument("--prompt", default="")
    lg.add_argument("--prefix-ids", default="")
    lg.add_argument("--backend", choices=["numpy", "c"], required=True)
    lg.add_argument("--check", action="store_true")
    lg.set_defaults(fn=cmd_logits)
    b = sub.add_parser("bench")
    b.add_argument("what")
    b.add_argument("--model", required=True)
    b.add_argument("--backend", required=True)
    b.add_argument("--tokens", type=int, default=128)
    b.add_argument("--prompt", default="Once upon a time")
    b.add_argument("--check", action="store_true")
    b.set_defaults(fn=cmd_bench)
    return ap


def intercept(argv: list[str]) -> int | None:
    if not argv or argv[0] not in ("generate", "logits", "bench"):
        return None
    if not any(x == "--backend" or x.startswith("--backend=") for x in argv):
        return None
    try:
        a = parser().parse_args(argv)
        if not is_llama_dir(a.model):
            raise UsageError(
                f"--backend runs a Llama-family model directory; {a.model} is not one"
            )
        result = a.fn(a)
    except Exception as e:  # noqa: BLE001 - every failure is reported, not traced
        code = getattr(e, "exit_code", 1)
        why = str(e) if hasattr(e, "exit_code") else f"{type(e).__name__}: {e}"
        print(f"tinyllm{'' if code == 2 else ' ' + argv[0]}: {why}", file=sys.stderr)
        if getattr(e, "result", None) is not None:
            print(json.dumps(e.result), flush=True)
        return code
    print(json.dumps(result), flush=True)
    return 0
