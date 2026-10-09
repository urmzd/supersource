"""The `tinyllm` CLI (contracts/spec/cli-roles.md).

Pass 1 verbs:

    train bigram --data <file> --out <dir> [--alpha A]
    generate --model <dir> --prompt <text> [--max-tokens N] [--greedy | --temperature T] [--seed S]
    logits --model <dir> --prompt <text> [--prefix-ids a,b,...]
    info --native

Pass 2 verbs and flags (course/milestones/MS-L0.toml, glue in cli_train.py):

    gradcheck --suite all
    train bigram --method autograd ...      train mlp ...

Entry-point territory (D16): this file is yours. It is glue over L0.0
(BigramLM, safetensors) and rt.01 (the ctypes loader); the reference is used
by course CI only. Runs as `python -m tinyllm` with python/ on the path, or
as `python python/tinyllm/__main__.py` from the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run as a script: make the package importable from python/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tinyllm.io.safetensors import load_safetensors, save_safetensors  # noqa: E402
from tinyllm.lm.bigram import BigramLM  # noqa: E402

VOCAB = 256  # the byte tokenizer (formats/tokenizer.md)
TENSOR = "bigram.weight"
CONFIG = {
    "tl_arch": "bigram",
    "tl_tokenizer": "bytes",
    "vocab_size": VOCAB,
    "tl_format": 1,
}


class UsageError(Exception):
    exit_code = 2


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


def load_model(model_dir: str) -> BigramLM:
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("tl_arch") != "bigram" or cfg.get("tl_tokenizer") != "bytes":
        raise ValueError(
            f"{d}/config.json: this CLI serves tl_arch=bigram with tl_tokenizer=bytes, got {cfg}"
        )
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    if TENSOR not in tensors:
        raise ValueError(f"{d}/model.safetensors has no {TENSOR} tensor")
    return BigramLM(weight=tensors[TENSOR])


def cmd_train(a: argparse.Namespace) -> dict:
    if a.what != "bigram" or a.method != "counts":
        from tinyllm.cli_train import cmd_train_p2

        return cmd_train_p2(a)
    if a.out is None:
        raise UsageError("train bigram needs --out <dir>")
    ids = np.frombuffer(Path(a.data).read_bytes(), dtype=np.uint8).astype(np.int64)
    if ids.size < 2:
        raise ValueError(f"{a.data}: need at least two bytes to count a bigram")
    model = BigramLM()
    model.fit_counts(ids, VOCAB, alpha=a.alpha)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        str(out / "model.safetensors"),
        {TENSOR: np.asarray(model.weight, dtype=np.float32)},
        {"format": "tinyllm"},
    )
    (out / "config.json").write_text(json.dumps(CONFIG) + "\n")
    print(f"trained bigram on {ids.size} bytes into {out}", flush=True)
    return {"out": str(out), "tokens": int(ids.size), "nll": model.nll(ids)}


def cmd_generate(a: argparse.Namespace) -> dict:
    prompt = encode(a.prompt)
    if not prompt:
        raise UsageError(
            "generate: --prompt must not be empty (the byte tokenizer has no start token)"
        )
    temperature = 0.0 if a.greedy else a.temperature
    ids = load_model(a.model).sample(prompt, a.max_tokens, temperature, a.seed)
    text = decode(ids)
    print(a.prompt + text, flush=True)
    return {"ids": ids, "text": text}


def cmd_logits(a: argparse.Namespace) -> dict:
    prefix = (
        [int(x) for x in a.prefix_ids.split(",") if x.strip()] if a.prefix_ids else []
    )
    ids = encode(a.prompt) + prefix
    if not ids:
        raise UsageError("logits: needs a non-empty --prompt or --prefix-ids")
    row = load_model(a.model).logits(ids[-1:])[0]
    return {"logits": [float(x) for x in row]}


def cmd_info(a: argparse.Namespace) -> dict:
    if not a.native:
        raise UsageError("info: Pass 1 has one form, `info --native`")
    from tinyllm.ffi import libtinyllm

    path = libtinyllm.library_path()
    lib = libtinyllm.load(path)
    return {"abi_version": int(lib.abi_version()), "lib": os.path.abspath(path)}


def cmd_gradcheck(a: argparse.Namespace) -> dict:
    from tinyllm.cli_train import cmd_gradcheck as run

    return run(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tinyllm")
    sub = ap.add_subparsers(dest="verb", required=True)
    t = sub.add_parser("train")
    t.add_argument("what")
    t.add_argument("--data", required=True)
    t.add_argument("--out")
    t.add_argument("--alpha", type=float, default=1.0)
    # Pass 2 (MS-L0): the autograd bigram, the token-stream run, the MLP.
    t.add_argument("--method", choices=["counts", "autograd"], default="counts")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lr", type=float)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--batch", type=int)
    t.add_argument("--seq-len", type=int)
    t.add_argument("--ckpt-every", type=int)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--ckpt")
    t.set_defaults(fn=cmd_train)
    gc = sub.add_parser("gradcheck")
    gc.add_argument("--suite", required=True)
    gc.set_defaults(fn=cmd_gradcheck)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=16)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)
    lg = sub.add_parser("logits")
    lg.add_argument("--model", required=True)
    lg.add_argument("--prompt", default="")
    lg.add_argument("--prefix-ids", default="")
    lg.set_defaults(fn=cmd_logits)
    i = sub.add_parser("info")
    i.add_argument("--native", action="store_true")
    i.set_defaults(fn=cmd_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)  # argparse exits 2 on a usage error
    try:
        result = a.fn(a)
    except Exception as e:  # noqa: BLE001 - every failure is reported, not traced
        # A usage error exits 2; a check that ran and failed prints its final
        # line and exits 1 (cli_train.CheckFailed); anything else exits 1.
        code = getattr(e, "exit_code", 1)
        why = str(e) if hasattr(e, "exit_code") else f"{type(e).__name__}: {e}"
        print(f"tinyllm{'' if code == 2 else ' ' + a.verb}: {why}", file=sys.stderr)
        if getattr(e, "result", None) is not None:
            print(json.dumps(e.result), flush=True)
        return code
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
