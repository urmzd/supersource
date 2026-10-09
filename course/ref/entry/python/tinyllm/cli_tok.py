"""The `tinyllm tok` verbs (Pass 3; course/milestones/MS-L1.toml fixes their
flags and final lines).

    tok train  --algo bpe --vocab N --in <text file> --out <dir> [--min-freq F] [--special S ...]
    tok encode --tokenizer <tokenizer.json> --in <texts.jsonl> [--impl python|rust]
    tok bench  --tokenizer <tokenizer.json> --in <texts.jsonl> --impl rust [--threads T] [--repeat R]

Entry-point territory (D16): this file is yours. It is glue over L1.2
(BPETokenizer: train, save, from_hf_json, encode) and L1.5 (tinyllm_rs.Bpe,
your Rust tokenizer through your PyO3 binding). Input texts are JSON lines
with a "text" key (other keys are ignored) in a .jsonl file, or one text per
line in any other file.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


class UsageError(Exception):
    exit_code = 2  # __main__ maps it to exit 2 (spec/cli-roles.md)


def read_texts(path: str) -> list[str]:
    """A .jsonl file: one JSON object with a string "text" per line. Any
    other file: one text per line."""
    if not path.endswith(".jsonl"):
        return Path(path).read_text(encoding="utf-8").splitlines()
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    raise UsageError(
                        f'{path}:{n}: want a JSON object with a string "text"'
                    )
                out.append(obj["text"])
    return out


def bytes_per_token(texts: list[str], n_tokens: int) -> float:
    total = sum(len(t.encode("utf-8")) for t in texts)
    return total / n_tokens if n_tokens else 0.0


def rust_module():
    """tinyllm_rs: importable as is, or built here from rust/crates/tl-py
    (cargo, PYO3_PYTHON = this interpreter, the macOS link arguments) into
    artifacts/pyext/, which is then put on sys.path."""
    try:
        import tinyllm_rs  # noqa: PLC0415

        return tinyllm_rs
    except ImportError:
        pass
    root = Path(__file__).resolve().parents[2]
    dest = root / "artifacts" / "pyext"
    if not (dest / "tinyllm_rs.so").is_file():
        env = dict(os.environ, PYO3_PYTHON=sys.executable)
        cmd = [
            "cargo",
            "rustc",
            "--release",
            "--quiet",
            "--manifest-path",
            str(root / "rust" / "Cargo.toml"),
            "-p",
            "tl-py",
            "--lib",
            "--crate-type",
            "cdylib",
        ]
        if platform.system() == "Darwin":
            cmd += ["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("building tinyllm_rs failed:\n" + p.stderr[-2000:])
        target = Path(env.get("CARGO_TARGET_DIR", root / "rust" / "target"))
        ext = "dylib" if platform.system() == "Darwin" else "so"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / "release" / f"libtl_py.{ext}", dest / "tinyllm_rs.so")
    sys.path.insert(0, str(dest))
    import tinyllm_rs  # noqa: PLC0415

    return tinyllm_rs


def cmd_train(a) -> dict:
    from tinyllm.tok.bpe import BPETokenizer

    if a.algo != "bpe":
        raise UsageError(
            f"tok train: --algo {a.algo} is not built yet (Pass 3 has bpe)"
        )
    if a.vocab is None or a.input is None or a.out is None:
        raise UsageError("tok train needs --vocab, --in, and --out")
    lines = Path(a.input).read_text(encoding="utf-8").splitlines()
    tok = BPETokenizer.train(
        lines, vocab_size=a.vocab, specials=a.special or (), min_freq=a.min_freq
    )
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tok.save(a.out)
    n = sum(len(tok.encode(t)) for t in lines)
    print(
        f"trained {tok.vocab_size} ids ({len(tok.merges)} merges) into {a.out}/tokenizer.json",
        flush=True,
    )
    return {
        "out": a.out,
        "algo": "bpe",
        "vocab_size": tok.vocab_size,
        "merges": len(tok.merges),
        "bytes_per_token": bytes_per_token(lines, n),
    }


def cmd_encode(a) -> dict:
    from tinyllm.tok.bpe import BPETokenizer

    texts = read_texts(a.input)
    if a.impl == "rust":
        tok = rust_module().Bpe.from_hf_json(a.tokenizer)
        per = tok.encode_batch(texts, 0)
    else:
        tok = BPETokenizer.from_hf_json(a.tokenizer)
        per = [tok.encode(t) for t in texts]
    ids = [i for row in per for i in row]
    return {
        "ids": ids,
        "texts": len(texts),
        "tokens": len(ids),
        "bytes_per_token": bytes_per_token(texts, len(ids)),
    }


def cmd_bench(a) -> dict:
    from tinyllm.tok.bpe import BPETokenizer

    if a.impl != "rust":
        raise UsageError("tok bench compares --impl rust with your Python BPE")
    texts = read_texts(a.input)
    rs = rust_module().Bpe.from_hf_json(a.tokenizer)

    def best(fn) -> float:
        times = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    # A fresh Python tokenizer per repetition: whatever it caches (pre-token
    # ids, a common speed-up) starts cold each time, as on a new corpus.
    fresh = [BPETokenizer.from_hf_json(a.tokenizer) for _ in range(a.repeat)]
    n = sum(len(rs.encode(t)) for t in texts)

    def python_run() -> None:
        py = fresh.pop()
        for t in texts:
            py.encode(t)

    t_py = best(python_run)
    t_rs = best(lambda: rs.encode_batch(texts, 1))  # one thread, one piece cache
    t_batch = best(lambda: rs.encode_batch(texts, a.threads))
    return {
        "impl": "rust",
        "texts": len(texts),
        "tokens": n,
        "threads": a.threads,
        "python_tokens_per_s": n / t_py,
        "rust_tokens_per_s": n / t_rs,
        "rust_batch_tokens_per_s": n / t_batch,
        "rust_speedup": t_py / t_rs,
        "batch_speedup": t_rs / t_batch,
    }


def add_parser(sub) -> None:
    """Called by __main__.parser(): the `tok` verb and its three actions."""
    t = sub.add_parser("tok")
    t.add_argument("action", choices=["train", "encode", "bench"])
    t.add_argument("--algo", default="bpe")
    t.add_argument("--vocab", type=int)
    t.add_argument("--in", dest="input")
    t.add_argument("--out")
    t.add_argument("--min-freq", type=int, default=2)
    t.add_argument("--special", action="append")
    t.add_argument("--tokenizer")
    t.add_argument("--impl", choices=["python", "rust"], default="python")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--repeat", type=int, default=3)
    t.set_defaults(fn=run)


def run(a) -> dict:
    if a.action == "train":
        return cmd_train(a)
    if a.tokenizer is None or a.input is None:
        raise UsageError(f"tok {a.action} needs --tokenizer and --in")
    return cmd_encode(a) if a.action == "encode" else cmd_bench(a)
