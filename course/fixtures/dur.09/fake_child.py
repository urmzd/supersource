"""A scripted fake Python activity for the Go runner's conformance tests (dur.09).

contracts/spec/subprocess-activity.md is the interface between the Go
worker and a Python entry. This script plays the Python side exactly as the
test tells it to, so the Go tests (course/tests/go/dur_09) can check the
runner on its own: progress tailing, heartbeat details, --resume, the
environment, exit-code mapping, SIGTERM and SIGKILL. Standard library only;
it never imports the learner's tinyllm.

Usage (what the runner execs):

    python3 fake_child.py <verb> --spec <dir>/spec.json --progress <dir>/progress.jsonl [--resume <ckpt>]

The spec is the script:

    {"on_term": "exit130" | "ignore" | "ckpt_then_130",   # what SIGTERM does (default exit130)
     "attempts": {"1": [action, ...], "2": [...], "*": [...]}}   # by TL_ATTEMPT, "*" otherwise

Actions, run in order:

    {"emit": {...}}             one progress line {"ts": ..., **fields}, flushed
    {"partial": "text"}         write text with no newline, flushed (a line being written)
    {"finish_line": true}       write "\n" (ends a partial line)
    {"stderr": "text"}          write text to stderr
    {"stdout": "text"}          write text to stdout
    {"sleep": seconds}
    {"wait_term": seconds}      wait up to `seconds` for SIGTERM, then go on
    {"record": "name"}          write {"argv", "env", "cwd", "pid", "pgid"} to <work dir>/<name>.json
    {"done": ["out/a", ...]}    publish DONE.json ({"outputs": [...]}) by rename, emit "done"
    {"big_done": n}             publish a DONE.json of n bytes
    {"spawn_sleeper": "name"}   start a grandchild in the same process group; write its pid to <name>.pid
    {"exit": code}              exit with code
    {"kill_self": signo}        die by a signal
The script falls off its end with exit 0.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time


def flag(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


def main() -> int:
    argv = sys.argv[1:]
    spec_path, progress_path = flag(argv, "--spec"), flag(argv, "--progress")
    if not spec_path or not progress_path:
        print("fake_child: --spec and --progress are required", file=sys.stderr)
        return 65
    with open(spec_path) as f:
        spec = json.load(f)
    work = os.path.dirname(os.path.abspath(progress_path))
    attempt = os.environ.get("TL_ATTEMPT", "1")
    actions = spec.get("attempts", {}).get(attempt) or spec.get("attempts", {}).get("*") or []
    on_term = spec.get("on_term", "exit130")
    termed = {"flag": False}
    prog = open(progress_path, "a")

    def emit(fields: dict) -> None:
        prog.write(json.dumps({"ts": time.time(), **fields}) + "\n")
        prog.flush()

    def handler(signum, frame):  # noqa: ARG001
        termed["flag"] = True
        if on_term == "ignore":
            return
        if on_term == "ckpt_then_130":
            emit({"kind": "ckpt", "step": 999, "ckpt": "runs/fake/ckpt/on-term"})
        os._exit(130)

    signal.signal(signal.SIGTERM, handler)

    def publish(name: str, data: bytes) -> None:
        tmp = os.path.join(work, name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, os.path.join(work, name))

    for a in actions:
        if "emit" in a:
            emit(a["emit"])
        elif "partial" in a:
            prog.write(a["partial"])
            prog.flush()
        elif "finish_line" in a:
            prog.write("\n")
            prog.flush()
        elif "stderr" in a:
            sys.stderr.write(a["stderr"])
            sys.stderr.flush()
        elif "stdout" in a:
            sys.stdout.write(a["stdout"])
            sys.stdout.flush()
        elif "sleep" in a:
            time.sleep(a["sleep"])
        elif "wait_term" in a:
            deadline = time.monotonic() + a["wait_term"]
            while not termed["flag"] and time.monotonic() < deadline:
                time.sleep(0.01)
        elif "record" in a:
            rec = {"argv": sys.argv, "env": dict(os.environ), "cwd": os.getcwd(),
                   "pid": os.getpid(), "pgid": os.getpgid(0)}
            publish(a["record"] + ".json", json.dumps(rec).encode())
        elif "done" in a:
            publish("DONE.json", json.dumps({"outputs": a["done"]}).encode())
            emit({"kind": "done", "outputs": a["done"]})
        elif "big_done" in a:
            publish("DONE.json", b"{" + b" " * (a["big_done"] - 2) + b"}")
        elif "spawn_sleeper" in a:
            p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            publish(a["spawn_sleeper"] + ".pid", str(p.pid).encode())
        elif "exit" in a:
            prog.close()
            return int(a["exit"])
        elif "kill_self" in a:
            prog.flush()
            os.kill(os.getpid(), int(a["kill_self"]))
            time.sleep(5)
    prog.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
