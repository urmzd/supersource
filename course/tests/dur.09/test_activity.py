"""Course tests for dur.09: the Python activity helper (python/tinyllm/io/activity.py).

Rung R0 for these course tests. Each test names why it exists (WHY), what
kind of check it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/dur.09), and the chapter section it comes from.

The helper keeps the Python half of contracts/spec/subprocess-activity.md:
progress events as flushed single lines, ckpt events only for complete
checkpoints, outputs published by rename with DONE.json last, SIGTERM as a
flag the loop honors, one attempt at a time per work directory, and the
exit-code table. Tests that need a real signal or a second process start a
child Python with the same import path (subprocess, own process group,
bounded by a timeout).

The chapter's worked example (section 3), clock fixed at 1760000000.0,
TL_ARTIFACTS=<tmp>/art, key train-1/3, attempt 2 resuming from step 500:

    step  {"ts":1760000000.0,"kind":"step","step":600,"loss":2.25,"lr":0.0009,"tokens":9830400}
    ckpt  {"ts":1760000000.0,"kind":"ckpt","step":1000,"ckpt":"runs/train-1/ckpt/step-001000"}
    done  {"ts":1760000000.0,"kind":"done","outputs":["runs/train-1/ckpt/step-001000"]}
    DONE.json  {"outputs":["runs/train-1/ckpt/step-001000"]}
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from tinyllm.io.activity import (
    EXIT_CANCELED,
    EXIT_DATAERR,
    EXIT_OK,
    EXIT_TEMPFAIL,
    Activity,
    Cancelled,
    RetryableError,
    SpecError,
    run,
)

TS = 1760000000.0
PATIENCE = 20.0


def clock() -> float:
    return TS


def make_ckpt(art: Path, rel: str, complete: bool = True) -> Path:
    """A step directory as formats/checkpoint.md lays it out: the required
    files and a MANIFEST.json with each one's size and sha256."""
    d = art / rel
    d.mkdir(parents=True)
    files = []
    for name in ("model.safetensors", "optimizer.safetensors", "trainer_state.json"):
        data = name.encode() * 3
        (d / name).write_bytes(data)
        files.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if complete:
        man = {"format": 1, "kind": "checkpoint", "files": files}
        (d / "MANIFEST.json").write_text(json.dumps(man))
    return d


def setup(tmp_path: Path, key: str = "train-1/3", attempt: int = 1, extra: list[str] | None = None):
    art = tmp_path / "art"
    work = art / "activities" / key
    work.mkdir(parents=True)
    (work / "spec.json").write_text(json.dumps({"steps": 1000}))
    argv = ["train", "--spec", str(work / "spec.json"), "--progress", str(work / "progress.jsonl"), *(extra or [])]
    env = {"TL_ARTIFACTS": str(art), "TL_IDEMPOTENCY_KEY": key, "TL_ATTEMPT": str(attempt)}
    return art, work, argv, env


def lines(work: Path) -> list[dict]:
    p = work / "progress.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return env


def spawn(code: str, env: dict, args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        env={**child_env(), **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def wait_for(cond, what: str, timeout: float = PATIENCE) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def finish(p: subprocess.Popen, timeout: float = PATIENCE) -> tuple[int, str]:
    try:
        _, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        _, err = p.communicate()
        raise AssertionError(f"child did not exit within {timeout} s: {err[-2000:]}") from None
    return p.returncode, err


def test_hand_example_progress_lines(tmp_path):
    # WHY: the chapter's worked example (section 3): attempt 2 of train-1/3
    #      resumes from step 500, reports a step, a complete checkpoint, and
    #      its outputs. The progress file and DONE.json must be exactly these
    #      bytes, because the Go runner and the schema read them, not Python.
    # KIND: unit
    # CATCHES: s18, s21
    # CHAPTER: dur.09 section 3, worked example
    art, work, argv, env = setup(tmp_path, attempt=2, extra=["--resume", "runs/train-1/ckpt/step-000500"])
    make_ckpt(art, "runs/train-1/ckpt/step-001000")
    act = Activity(argv, env, clock=clock)
    assert (act.key, act.attempt, act.resume) == ("train-1/3", 2, "runs/train-1/ckpt/step-000500")
    act.step(600, 2.25, 0.0009, 9830400)
    assert act.checkpoint(1000, str(art / "runs/train-1/ckpt/step-001000")) == "runs/train-1/ckpt/step-001000"
    result = act.done(["runs/train-1/ckpt/step-001000"])
    act.close()
    assert (work / "progress.jsonl").read_text() == (
        '{"ts":1760000000.0,"kind":"step","step":600,"loss":2.25,"lr":0.0009,"tokens":9830400}\n'
        '{"ts":1760000000.0,"kind":"ckpt","step":1000,"ckpt":"runs/train-1/ckpt/step-001000"}\n'
        '{"ts":1760000000.0,"kind":"done","outputs":["runs/train-1/ckpt/step-001000"]}\n'
    )
    assert (work / "DONE.json").read_bytes() == b'{"outputs":["runs/train-1/ckpt/step-001000"]}' == result


def test_flags_and_environment(tmp_path):
    # WHY: the runner passes --spec, --progress, and --resume (as two words;
    #      `--x=v` is accepted too) and the TL_* variables; everything else in
    #      argv belongs to the entry. Without --spec or --progress the run
    #      cannot keep the contract at all: SpecError, which is exit 65.
    # KIND: unit
    # CHAPTER: dur.09 section 4, The interface
    art, work, argv, env = setup(tmp_path)
    act = Activity(["train", "--spec=" + str(work / "spec.json"), "--progress", str(work / "progress.jsonl"), "--lr", "3"], env)
    assert act.resume is None and act.attempt == 1 and act.key == "train-1/3"
    assert act.work_dir == str(work) and act.artifacts == str(art)
    assert act.load_spec() == {"steps": 1000}
    defaults = Activity(argv, {})
    assert (defaults.attempt, defaults.key, defaults.artifacts) == (1, "", "/artifacts")
    for bad in (["train", "--progress", "p"], ["train", "--spec", "s"], ["train", "--spec"]):
        with pytest.raises(SpecError):
            Activity(bad, env)


def test_each_event_is_one_flushed_line(tmp_path):
    # WHY: the Go runner tails progress.jsonl while the child runs. An event
    #      that sits in Python's write buffer is invisible until exit, so a
    #      worker killed mid-run never heartbeats the checkpoint it reported.
    #      Every event must be on disk as one complete line when emit returns.
    # KIND: unit
    # CATCHES: s17
    # CHAPTER: dur.09 section 2.3
    art, work, argv, env = setup(tmp_path)
    act = Activity(argv, env, clock=clock)
    for i in range(3):
        act.step(i, 1.0, 0.1, 64)
        raw = (work / "progress.jsonl").read_text()
        assert raw.endswith("\n") and raw.count("\n") == i + 1, f"after event {i}: {raw!r}"
    act.metric("val_loss", 2.40)
    assert lines(work)[-1] == {"ts": TS, "kind": "metric", "name": "val_loss", "value": 2.40}
    act.close()


def test_rejects_bad_events(tmp_path):
    # WHY: the progress schema has four kinds, and JSON has no NaN: a loss
    #      that went NaN must not write a line the runner (and every JSON
    #      parser) rejects.
    # KIND: boundary
    # CATCHES: s19
    # CHAPTER: dur.09 section 5, Pitfalls
    art, work, argv, env = setup(tmp_path)
    act = Activity(argv, env, clock=clock)
    with pytest.raises(ValueError):
        act.emit("progress", step=1)
    for v in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            act.step(1, v, 0.1, 1)
    assert lines(work) == []
    act.close()


def test_checkpoint_only_when_complete(tmp_path):
    # WHY: a ckpt event becomes the heartbeat detail and the next attempt's
    #      --resume. Emitting it for a directory whose MANIFEST.json is not
    #      written yet (a crash mid-save) hands the next attempt a checkpoint
    #      it cannot load. The event is for complete step directories only,
    #      and its path is relative to TL_ARTIFACTS.
    # KIND: unit
    # CATCHES: s20, s21
    # CHAPTER: dur.09 section 2.3
    art, work, argv, env = setup(tmp_path)
    act = Activity(argv, env, clock=clock)
    make_ckpt(art, "runs/r/ckpt/step-000100", complete=False)
    with pytest.raises(ValueError):
        act.checkpoint(100, "runs/r/ckpt/step-000100")
    assert lines(work) == [], "no event for an incomplete checkpoint"
    make_ckpt(art, "runs/r/ckpt/step-000200")
    assert act.checkpoint(200, "runs/r/ckpt/step-000200") == "runs/r/ckpt/step-000200"
    assert act.checkpoint(200, str(art / "runs/r/ckpt/step-000200")) == "runs/r/ckpt/step-000200"
    for outside in ("/etc", str(tmp_path / "elsewhere"), "../x"):
        with pytest.raises(ValueError):
            act.checkpoint(1, outside)
    assert [e["ckpt"] for e in lines(work)] == ["runs/r/ckpt/step-000200"] * 2
    act.close()


def test_publish_never_exposes_a_partial_file(tmp_path, monkeypatch):
    # WHY: a kill between "write the bytes" and "done" must leave either no
    #      output or the whole output under its final name. Publishing is
    #      write name.tmp, fsync, rename; when the rename never happens the
    #      final name must not exist at all.
    # KIND: fault
    # CATCHES: s22
    # CHAPTER: dur.09 section 2.2
    art, work, argv, env = setup(tmp_path)
    act = Activity(argv, env, clock=clock)
    p = act.publish("shard.bin", b"x" * 100)
    assert Path(p).read_bytes() == b"x" * 100 and not Path(p + ".tmp").exists()

    def crash(src, dst):
        raise OSError("killed before the rename")

    monkeypatch.setattr(os, "replace", crash)
    monkeypatch.setattr(os, "rename", crash)
    with pytest.raises(OSError):
        act.publish("DONE.json", b'{"outputs":[]}')
    assert not (work / "DONE.json").exists(), "DONE.json must appear only by rename"
    act.close()


def test_outputs_stay_in_the_work_dir(tmp_path):
    # WHY: output names come from specs and code; a name that climbs out of
    #      the work dir would let one activity overwrite another's files.
    # KIND: boundary
    # CATCHES: s23
    # CHAPTER: dur.09 section 2.2
    art, work, argv, env = setup(tmp_path)
    act = Activity(argv, env)
    assert act.output("a/b.txt") == str(work / "a" / "b.txt")
    for bad in ("../x", "a/../../x", "/etc/passwd", "", "."):
        with pytest.raises(ValueError):
            act.output(bad)


def test_run_exit_codes(tmp_path):
    # WHY: the exit code is the verdict the Go runner maps (contract table).
    #      A wrong input must exit 65 (never retried), a transient failure 75,
    #      a cancellation 130, a crash nonzero and retryable; success writes
    #      DONE.json and exits 0.
    # KIND: unit
    # CATCHES: s24
    # CHAPTER: dur.09 section 2.4
    def raiser(exc):
        def main(act):
            raise exc

        return main

    cases = [
        (raiser(SpecError("bad spec")), EXIT_DATAERR),
        (raiser(RetryableError("flaky")), EXIT_TEMPFAIL),
        (raiser(Cancelled("term")), EXIT_CANCELED),
        (raiser(RuntimeError("bug")), 1),
        (lambda act: ["out/a"], EXIT_OK),
    ]
    for i, (main, code) in enumerate(cases):
        art, work, argv, env = setup(tmp_path / str(i))
        assert run(main, argv, env) == code, f"case {i}"
        assert (work / "DONE.json").exists() == (code == EXIT_OK), f"case {i}: DONE.json only on success"
    assert json.loads((work / "DONE.json").read_text()) == {"outputs": ["out/a"]}
    assert run(lambda act: [], ["train"], {}) == EXIT_DATAERR, "no --spec: exit 65"


def test_spec_validation_is_exit_65(tmp_path):
    # WHY: "a spec that fails validation exits 65 before writing anything":
    #      retrying a spec that will never validate burns every attempt and
    #      lands in the dead-letter queue instead of failing fast.
    # KIND: unit
    # CATCHES: s25
    # CHAPTER: dur.09 section 2.4
    art, work, argv, env = setup(tmp_path)
    act = Activity(argv, env)
    with pytest.raises(SpecError, match="steps must be positive"):
        act.load_spec(lambda s: ["steps must be positive"] if s["steps"] > 999 else [])
    (work / "spec.json").write_text("[1, 2]")
    with pytest.raises(SpecError):
        act.load_spec()
    (work / "spec.json").write_text("{not json")
    with pytest.raises(SpecError):
        act.load_spec()
    assert run(lambda a: a.load_spec(lambda s: ["no"]), argv, env) == EXIT_DATAERR
    assert not (work / "DONE.json").exists()


def test_rerun_after_done_is_a_noop(tmp_path):
    # WHY: a duplicate delivery (the worker died after the child succeeded
    #      but before CompleteActivityTask) must not redo the work: the second
    #      run finds DONE.json, re-emits its done event, and exits 0 without
    #      calling main.
    # KIND: regression
    # CATCHES: s26
    # CHAPTER: dur.09 section 2.2
    art, work, argv, env = setup(tmp_path)
    calls = []

    def main(act):
        calls.append(act.attempt)
        return ["out/model"]

    assert run(main, argv, env) == EXIT_OK
    first = (work / "DONE.json").read_bytes()
    env["TL_ATTEMPT"] = "2"
    assert run(main, argv, env) == EXIT_OK
    assert calls == [1], "main ran again after DONE.json existed"
    assert (work / "DONE.json").read_bytes() == first
    dones = [e for e in lines(work) if e["kind"] == "done"]
    assert [d["outputs"] for d in dones] == [["out/model"], ["out/model"]]


LOOP = """
import os, sys, time, json, hashlib
from tinyllm.io.activity import run

def make_ckpt(art, rel):
    d = os.path.join(art, rel); os.makedirs(d)
    files = []
    for n in ("model.safetensors", "optimizer.safetensors", "trainer_state.json"):
        b = n.encode()
        open(os.path.join(d, n), "wb").write(b)
        files.append({"name": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    json.dump({"format": 1, "kind": "checkpoint", "files": files}, open(os.path.join(d, "MANIFEST.json"), "w"))
    return d

def main(act):
    step = 0
    while True:
        step += 1
        act.step(step, 1.0, 0.1, 64)
        if act.cancelled:
            act.checkpoint(step, make_ckpt(act.artifacts, "runs/t/ckpt/step-%06d" % step))
        act.check_cancel()
        time.sleep(0.02)

sys.exit(run(main, sys.argv[1:]))
"""


def test_sigterm_checkpoints_and_exits_130(tmp_path):
    # WHY: cancellation and worker drains arrive as SIGTERM. The handler only
    #      sets a flag; the training loop notices it at its next step,
    #      writes a checkpoint (so the work survives), emits its ckpt event,
    #      and exits 130 well inside the runner's 30 s grace. A handler that
    #      exits at once loses the work since the last checkpoint.
    # KIND: fault
    # CATCHES: s27
    # CHAPTER: dur.09 section 2.5
    art, work, argv, env = setup(tmp_path)
    p = spawn(LOOP, env, argv[1:])
    wait_for(lambda: len(lines(work)) >= 3, "the loop's first steps")
    os.killpg(p.pid, signal.SIGTERM)
    t0 = time.monotonic()
    code, err = finish(p)
    assert code == EXIT_CANCELED, f"exit {code}, stderr: {err[-1500:]}"
    assert time.monotonic() - t0 < 5.0
    ev = lines(work)
    assert ev[-1]["kind"] == "ckpt", f"the last event must be the checkpoint written on SIGTERM: {ev[-3:]}"
    assert (art / ev[-1]["ckpt"] / "MANIFEST.json").exists()
    assert not (work / "DONE.json").exists()


HOLD = """
import sys, time
from tinyllm.io.activity import run

def main(act):
    open(act.output("started"), "w").close()
    while not act.cancelled:
        time.sleep(0.01)
    act.check_cancel()

sys.exit(run(main, sys.argv[1:]))
"""


def test_second_attempt_is_locked_out(tmp_path):
    # WHY: a worker SIGKILLed mid-activity can leave its child running while
    #      the server redelivers the task to another worker, whose child would
    #      write into the same work dir. While one attempt holds the work-dir
    #      lock, another must exit 75 (retry later) without running main.
    # KIND: fault
    # CATCHES: s28
    # CHAPTER: dur.09 section 5, Pitfalls
    art, work, argv, env = setup(tmp_path)
    p = spawn(HOLD, env, argv[1:])
    try:
        wait_for(lambda: (work / "started").exists(), "the first attempt to start")
        ran = []
        env2 = dict(env, TL_ATTEMPT="2")
        assert run(lambda act: ran.append(1) or [], argv, env2) == EXIT_TEMPFAIL
        assert ran == [], "main ran while another attempt held the lock"
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        finish(p)
    assert run(lambda act: ["ok"], argv, dict(env, TL_ATTEMPT="3")) == EXIT_OK, "the lock is released at exit"


ORPHAN = """
import os, subprocess, sys, time
# The "worker": start the activity child, wait until it runs, report its
# pid, and die without stopping it (as a SIGKILLed worker would).
started = sys.argv[1]
child = subprocess.Popen([sys.executable, "-c", sys.argv[2]] + sys.argv[3:])
deadline = time.monotonic() + 20
while not os.path.exists(started) and time.monotonic() < deadline:
    time.sleep(0.01)
print(child.pid, flush=True)
os._exit(0)
"""

ORPHAN_CHILD = """
import os, sys, time
from tinyllm.io.activity import run

def main(act):
    open(act.output("started"), "w").close()
    deadline = time.monotonic() + 5
    while not act.cancelled and time.monotonic() < deadline:
        time.sleep(0.01)
    open(act.output("noticed"), "w").write(str(act.cancelled))
    act.check_cancel()

sys.exit(run(main, sys.argv[1:]))
"""


def test_orphaned_child_stops(tmp_path):
    # WHY: when the worker dies, its child is reparented. Nobody will read
    #      its result, and the redelivered attempt is about to start; the
    #      orphan must notice (its parent pid changed) and stop as if
    #      cancelled instead of training on for hours.
    # KIND: fault
    # CATCHES: s29
    # CHAPTER: dur.09 section 5, Pitfalls
    art, work, argv, env = setup(tmp_path)
    p = spawn(ORPHAN, env, [str(work / "started"), textwrap.dedent(ORPHAN_CHILD), *argv[1:]])
    out, _ = p.communicate(timeout=PATIENCE)
    pid = int(out.split()[0])
    try:
        wait_for(lambda: (work / "noticed").exists(), "the orphan to notice its parent died")
        assert (work / "noticed").read_text() == "True"
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
