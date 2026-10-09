"""The verdict ledger: <learner>/.ss/verdicts.jsonl (DESIGN 5.4).

One JSON object per line. Check verdicts carry `result`; events (`start`,
`spoiled`) carry `event`. Readers take the last matching line.

  {"id":"L8.3","kind":"build","tree":"sha256:..","sources":{"L8.2":"learner"},
   "result":"pass","assisted":false,"mutation":null,"ss":"a5a70b1","ts":"..."}
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import ctx
from .registry import Registry


def path(learner: Path) -> Path:
    return learner / ".ss" / "verdicts.jsonl"


def entries(learner: Path) -> list[dict]:
    p = path(learner)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append(learner: Path, entry: dict) -> dict:
    entry = dict(entry)
    entry.setdefault("ss", ctx.harness_sha())
    entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    p = path(learner)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry, sort_keys=False) + "\n")
    return entry


def event(learner: Path, mid: str, name: str, **extra) -> None:
    append(learner, {"id": mid, "event": name, **extra})


def has_event(learner: Path, mid: str, name: str) -> bool:
    return any(e.get("id") == mid and e.get("event") == name for e in entries(learner))


def latest(learner: Path, mid: str, full_only: bool = False) -> dict | None:
    """The last verdict for an id. `full_only` skips `--smoke` milestone runs,
    which never stand in for the full milestone."""
    for e in reversed(entries(learner)):
        if (
            e.get("id") == mid
            and "result" in e
            and not (full_only and e.get("mode") == "smoke")
        ):
            return e
    return None


def tree_hash(learner: Path, reg: Registry, mid: str) -> str:
    """Hash of the module's owned units plus its graded learner tests. A unit
    a later module took over (and the learner started) no longer counts: from
    then on edits to it make only the later module stale (DESIGN 5.2 point 4)."""
    m = reg.get(mid)
    gone = {
        u
        for u, later in reg.superseded_by(mid).items()
        if has_event(learner, later, "start")
    }
    h = hashlib.sha256()
    for u in sorted(m.owned):
        if u in gone:
            continue
        p = learner / u
        h.update(
            u.encode()
            + b"\0"
            + (p.read_bytes() if p.is_file() else b"<missing>")
            + b"\0"
        )
    lt = (m.learner_tests or {}).get("path")
    if lt:
        base = learner / lt
        files = (
            sorted(x for x in base.rglob("*") if x.is_file())
            if base.is_dir()
            else ([base] if base.is_file() else [])
        )
        for p in files:
            if "__pycache__" in p.parts:
                continue
            h.update(
                p.relative_to(learner).as_posix().encode()
                + b"\0"
                + p.read_bytes()
                + b"\0"
            )
    return "sha256:" + h.hexdigest()


def fresh_pass(learner: Path, reg: Registry, mid: str) -> dict | None:
    v = latest(learner, mid)
    if (
        v
        and v.get("result") == "pass"
        and v.get("tree") == tree_hash(learner, reg, mid)
    ):
        return v
    return None
