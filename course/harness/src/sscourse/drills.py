"""Drill specs, injectors, and the undo journal (DESIGN 5.10).

    course/drills/<name>/drill.toml
      id, title, requires, seeded, time_limit_min, slo_profile, symptom
      [[inject]]  kind, target, at ("start" | "+30s" | "loadgen+30s"), plus per-kind keys
      [loadgen]   argv  (for at = "loadgen+Ns"; default --rate 20 --target {deploy.gateway_url})
      [detect]    alert, within_s
      [[resolve.check]]  promql + hold_s, rollout (kubectl rollout status), or suite
      [[doc]]            path + sections (+ label): a document graded like the postmortem
      [evidence.trace]   a trace matcher table (service, services, spans)
      [postmortem] path (with {date}), sections

Every injection appends one line to .ss/drills/<run>/journal.jsonl with the
kubectl argv that undoes it; `ss drill reset` replays them in reverse.
"""

from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, placeholders
from .kube import Kube

BUILT = ("pod-kill", "deploy-patch", "scale-zero", "resource-limit", "config-drift")
LATER = {
    "netem": "B13 (ops.12)",
    "wal-quota": "B13 (ops.11)",
    "clock-skew": "B10 (dur.07)",
    "activity-poison": "B10 (ops.03)",
    "tenant-flood": "B13 (ops.09)",
    "git-branch": "B13 (ops.06 to ops.08)",
    "contract-bump": "B13 (ops.04, ops.05)",
}
AT = re.compile(r"^(start|(loadgen)?\+(\d+(?:\.\d+)?)s)$")


@dataclass
class Drill:
    name: str
    path: Path
    raw: dict

    @property
    def id(self) -> str:
        return str(self.raw.get("id", self.name))

    @property
    def title(self) -> str:
        return str(self.raw.get("title", ""))

    @property
    def injects(self) -> list[dict]:
        return list(self.raw.get("inject", []))

    @property
    def requires(self) -> list[str]:
        return list(self.raw.get("requires", []))


def all_drills(course: Path) -> list[Drill]:
    d = course / "drills"
    out = []
    for p in sorted(d.glob("*/drill.toml")) if d.is_dir() else []:
        out.append(load_file(p))
    return out


def load_file(p: Path) -> Drill:
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{p}: {e}") from None
    d = Drill(p.parent.name, p, raw)
    for i, inj in enumerate(d.injects):
        k = inj.get("kind")
        if k not in BUILT and k not in LATER:
            raise HarnessError(f"{p}: inject {i + 1}: unknown kind {k!r}")
        if not AT.match(str(inj.get("at", "start"))):
            raise HarnessError(
                f"{p}: inject {i + 1}: at {inj.get('at')!r} is not start, +Ns, or loadgen+Ns"
            )
    return d


def find(course: Path, name: str) -> Drill:
    for d in all_drills(course):
        if name in (d.name, d.id):
            return d
    raise HarnessError(f"no drill {name!r} in {course / 'drills'} (ss drill list)")


# ---------------------------------------------------------------------------
# the journal


@dataclass
class Journal:
    dir: Path
    entries: list[dict] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.dir / "journal.jsonl"

    @classmethod
    def open(cls, d: Path) -> "Journal":
        j = cls(d)
        if j.path.is_file():
            j.entries = [
                json.loads(x) for x in j.path.read_text().splitlines() if x.strip()
            ]
        return j

    def append(self, entry: dict) -> None:
        entry = {"ts": time.time(), **entry}
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self.entries.append(entry)

    def pending_undos(self) -> list[dict]:
        done = {e["undone"] for e in self.entries if "undone" in e}
        return [
            e for e in self.entries if "n" in e and e.get("undo") and e["n"] not in done
        ]


def undo(k: Kube, j: Journal) -> list[str]:
    """Replay every pending undo in reverse order; returns what was undone."""
    did = []
    for e in reversed(j.pending_undos()):
        for action in e["undo"]:
            if action[0] == "kill":
                try:
                    os.killpg(int(action[1]), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                k.must(action)
        j.append({"undone": e["n"]})
        did.append(f"{e['kind']} {e.get('target', '')}".strip())
    return did


# ---------------------------------------------------------------------------
# injectors: each returns (what it did, undo actions)


def _kind_name(target: str) -> tuple[str, str]:
    kind, _, name = target.partition("/")
    if not name:
        raise HarnessError(
            f"target {target!r}: want <kind>/<name>, e.g. deploy/forge-gateway"
        )
    return kind, name


def _container(obj: dict, idx: int) -> dict:
    cs = obj.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if idx >= len(cs):
        raise HarnessError(f"{obj.get('metadata', {}).get('name')}: no container {idx}")
    return cs[idx]


def _patch(k: Kube, target: str, ops: list[dict]) -> list[str]:
    return ["patch", target, "--type=json", "-p", json.dumps(ops)]


def ready_pods(k: Kube, target: str) -> list[str]:
    obj = k.json(["get", target])
    sel = obj.get("spec", {}).get("selector", {}).get("matchLabels") or {}
    if not sel:
        raise HarnessError(f"{target} has no spec.selector.matchLabels")
    pods = k.json(
        ["get", "pods", "-l", ",".join(f"{a}={b}" for a, b in sorted(sel.items()))]
    ).get("items", [])
    out = []
    for p in pods:
        conds = (p.get("status") or {}).get("conditions") or []
        if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds):
            out.append(p["metadata"]["name"])
    return sorted(out)


def inject(
    k: Kube, inj: dict, rng: random.Random, lookup
) -> tuple[str, list[list[str]], dict]:
    kind = inj["kind"]
    if kind in LATER:
        raise HarnessError(f"the {kind} injector arrives with {LATER[kind]}")
    target = placeholders.expand(str(inj.get("target", "")), lookup, f"inject {kind}")
    idx = int(inj.get("container", 0))
    if kind == "pod-kill":
        pods = ready_pods(k, target)
        if not pods:
            raise HarnessError(f"pod-kill: {target} has no ready pod")
        pod = (
            rng.choice(pods)
            if inj.get("pick", "random-ready-replica") == "random-ready-replica"
            else pods[0]
        )
        k.must(["delete", "pod", pod, "--grace-period=0", "--force", "--wait=false"])
        return f"deleted pod {pod}", [], {"pod": pod}
    if kind == "scale-zero":
        old = int(k.json(["get", target]).get("spec", {}).get("replicas", 1))
        k.must(["scale", target, "--replicas=0"])
        return (
            f"scaled {target} from {old} to 0",
            [["scale", target, f"--replicas={old}"]],
            {"replicas": old},
        )
    if kind == "deploy-patch":
        c = _container(k.json(["get", target]), idx)
        base = f"/spec/template/spec/containers/{idx}"
        ops, undo_ops = [], []
        if "env" in inj:
            old_env = c.get("env")
            env = [dict(x) for x in (old_env or [])]
            for name, value in inj["env"].items():
                value = placeholders.expand(str(value), lookup, f"inject {kind}")
                for x in env:
                    if x.get("name") == name:
                        x.clear()
                        x.update({"name": name, "value": value})
                        break
                else:
                    env.append({"name": name, "value": value})
            ops.append({"op": "add", "path": f"{base}/env", "value": env})
            undo_ops.append(
                {"op": "add", "path": f"{base}/env", "value": old_env}
                if old_env is not None
                else {"op": "remove", "path": f"{base}/env"}
            )
        if "args" in inj:
            old_args = c.get("args")
            ops.append(
                {
                    "op": "add",
                    "path": f"{base}/args",
                    "value": [str(a) for a in inj["args"]],
                }
            )
            undo_ops.append(
                {"op": "add", "path": f"{base}/args", "value": old_args}
                if old_args is not None
                else {"op": "remove", "path": f"{base}/args"}
            )
        if not ops:
            raise HarnessError("deploy-patch needs `env` or `args`")
        k.must(_patch(k, target, ops))
        return (
            f"patched {target} ({', '.join(sorted(inj.get('env', {})) or ['args'])})",
            [_patch(k, target, undo_ops)],
            {},
        )
    if kind == "resource-limit":
        c = _container(k.json(["get", target]), idx)
        old = c.get("resources")
        new = json.loads(json.dumps(old or {}))
        new.setdefault("limits", {}).update(
            {a: str(b) for a, b in (inj.get("limits") or {}).items()}
        )
        path = f"/spec/template/spec/containers/{idx}/resources"
        k.must(_patch(k, target, [{"op": "add", "path": path, "value": new}]))
        undo_op = (
            {"op": "add", "path": path, "value": old}
            if old is not None
            else {"op": "remove", "path": path}
        )
        return (
            f"set limits {new['limits']} on {target}",
            [_patch(k, target, [undo_op])],
            {},
        )
    if kind == "config-drift":
        cm = str(inj.get("configmap", ""))
        if not cm:
            raise HarnessError("config-drift needs `configmap`")
        data = k.json(["get", f"configmap/{cm}"]).get("data") or {}
        new = {
            a: placeholders.expand(str(b), lookup, "config-drift")
            for a, b in (inj.get("set") or {}).items()
        }
        restore = {a: data.get(a) for a in new}  # None removes a key that was not there
        k.must(
            [
                "patch",
                f"configmap/{cm}",
                "--type=merge",
                "-p",
                json.dumps({"data": new}),
            ]
        )
        undo_actions = [
            [
                "patch",
                f"configmap/{cm}",
                "--type=merge",
                "-p",
                json.dumps({"data": restore}),
            ]
        ]
        if target:
            k.must(["rollout", "restart", target])
            undo_actions.append(["rollout", "restart", target])
        return (
            f"changed configmap/{cm} keys {sorted(new)}"
            + (f", restarted {target}" if target else ""),
            undo_actions,
            {},
        )
    raise HarnessError(f"unknown injector {kind!r}")


def start_loadgen(argv: list[str], cwd: Path, env: dict, log: Path) -> int:
    with open(log, "w") as f:
        p = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return p.pid


# ---------------------------------------------------------------------------
# grading


def query_range(
    base: str, query: str, start: float, end: float, step: float
) -> tuple[list | None, str]:
    from . import web

    r = web.get(
        f"{base.rstrip('/')}/api/v1/query_range",
        {
            "query": query,
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "step": f"{step:g}",
        },
        timeout=20,
    )
    if r.status != 200:
        return None, r.error or f"HTTP {r.status}: {r.text[:200]}"
    try:
        return r.json()["data"]["result"] or [], ""
    except (ValueError, KeyError, TypeError):
        return None, f"unexpected Prometheus reply: {r.text[:200]}"


def true_times(series: list) -> list[float]:
    return sorted({float(v[0]) for s in series for v in s.get("values", [])})


def final_run_start(times: list[float], end: float, step: float) -> float | None:
    """Start of the continuous run of true samples that reaches `end`."""
    if not times or end - times[-1] > 1.5 * step:
        return None
    start = times[-1]
    for t in reversed(times[:-1]):
        if start - t > 1.5 * step:
            break
        start = t
    return start


def postmortem_errors(
    learner: Path, spec: dict, what: str = "postmortem"
) -> tuple[Path | None, list[str]]:
    pat = str(spec.get("path", ""))
    if not pat:
        return None, []
    glob = pat.replace("{date}", "*")
    hits = sorted(learner.glob(glob), key=lambda p: p.stat().st_mtime)
    if not hits:
        return None, [f"no {what} at {pat}"]
    p = hits[-1]
    text = p.read_text(errors="replace")
    heads = {
        m.group(1).strip().lower()
        for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M)
    }
    missing = [s for s in spec.get("sections", []) if s.lower() not in heads]
    return p, [
        f"{p.relative_to(learner)} lacks section(s): {', '.join(missing)}"
    ] if missing else []
