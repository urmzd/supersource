"""ss check <ID> [--ref-deps[=all|ID,...]] [--no-cumulative] [--kind K] [--json] [--seed N]
ss check --all [--ci]

--all checks every started module in pass order. --ci is what the learner's
own CI runs (`ss course ci` prints that recipe): it forbids --ref-deps, skips
interactive proof rubrics (reported `self`), and runs practice checks with
SS_SMOKE=1 (no cluster tier; a practice verdict under SS_SMOKE is recorded with
mode "smoke"). With nothing started it is trivially green.

Order (DESIGN 5.3, 5.4): not started (2), contracts content hash (4), dependency
resolution (3), unit contract pre-check (4), cumulative smoke of every
learner-sourced dep, then the module's course tests through the overlay.
The exit code is the verdict: 0 pass, 1 fail, 2 not started, 3 blocked by deps,
4 contract drift, 5 harness or toolchain."""

from __future__ import annotations

import argparse
import json
import os

from .. import (
    EXIT_BLOCKED,
    EXIT_DRIFT,
    EXIT_FAIL,
    EXIT_HARNESS,
    EXIT_NOT_STARTED,
    EXIT_PASS,
    HarnessError,
    catalog,
    ctx,
    learner,
    ledger,
    precheck,
    tree,
)
from ..overlay import Overlay
from ..session import Session, open_session

CODE_KINDS = ("build", "side", "practice")


def parse_ref_deps(value: str | None) -> tuple[str, set[str]]:
    """None -> ("none", {}); bare -> ("failing", {}); "all"; "A,B" -> ("list", {A, B})."""
    if value is None:
        return "none", set()
    if value in ("", "failing"):
        return "failing", set()
    if value == "all":
        return "all", set()
    return "list", {x.strip() for x in value.split(",") if x.strip()}


def resolve_sources(
    s: Session, mid: str, mode: str, chosen: set[str]
) -> tuple[dict[str, str], list[str], list[str]]:
    """sources for the target and its closure, the blocked deps, and notes."""
    closure = s.reg.closure(mid)
    unknown = chosen - set(closure)
    if unknown:
        raise HarnessError(
            f"--ref-deps names {sorted(unknown)}, which {mid} does not depend on (closure: {closure})"
        )
    sources = {mid: "learner"}
    blocked, notes = [], []
    for d in closure:
        dm = s.reg.get(d)
        if not dm.owned:
            if not ledger.fresh_pass(s.learner, s.reg, d):
                notes.append(
                    f"{d} has no code to substitute (not passed yet; nothing blocks on it)"
                )
            continue
        if mode == "all" or d in chosen:
            sources[d] = "ref"
        elif (
            ledger.fresh_pass(s.learner, s.reg, d)
            or learner.state(s.learner, s.course, s.reg, d).status == "pass"
        ):
            sources[d] = "learner"
        elif mode == "failing":
            sources[d] = "ref"
        else:
            blocked.append(d)
    return sources, blocked, notes


def _print_runs(runs, label: str, say=ctx.say) -> bool:
    ok = True
    for r in runs:
        if r.ok:
            say(f"  {ctx.GRN}PASS{ctx.RST} {label} [{r.lang}]")
        else:
            ok = False
            what = "does not compile" if not r.compiled else "FAIL"
            say(f"  {ctx.RED}{what}{ctx.RST} {label} [{r.lang}]")
            say(ctx.indent(ctx.tail(r.output, 60), 6))
    return ok


def check_one(
    s: Session,
    mid: str,
    ref_mode: str = "none",
    chosen: set[str] = frozenset(),
    cumulative: bool = True,
    kind: str | None = None,
    as_json: bool = False,
    seed: int = 0,
    quiet: bool = False,
) -> int:
    m = s.module(mid)
    say = (lambda *_: None) if as_json else ctx.say
    if m.kind in ("solve", "proof"):
        say(
            f"{ctx.YEL}skip{ctx.RST} {m.id}: {m.kind} checker arrives with B2 (DESIGN 5.5)"
        )
        return EXIT_HARNESS
    if m.kind == "drill":
        say(f"{m.id} is a drill: use `ss drill start {m.id}`")
        return EXIT_HARNESS
    later = learner.superseded(s.learner, s.course, s.reg, mid)
    prev = ledger.latest(s.learner, mid)
    if later and prev and prev.get("result") == "pass":
        say(
            f"{ctx.GRN}PASS{ctx.RST} {mid} (superseded by {later}; frozen verdict, its tests run as {later}'s regression)"
        )
        return EXIT_PASS
    if not learner.started(s.learner, s.course, s.reg, mid):
        say(f"{ctx.DIM}not started{ctx.RST} {mid}: ss start {mid}")
        return EXIT_NOT_STARTED

    def record(result: str, sources: dict | None = None, **extra) -> dict:
        srcs = {k: v for k, v in (sources or {}).items() if k != mid}
        return ledger.append(
            s.learner,
            {
                "id": mid,
                "kind": m.kind,
                "tree": ledger.tree_hash(s.learner, s.reg, mid),
                "sources": srcs,
                "result": result,
                "assisted": any(v == "ref" for v in srcs.values()),
                "mutation": None,
                **(
                    {"mode": "smoke"}
                    if m.kind == "practice"
                    and os.environ.get("SS_SMOKE", "") not in ("", "0")
                    else {}
                ),
                "tainted": s.tree.tainted,
                "course_sha": s.tree.sha[:12],
                **extra,
            },
        )

    def finish(code: int, v: dict | None) -> int:
        if as_json:
            print(json.dumps({"id": mid, "exit": code, "verdict": v}))
        return code

    drift = tree.precheck_contracts(s.learner)
    if drift:
        say(f"{ctx.RED}DRIFT{ctx.RST} {mid}: " + "; ".join(drift))
        return finish(EXIT_DRIFT, record("drift", reason=drift))
    hint = tree.advise_semver(s.learner)
    if hint:
        say(f"{ctx.DIM}{hint}{ctx.RST}")

    sources, blocked, notes = resolve_sources(s, mid, ref_mode, set(chosen))
    if blocked:
        why = ", ".join(
            f"{d} ({learner.state(s.learner, s.course, s.reg, d).status})"
            for d in blocked
        )
        say(
            f"{ctx.YEL}BLOCKED{ctx.RST} {mid} needs {why}. Fix it, or rerun with --ref-deps"
        )
        return finish(EXIT_BLOCKED, record("blocked", sources, blocked=blocked))
    for n in notes:
        say(f"  {ctx.DIM}{n}{ctx.RST}")

    scratch = s.learner / ".ss" / "build" / mid / "precheck"
    errs = precheck.module(s.learner, s.course, m, scratch)
    if errs:
        say(f"{ctx.RED}DRIFT{ctx.RST} {mid}: contract pre-check failed")
        for e in errs:
            say(ctx.indent(e, 4))
        return finish(EXIT_DRIFT, record("drift", sources, reason=errs))

    deps_line = ", ".join(f"{d} {src}" for d, src in sources.items() if d != mid)
    say(
        f"{ctx.BLD}check {mid}{ctx.RST}  {m.title}"
        + (f"  {ctx.DIM}deps: {deps_line}{ctx.RST}" if deps_line else "")
    )
    ov = Overlay(
        s.reg,
        s.course,
        mid,
        sources,
        s.learner,
        s.learner / ".ss" / "overlay" / mid,
        s.learner / ".ss",
        seed=seed,
    )

    # Which deps' tests ride along: smoke tests of learner-sourced deps, and
    # the full tests of every earlier owner of a unit this module took over.
    regress: list[tuple[str, list[str] | None]] = []
    if cumulative:
        for d in s.reg.closure(mid):
            if sources.get(d) == "learner" and s.reg.get(d).smoke:
                regress.append((d, s.reg.get(d).smoke))
        for u in m.upgrades:
            chain = s.reg.unit_chain(u)
            for prev_id in chain[: chain.index(mid)]:
                if prev_id not in [r[0] for r in regress]:
                    regress.append((prev_id, None))
                else:
                    regress = [(a, None if a == prev_id else b) for a, b in regress]
    rust_ids = [mid] + [d for d, _ in regress]

    ok = True
    regressions: list[str] = []
    for d, names in regress:
        if not ov.test_langs(d):
            continue
        runs = ov.run_tests(d, names, rust_tests=rust_ids)
        label = f"{d} " + (
            f"smoke: {', '.join(names)}" if names else f"tests (taken over by {mid})"
        )
        if not all(r.ok for r in runs):
            ok = False
            regressions.append(d)
            say(
                f"  {ctx.RED}REGRESSION{ctx.RST} {d} ({'smoke: ' + ', '.join(names) if names else 'full suite'})"
            )
            _print_runs([r for r in runs if not r.ok], label, say)
        elif not quiet:
            say(f"  {ctx.DIM}ok   {label}{ctx.RST}")

    names = None
    if kind:
        names = [
            t.name
            for t in catalog.collect(catalog.files_for(s.course, mid, ov.test_dir(mid)))
            if kind in t.kinds
        ]
        if not names:
            say(f"  no {mid} test has KIND {kind}")
            return finish(EXIT_HARNESS, None)
    runs = ov.run_tests(mid, names, rust_tests=rust_ids)
    if not runs:
        say(
            f"  {ctx.YEL}no course tests found for {mid}{ctx.RST} (looked in {ov.test_dir(mid)})"
        )
        return finish(EXIT_HARNESS, None)
    ok = _print_runs(runs, mid, say) and ok

    if m.learner_tests:
        say(
            f"  {ctx.DIM}mutation grade of your tests: `ss mutate {mid}` arrives with B3; not part of this verdict yet{ctx.RST}"
        )
    result = "pass" if ok else "fail"
    v = record(
        result,
        sources,
        regressions=regressions,
        **({"kind_filter": kind} if kind else {}),
    )
    tag = " (assisted)" if v["assisted"] else ""
    if ok:
        say(f"{ctx.GRN}PASS{ctx.RST} {mid}{tag}")
    else:
        say(
            f"{ctx.RED}FAIL{ctx.RST} {mid}{tag}"
            + (f"  regressions: {', '.join(regressions)}" if regressions else "")
        )
        say(f"  {ctx.DIM}ss tests {mid}  reads what each test checks and why{ctx.RST}")
    return finish(EXIT_PASS if ok else EXIT_FAIL, v)


def check_all(s: Session, ci: bool, as_json: bool) -> int:
    rows = []
    worst = 0
    if ci:
        # The learner's CI has no cluster: practice checks skip their cluster
        # tier (named in their output), which the milestone kind steps grade.
        os.environ["SS_SMOKE"] = "1"
    for m in s.reg.ordered():
        if not learner.started(s.learner, s.course, s.reg, m.id):
            continue
        if m.kind in ("solve", "proof", "drill"):
            # --ci never runs an interactive rubric: a proof is reported `self` and skipped.
            note = {
                "proof": "self-graded rubric, skipped in --ci"
                if ci
                else "proof rubric arrives with B2",
                "solve": "answer checker arrives with B2",
                "drill": "graded by `ss drill end`",
            }[m.kind]
            rows.append(
                (m.id, "self" if (ci and m.kind == "proof") else "skipped", note)
            )
            continue
        code = check_one(s, m.id, "none", set(), True, None, False, quiet=True)
        st = {0: "pass", 1: "fail", 2: "not started", 3: "blocked", 4: "drift"}.get(
            code, "error"
        )
        rows.append((m.id, st, ""))
        if code != EXIT_NOT_STARTED:
            worst = max(worst, code)
    if as_json:
        print(json.dumps([{"id": a, "status": b, "note": c} for a, b, c in rows]))
    else:
        ctx.say(
            f"\n{ctx.BLD}ss check --all{' --ci' if ci else ''}{ctx.RST}: {len(rows)} started module(s)"
        )
        for a, b, c in rows:
            ctx.say(f"  {a:<12} {b:<10} {c}")
        if not rows:
            ctx.say("  nothing started yet: trivially green")
    return worst


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ss check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--ref-deps", nargs="?", const="", default=None)
    ap.add_argument("--no-cumulative", action="store_true")
    ap.add_argument("--kind")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    s = open_session()
    with ctx.lock(s.learner / ".ss"):
        return _run(s, a, ap)


def _run(s: Session, a, ap) -> int:
    if a.all:
        if a.ref_deps is not None and a.ci:
            raise HarnessError("--ci forbids --ref-deps")
        return check_all(s, a.ci, a.json)
    if not a.id:
        ap.print_usage()
        return EXIT_HARNESS
    mode, chosen = parse_ref_deps(a.ref_deps)
    return check_one(s, a.id, mode, chosen, not a.no_cumulative, a.kind, a.json, a.seed)
