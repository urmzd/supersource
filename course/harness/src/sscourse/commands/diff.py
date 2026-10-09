"""ss diff <ID> [--spoil]   your units against the reference, markers dropped

Meant for after you pass. Before a pass it refuses unless --spoil, which
records the module as `spoiled` in the ledger."""

from __future__ import annotations

import argparse
import difflib

from .. import EXIT_FAIL, ctx, ledger, markers, units
from ..session import open_session


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ss diff")
    ap.add_argument("id")
    ap.add_argument("--spoil", action="store_true")
    a = ap.parse_args(argv)
    s = open_session()
    m = s.module(a.id)
    v = ledger.latest(s.learner, m.id)
    if not (v and v.get("result") == "pass"):
        if not a.spoil:
            ctx.say(
                f"{m.id} has not passed yet; the diff would spoil it. `ss diff {m.id} --spoil` records it as spoiled."
            )
            return EXIT_FAIL
        ledger.event(s.learner, m.id, "spoiled", via="diff")
    ctx.say(
        f"{ctx.DIM}Reference on the right. Different is not the same as worse.{ctx.RST}"
    )
    for u in m.owned:
        ref = markers.drop_markers(
            units.ref_text(s.course, s.reg, u, m.id)
        ).splitlines()
        mine = (units.learner_text(s.learner, u) or "").splitlines()
        for line in difflib.unified_diff(
            mine, ref, f"yours: {u}", f"reference: {u}", lineterm=""
        ):
            ctx.say(line)
    return 0
