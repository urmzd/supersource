"""ss course init --name <system> [--at DIR]   create the learner repo (DESIGN 2.15)
ss course where                             print the learner repo and course tree in use
ss course ci [--upstream URL]               print the learner CI recipe (DESIGN 5.13, craft.01)"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import ctx, learner, tree

UPSTREAM = "https://github.com/urmzd/supersource.git"

RECIPE = """\
# Learner CI recipe (course/DESIGN.md 5.13, taught in craft.01). Copy these
# steps into YOUR CI workflow; supersource ships no CI file (P2). The job needs
# git, uv, and the toolchains of the languages you have started (`ss doctor`).
# It checks out public supersource at the sha your contracts/VERSION names, so
# the course tests that grade you are the ones your contracts came with.
set -euo pipefail
SHA="$(sed -n 's/^sha = "\\(.*\\)"$/\\1/p' contracts/VERSION)"
[ -n "$SHA" ] || {{ echo "contracts/VERSION names no sha; run ss contracts sync" >&2; exit 5; }}
if [ ! -d .ss/supersource/.git ]; then
  git clone --quiet {upstream} .ss/supersource
fi
git -C .ss/supersource fetch --quiet origin
git -C .ss/supersource checkout --quiet --detach "$SHA"
SS_COURSE_HOME="$PWD" .ss/supersource/practice/bin/ss check --all --ci
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("init", "where", "ci"):
        print(__doc__)
        return 5
    if argv[0] == "ci":
        ap = argparse.ArgumentParser(prog="ss course ci")
        ap.add_argument("--upstream", default=UPSTREAM)
        a = ap.parse_args(argv[1:])
        print(RECIPE.format(upstream=a.upstream), end="")
        return 0
    if argv[0] == "where":
        h = ctx.learner_home()
        ctx.say(
            f"learner  {h}{'' if (h / 'system.toml').is_file() else '  (not created: ss course init --name <system>)'}"
        )
        ctx.say(f"course   {ctx.live_course()}")
        if (h / "system.toml").is_file():
            ct = tree.resolve(h)
            ctx.say(
                f"tree     {ct.path}  (sha {ct.sha[:12]}{', live' if ct.live else ''}{', tainted' if ct.tainted else ''})"
            )
        return 0
    ap = argparse.ArgumentParser(prog="ss course init")
    ap.add_argument("--name", required=True)
    ap.add_argument("--at", type=Path)
    a = ap.parse_args(argv[1:])
    dest = learner.init(a.name, a.at)
    v = tree.read_version(dest / "contracts" / "VERSION")
    ctx.say(f"{ctx.GRN}created{ctx.RST} {dest}")
    ctx.say(f"  system.toml   name = {a.name}")
    ctx.say(f"  contracts/    {v.get('semver')} at {str(v.get('sha'))[:12]}")
    ctx.say("  next          ss next")
    if a.at:
        ctx.say(
            f"  {ctx.DIM}point every command at it: export SS_COURSE_HOME={dest}{ctx.RST}"
        )
    return 0
