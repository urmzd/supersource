#!/usr/bin/env bash
# craft.01 reference: the three CI gates run locally, for a repo with no
# GitHub remote. system.toml declares it as `[ci] local = [["bash", "scripts/ci-local.sh"]]`,
# which is what MS-P0's ci-status step runs in that case. Set SS_BIN to your
# supersource clone's practice/bin/ss.
# SOLUTION-BEGIN craft.01
set -euo pipefail

echo "== commit-lint (HEAD)"
msg="$(mktemp)"
trap 'rm -f "$msg"' EXIT
git log -1 --format=%B HEAD > "$msg"
.githooks/commit-msg "$msg"

echo "== native-tests"
make -s -C primers/lang.02 && primers/lang.02/greet ci
make -s -C primers/lang.02 clean
(cd primers/lang.01 && uv run --quiet python -c "import bigram, worksheet")

echo "== course-check"
# The supersource checkout whose ss grades you: $SS_BIN when set (your local
# clone's practice/bin/ss), else the one the CI recipe clones (ss course ci).
ss="${SS_BIN:-.ss/supersource/practice/bin/ss}"
[ -x "$ss" ] || { echo "no ss at $ss: set SS_BIN, or clone supersource as \`ss course ci\` shows" >&2; exit 5; }
SS_COURSE_HOME="$PWD" "$ss" check --all --ci
# SOLUTION-END
