#!/usr/bin/env bash
# Threads need -pthread, which the harness's default C compile line does not
# pass. An exercise that ships an executable run.sh gets run by it instead.
#
# Built into a temp dir so nothing lands next to the source.
set -euo pipefail
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
cc -std=c11 -Wall -Wextra -pthread -o "$out/pool" ./*.c
"$out/pool"
