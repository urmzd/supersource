#!/usr/bin/env bash
# Threads need -pthread on Linux; the harness's default C++ line does not pass
# it. Built into a temp dir so nothing lands next to the source.
set -euo pipefail
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
c++ -std=c++20 -Wall -Wextra -pthread -o "$out/queue" main.cpp
"$out/queue"
