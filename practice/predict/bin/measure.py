#!/usr/bin/env python3
"""Measure one command's wall time and peak memory, portably.

Usage:
    measure.py <runs> <command> [args...]

Prints one tab-separated line to stdout::

    <min_wall_ms>\t<peak_rss_kb>

Why these two statistics:

* **Minimum** wall time, not mean. Every source of noise on a developer laptop
  or a CI runner (scheduling, other processes, thermal throttling) makes a run
  slower, never faster, so the minimum of several runs is the closest thing to
  the program's actual cost.
* **Maximum** resident set size, because peak memory is what decides whether a
  process gets killed. ``getrusage(RUSAGE_CHILDREN)`` reports the high-water
  mark across every child this process has reaped, which is exactly the max
  over our runs.

The units of ``ru_maxrss`` are famously platform-dependent: bytes on macOS and
BSD, kilobytes on Linux. Getting that wrong is a factor-of-1024 error in a
number people then write into a budget file, so it is normalised here once.

The command's own output is discarded. This is a measurement tool, and printing
snippet output would spoil the exercise it is measuring.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time


def normalize_rss_to_kb(raw: int) -> int:
    """ru_maxrss is bytes on macOS/BSD and kilobytes on Linux."""
    if sys.platform == "darwin" or "bsd" in sys.platform:
        return raw // 1024
    return raw


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    runs = int(sys.argv[1])
    command = sys.argv[2:]

    best_ms: float | None = None
    for _ in range(runs):
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if completed.returncode != 0:
            print(
                f"measure: {command[0]} exited {completed.returncode}",
                file=sys.stderr,
            )
            return 1
        best_ms = elapsed_ms if best_ms is None else min(best_ms, elapsed_ms)

    peak_kb = normalize_rss_to_kb(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    print(f"{round(best_ms)}\t{peak_kb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
