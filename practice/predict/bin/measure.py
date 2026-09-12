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
  process gets killed. On Linux, GNU time measures the command from a small
  native launcher so the Python harness's inherited RSS is not counted. On
  other platforms, ``getrusage(RUSAGE_CHILDREN)`` supplies the high-water
  mark across our runs.

The units of ``ru_maxrss`` are famously platform-dependent: bytes on macOS,
kilobytes on Linux and FreeBSD. Getting that wrong is a factor-of-1024 error in a
number people then write into a budget file, so it is normalised here once.

The command's own output is discarded. This is a measurement tool, and printing
snippet output would spoil the exercise it is measuring.
"""

from __future__ import annotations

from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def normalize_rss_to_kb(raw: int) -> int:
    """ru_maxrss is bytes on macOS and kilobytes on Linux/FreeBSD."""
    if sys.platform == "darwin":
        return raw // 1024
    return raw


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    runs = int(sys.argv[1])
    command = sys.argv[2:]

    if runs < 1:
        print("measure: runs must be positive", file=sys.stderr)
        return 2

    if sys.platform == "linux" and not Path("/usr/bin/time").is_file():
        print("measure: Linux requires GNU time at /usr/bin/time", file=sys.stderr)
        return 2

    best_ms: float | None = None
    peak_kb = 0
    with tempfile.TemporaryDirectory(prefix="lr-measure-") as directory:
        rss_file = Path(directory) / "rss"
        for _ in range(runs):
            # Linux preserves pre-exec RSS: measuring Python's immediate child
            # includes the harness's memory. GNU time forks the actual command
            # after exec has replaced Python, and reports that child's RSS.
            measured_command = command
            if sys.platform == "linux":
                measured_command = [
                    "/usr/bin/time",
                    "-f",
                    "%M",
                    "-o",
                    str(rss_file),
                    "--",
                    *command,
                ]
            started = time.perf_counter()
            completed = subprocess.run(
                measured_command,
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
            if sys.platform == "linux":
                peak_kb = max(peak_kb, int(rss_file.read_text().strip()))

    if sys.platform != "linux":
        peak_kb = normalize_rss_to_kb(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        )
    print(f"{round(best_ms)}\t{peak_kb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
