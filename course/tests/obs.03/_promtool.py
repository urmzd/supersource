"""promtool for the observability checks (obs.03, obs.04).

`promtool` from your PATH when it is there; otherwise the pinned official
image prom/prometheus (Apache-2.0), which ships promtool, run with no network
and only the check's scratch directory mounted. Docker is required from
Pass 1, so every learner has one of the two.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# prom/prometheus v3.5.0 (an LTS release), pinned by digest.
IMAGE = "prom/prometheus:v3.5.0@sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996"


def run(c, work: Path, args: list[str], files: dict[str, str], timeout: float = 120):
    """Write `files` into `work` (a fresh directory) and run `promtool <args>`
    there. Returns the CompletedProcess (never raises on a nonzero exit)."""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for name, text in files.items():
        (work / name).write_text(text)
    if shutil.which("promtool"):
        return c.sh(["promtool", *args], cwd=work, timeout=timeout, check=False)
    c.need_docker()
    argv = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{work}:/w",
        "-w",
        "/w",
        "--entrypoint",
        "promtool",
        IMAGE,
        *args,
    ]
    return c.sh(argv, timeout=timeout, check=False)
