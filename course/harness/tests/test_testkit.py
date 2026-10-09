"""The Go, Rust, and C halves of the testkit (course/testkit, DESIGN 4.4) pass
their own tests: chaosproxy, otlpsink, promscrape, clock, effects, KillLoop,
failpoints, faketool (Go); failpoints and the fake clock (Rust); the
header-only C failpoint."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

COURSE = Path(__file__).resolve().parents[2]
KIT = COURSE / "testkit"


@pytest.mark.skipif(shutil.which("go") is None, reason="go not installed")
def test_go_testkit():
    env = {**os.environ, "GOWORK": "off", "GOFLAGS": "-count=1"}
    p = subprocess.run(
        ["go", "test", "./..."],
        cwd=KIT / "go",
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stdout + p.stderr


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed")
def test_rust_testkit(tmp_path):
    env = {**os.environ, "CARGO_TARGET_DIR": str(tmp_path / "target")}
    p = subprocess.run(
        [
            "cargo",
            "test",
            "-q",
            "--offline",
            "--manifest-path",
            str(KIT / "rust" / "tl-testkit" / "Cargo.toml"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert p.returncode == 0, p.stdout + p.stderr


C = r"""
#include <stdio.h>
#include "tinyllm/failpoint.h"
int main(void) {
    int a = tl_failpoint("a"), b1 = tl_failpoint("b"), b2 = tl_failpoint("b"), c = tl_failpoint("c"), s = tl_failpoint("s");
    printf("%d %d %d %d %d\n", a, b1, b2, c, s);
    fflush(stdout);
    tl_failpoint("k");
    puts("not reached");
    return 0;
}
"""


def test_c_failpoint_header(tmp_path):
    src = tmp_path / "fp.c"
    src.write_text(C)
    exe = tmp_path / "fp"
    inc = COURSE / "contracts" / "c" / "include"
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-pedantic",
            "-Wall",
            "-Werror",
            f"-I{inc}",
            str(src),
            "-o",
            str(exe),
        ],
        check=True,
    )
    p = subprocess.run(
        [str(exe)],
        capture_output=True,
        text=True,
        env={"TL_FAILPOINTS": "a=error;b=2*error; c=off;s=sleep(1);k=crash"},
    )
    assert p.returncode == 137 and p.stdout == "1 0 1 0 0\n"
    p = subprocess.run([str(exe)], capture_output=True, text=True, env={})
    assert p.returncode == 0 and "not reached" in p.stdout
