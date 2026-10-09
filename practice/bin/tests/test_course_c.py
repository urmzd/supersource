"""C modules: compiling stubs, two builds, counting allocator, symbol drift."""

import subprocess


def _syntax_ok(ss, unit):
    inc = ss.learner / "contracts/c/include"
    r = subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            f"-I{inc}",
            "-fsyntax-only",
            str(ss.learner / unit),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_stub_compiles_fails_then_reference_passes(ss):
    ss.init()
    ss("start", "rt.90", rc=0)
    text = (ss.learner / "c/src/runtime/abi.c").read_text()
    assert (
        "return TL_EUNSUPPORTED;" in text
        and "abort();" in text
        and "return NULL;" in text
    )
    _syntax_ok(ss, "c/src/runtime/abi.c")
    out = ss("check", "rt.90", rc=1).out
    assert "counting allocator unavailable" in out and "0 passed, 4 failed" in out
    ss.implement("c/src/runtime/abi.c")
    assert "PASS rt.90" in ss("check", "rt.90", rc=0).out


def test_leaks_symbols_and_staleness(ss):
    ss.init()
    for mid, unit in (
        ("rt.90", "c/src/runtime/abi.c"),
        ("rt.91", "c/src/runtime/demo.c"),
    ):
        ss("start", mid, rc=0)
        ss.implement(unit)
    ss("check", "rt.90", rc=0)
    assert "ok   rt.90 smoke: status_strings" in ss("check", "rt.91", rc=0).out
    demo = ss.learner / "c/src/runtime/demo.c"
    good = demo.read_text()
    demo.write_text(
        good.replace("tl_free(b); /* the half-built object must not leak */", "")
    )
    out = ss("check", "rt.91", rc=1).out
    assert (
        "leak: 1 allocation(s) still live" in out
        and "FAIL alloc_failure_is_clean" in out
    )
    demo.write_text(good + "\nint tl_helper(void) { return 1; }\n")
    assert (
        "exports tl_helper, which no contract header declares"
        in ss("check", "rt.91", rc=4).out
    )
    demo.write_text(good)
    abi = ss.learner / "c/src/runtime/abi.c"
    abi.write_text(abi.read_text() + "\n/* touched */\n")
    assert "rt.91 needs rt.90 (stale)" in ss("check", "rt.91", rc=3).out
