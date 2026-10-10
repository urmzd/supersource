"""Python modules: start -> stub compiles and fails -> reference passes -> verdicts."""

import py_compile


def test_start_stub_fails_then_reference_passes(ss):
    ss.init()
    out = ss("start", "M90.1", rc=0).out
    assert (
        "wrote     python/tinyllm/demo/scale.py" in out
        and "python/pyproject.toml" in out
    )
    unit = ss.learner / "python/tinyllm/demo/scale.py"
    assert 'raise NotImplementedError("M90.1")' in unit.read_text()
    py_compile.compile(str(unit), doraise=True)  # the stub compiles
    assert "NotImplementedError: M90.1" in ss("check", "M90.1", rc=1).out
    assert ss.verdicts("M90.1")[-1]["result"] == "fail"
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    assert "PASS M90.1" in ss("check", "M90.1", rc=0).out
    v = ss.verdicts("M90.1")[-1]
    assert (
        v["result"] == "pass"
        and v["tree"].startswith("sha256:")
        and v["assisted"] is False
    )
    assert "pass      M90.1" in ss("status", rc=0).out
    unit.write_text(unit.read_text() + "\n# edited\n")
    assert "stale     M90.1" in ss("status", rc=0).out


def test_blocked_then_ref_deps_is_assisted(ss):
    ss.init()
    ss("start", "M90.2", rc=0)
    ss.implement("python/tinyllm/demo/norm.py")
    out = ss("check", "M90.2", rc=3).out
    assert "BLOCKED M90.2 needs M90.1 (todo)" in out
    out = ss("check", "M90.2", "--ref-deps", rc=0).out
    assert "deps: M90.1 ref" in out and "PASS M90.2 (assisted)" in out
    v = ss.verdicts("M90.2")[-1]
    assert v["assisted"] and v["sources"] == {"M90.1": "ref"}
    assert "assisted  M90.2" in ss("status", rc=0).out
    assert "does not depend on" in ss("check", "M90.2", "--ref-deps=ds.90", rc=5).out
    # The C demo is a standalone module now: no Python module depends on it.
    assert "does not depend on" in ss("check", "M90.2", "--ref-deps=rt.91", rc=5).out


def test_upgrade_takes_over_and_supersedes(ss):
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss("check", "M90.1", rc=0)
    out = ss("start", "M90.3", rc=0).out
    assert (
        "M90.3 takes over python/tinyllm/demo/scale.py from M90.1" in out
        and "kept" in out
    )
    # A broken upgrade fails the earlier owner's tests as a regression.
    unit = ss.learner / "python/tinyllm/demo/scale.py"
    unit.write_text(
        "import math\n\n\ndef scale(xs, k):\n    if not math.isfinite(k):\n        raise ValueError(k)\n"
        "    return [k * x + 1 for x in xs]\n"
    )
    out = ss("check", "M90.3", rc=1).out
    assert "REGRESSION M90.1 (full suite)" in out
    ss.implement("python/tinyllm/demo/scale.py")
    out = ss("check", "M90.3", rc=0).out
    assert "ok   M90.1 tests (taken over by M90.3)" in out
    assert "superseded by M90.3" in ss("check", "M90.1", rc=0).out
    assert "superseded by M90.3" in ss("status", rc=0).out
