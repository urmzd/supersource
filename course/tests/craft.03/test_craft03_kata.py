"""craft.03 course tests: your kata, primers/craft.03/failpoints.py.

Rung R0 for the kata itself: these are annotated exemplars of the tests the
chapter asks you to write, aimed at YOUR code. Each names why it exists
(WHY), its kind (KIND), the planted faults it kills (CATCHES,
course/mutants/craft.03), and the chapter section.

The worked example (chapter section 3): the spec
"train/after-step=3*crash; data/fetch = error(disk full) ;x=off;"
parses to three rules; evaluating train/after-step five times fires only on
the third evaluation; x never fires and is never counted.
"""

from __future__ import annotations

import pytest

from failpoints import ACTIONS, Failpoints, Rule, parse, sleep_seconds

HAND = "train/after-step=3*crash; data/fetch = error(disk full) ;x=off;"


def test_hand_example_spec():
    # WHY: the chapter's worked example, rule by rule: whitespace around a
    #      name or an action is ignored, a trailing ";" is an empty entry,
    #      error's message is kept as written.
    # KIND: unit
    # CATCHES: s04, s10
    # CHAPTER: craft.03 section 3, Worked example by hand
    assert parse(HAND) == {
        "train/after-step": Rule("crash", "", 3),
        "data/fetch": Rule("error", "disk full", 0),
        "x": Rule("off", "", 0),
    }


def test_hand_example_evaluations():
    # WHY: N* fires on the Nth evaluation only, not from the Nth on; a rule
    #      without N fires every time; off neither fires nor counts.
    # KIND: unit
    # CATCHES: s01, s02, m02
    # CHAPTER: craft.03 section 3, Worked example by hand
    fp = Failpoints(HAND)
    fired = [fp.evaluate("train/after-step") for _ in range(5)]
    assert fired == [None, None, Rule("crash", "", 3), None, None]
    assert fp.count("train/after-step") == 5
    assert fp.evaluate("data/fetch") == Rule("error", "disk full", 0)
    assert fp.evaluate("data/fetch") == Rule("error", "disk full", 0)
    assert fp.evaluate("x") is None and fp.count("x") == 0
    assert fp.evaluate("nowhere") is None and fp.count("nowhere") == 0


def test_counts_are_per_name():
    # WHY: each failpoint counts its own evaluations; one shared counter
    #      makes "fire on the 2nd train step" depend on how often an
    #      unrelated failpoint was reached.
    # KIND: unit
    # CATCHES: s08
    # CHAPTER: craft.03 section 5, Pitfalls, item 4
    fp = Failpoints("a=2*panic;b=2*panic")
    assert fp.evaluate("a") is None
    assert fp.evaluate("b") is None
    assert fp.evaluate("a") == Rule("panic", "", 2)
    assert (fp.count("a"), fp.count("b")) == (2, 1)


def test_sleep_durations():
    # WHY: a duration is digits then ms or s; 50ms is 0.05 seconds, not 50
    #      (a test that sleeps 50 s instead of 50 ms times out the suite).
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: craft.03 section 5, Pitfalls, item 5
    assert sleep_seconds("50ms") == 0.05
    assert sleep_seconds("2s") == 2.0
    assert sleep_seconds("1.5s") == 1.5
    assert parse("s=sleep(20ms)")["s"] == Rule("sleep", "20ms", 0)
    for bad in ("50", "ms", "5m", "-1s", ""):
        with pytest.raises(ValueError):
            sleep_seconds(bad)


@pytest.mark.parametrize(
    "spec",
    [
        "noequals",
        "=crash",
        "a=explode",
        "a=0*crash",
        "a=crash;a=off",
        "a=crash(now)",
        "a=sleep",
        "a=sleep(soon)",
        "a=b=crash",
    ],
)
def test_parse_rejects(spec):
    # WHY: a spec is typed by hand into an env var: every malformed entry is
    #      an error that names it, never a silently ignored failpoint. The
    #      first "=" splits name from action, so "a=b=crash" is the name "a"
    #      with the action "b=crash", which is not an action.
    # KIND: boundary
    # CATCHES: s03, s06, s07, s09, s11
    # CHAPTER: craft.03 section 5, Pitfalls, items 1 to 3
    with pytest.raises(ValueError):
        parse(spec)


def test_every_action_parses():
    # WHY: the five actions of the spec, each in its own form.
    # KIND: unit
    # CATCHES: m01, s10
    # CHAPTER: craft.03 section 2, Principles (the spec)
    assert set(ACTIONS) == {"crash", "panic", "error", "sleep", "off"}
    got = parse(
        "a=crash;b=panic;c=error;d=error( boom );e=sleep(1s);f=off;g=7*error(x)"
    )
    assert got["c"] == Rule("error", "", 0) and got["d"] == Rule("error", "boom", 0)
    assert got["g"] == Rule("error", "x", 7) and got["e"].arg == "1s"
    assert parse("") == {} and parse(" ; ;") == {}
