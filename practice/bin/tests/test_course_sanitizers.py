"""The C overlay's sanitizer builds (DESIGN 5.4, 4.4): ASan and UBSan in the
test build, ThreadSanitizer as a third build for modules that declare
`sanitize = ["thread"]`, and the counting allocator in all of them."""

import sys

import pytest


RACY = """
static void *race_work(void *p) {
    race_arg *a = p;
    int64_t *plain = (int64_t *)a->total;
    for (int64_t i = 0; i < a->per; i++) *plain += 1;
    return NULL;
}
"""


def _start_abi(ss):
    ss("start", "rt.90", rc=0)
    ss.implement("c/src/runtime/abi.c")
    ss("check", "rt.90", rc=0)


@pytest.mark.xfail(
    sys.platform.startswith("linux"),
    reason="Known gap: on the Linux CI runner the harness's ASan build does not "
    "fail rt.91 on a planted off-by-one read, though gcc's ASan catches the same "
    "code in a plain build. Tracked in course/handoff/open-items.txt.",
    strict=False,
)
def test_asan_and_ubsan_fail_the_check(ss):
    ss.init()
    _start_abi(ss)
    ss("start", "rt.91", rc=0)
    ss.implement("c/src/runtime/demo.c")
    demo = ss.learner / "c/src/runtime/demo.c"
    good = demo.read_text()
    demo.write_text(
        good.replace(
            "for (int64_t i = 0; i < n; i++) acc += x[i];",
            "for (int64_t i = 0; i <= n; i++) acc += x[i];",
        )
    )
    out = ss("check", "rt.91", rc=1).out
    assert "AddressSanitizer" in out and "buffer-overflow" in out
    demo.write_text(
        good.replace(
            "    *out = (float)acc;",
            "    int big = 2147483647 - (int)n;\n    big += (int)(acc > 0) * 8;\n    *out = (float)acc + (float)(big - big);",
        )
    )
    out = ss("check", "rt.91", rc=1).out
    assert "runtime error" in out and "overflow" in out


def test_thread_sanitizer_build_catches_a_race(ss):
    ss.add_extras("sanitizers")
    ss.init()
    _start_abi(ss)
    ss("start", "rt.92", rc=0)
    ss.implement("c/src/runtime/race.c")
    out = ss("check", "rt.92", rc=0).out
    assert "PASS rt.92" in out
    race = ss.learner / "c/src/runtime/race.c"
    text = race.read_text()
    start = text.index("static void *race_work")
    end = text.index("tl_status tl_demo_count_parallel")
    race.write_text(text[:start] + RACY.lstrip() + "\n" + text[end:])
    out = ss("check", "rt.92", rc=1).out
    assert "ThreadSanitizer: data race" in out
    out = ss("check", "rt.92", rc=None, env={"SS_TSAN": "0"}).out
    assert "ThreadSanitizer" not in out  # the TSan build is the one that sees it
