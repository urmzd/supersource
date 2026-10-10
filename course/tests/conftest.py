"""Shared pytest wiring for every Python course test (DESIGN 5.9).

`ss check` runs pytest with --confcutdir at course/tests, so this file is
always loaded. It registers the Hypothesis profile `ss` (derandomized, no
example database, no deadline) when Hypothesis is installed, which keeps
property tests reproducible and stops them writing into the learner's repo.
"""

try:
    from hypothesis import settings
except ImportError:  # Hypothesis is optional for modules without property tests
    settings = None

if settings is not None:
    settings.register_profile(
        "ss", derandomize=True, database=None, deadline=None, max_examples=100
    )
    settings.load_profile("ss")


def pytest_configure(config):
    """Print test ids relative to course/tests (`lang.03/test_x.py::test_y`).

    ss runs pytest from the learner's repo with --rootdir at course/tests,
    so pytest would otherwise print each failing test as a long
    `../../..` path from the learner's repo to the course tree."""
    config.cwd_relative_nodeid = lambda nodeid: nodeid
