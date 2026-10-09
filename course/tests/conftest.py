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
