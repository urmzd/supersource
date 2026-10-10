"""Frozen course-test helpers (DESIGN D35).

Course tests assert, gradient-check, and draw random numbers only through
these modules, never through the learner's own `gradcheck` (M04.1),
tolerance helpers (M09.3), or PCG32 (M06.3), so a wrong learner helper can
fail only its own module. Changing a helper's behavior is a course-wide
event: every module's verdict depends on it.
"""
