"""Scale a vector by a constant (fixture module M90.1, upgraded by M90.3)."""

import math


def scale(xs: list[float], k: float) -> list[float]:
    """Return a new list [k * x for x in xs]; a non-finite k is a ValueError."""
    # SOLUTION-BEGIN M90.3
    if not math.isfinite(k):
        raise ValueError(f"scale: k must be finite, got {k!r}")
    return [k * x for x in xs]
    # SOLUTION-END
