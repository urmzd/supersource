"""Scale a vector by a constant (fixture module M90.1)."""


def scale(xs: list[float], k: float) -> list[float]:
    """Return a new list [k * x for x in xs]; the input is left unchanged."""
    # SOLUTION-BEGIN M90.1
    return [k * x for x in xs]
    # SOLUTION-END
