"""L1 normalization (fixture module M90.2)."""

from tinyllm.demo.scale import scale


def l1_normalize(xs: list[float]) -> list[float]:
    """Divide xs by sum(|x|) so the absolute values sum to 1."""
    # SOLUTION-BEGIN M90.2
    n = len(xs)
    total = sum(abs(x) for x in xs)
    if total == 0.0:
        raise ValueError("l1_normalize: all-zero vector has no direction")
    return scale(xs, 1.0 / total)
    # SOLUTION-END
