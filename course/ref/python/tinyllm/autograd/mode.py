"""Grad mode: whether ops record graph nodes (L0.1).

On by default and per thread. `no_grad()` switches it off for a `with`
block, so evaluation, sampling, and optimizer updates build no graph.

Contract: contracts/py/tinyllm/autograd/mode.pyi.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_local = threading.local()


def is_grad_enabled() -> bool:
    # SOLUTION-BEGIN L0.1
    # A thread that never entered no_grad has no attribute yet: on by default.
    return getattr(_local, "enabled", True)
    # SOLUTION-END


@contextmanager
def no_grad() -> Iterator[None]:
    # SOLUTION-BEGIN L0.1
    prev = is_grad_enabled()
    _local.enabled = False
    try:
        yield
    finally:
        # Restore the previous value, not True: no_grad blocks nest, and an
        # exception inside the block must not leave grad mode off.
        _local.enabled = prev
    # SOLUTION-END
