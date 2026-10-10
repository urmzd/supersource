import math

import pytest
from tinyllm.demo.scale import scale


@pytest.mark.parametrize("k", [math.inf, -math.inf, math.nan])
def test_rejects_nonfinite_k(k):
    # WHY: a non-finite factor turns every element into inf or nan, which the
    #      sum in M90.2 would then carry silently into every later stage.
    # KIND: boundary
    with pytest.raises(ValueError):
        scale([1.0, 2.0], k)
