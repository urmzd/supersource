"""The frozen helpers in course/tests/_lib (D35) are part of the harness:
every module's verdict depends on them, so they are tested here."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
np = pytest.importorskip("numpy")

from _lib.close import assert_close, assert_close_bounded, tolerances  # noqa: E402
from _lib.gradcheck import gradcheck  # noqa: E402
from _lib.pcg32 import PCG32, splitmix64  # noqa: E402


def test_pcg32_reference_stream():
    # pcg32-demo from the PCG reference implementation: seed 42, sequence 54.
    r = PCG32(42, 54)
    assert [r.next_u32() for _ in range(6)] == [
        0xA15C02B7,
        0x7B47F409,
        0xBA1D3330,
        0x83D2F293,
        0xBFA4784B,
        0xCBED606E,
    ]


def test_pcg32_uniform_and_below():
    r = PCG32(0)
    us = [r.uniform() for _ in range(1000)]
    assert all(0.0 <= u < 1.0 for u in us)
    assert 0.45 < sum(us) / len(us) < 0.55
    assert {r.below(3) for _ in range(200)} == {0, 1, 2}
    assert PCG32(7).split(1).next_u32() == PCG32(7).split(1).next_u32()
    assert PCG32(7).split(1).next_u32() != PCG32(7).split(2).next_u32()


def test_splitmix64_reference():
    # First output of SplitMix64 seeded with 0 (Vigna's reference).
    assert splitmix64(0)[1] == 0xE220A8397B1DCDAF


def test_close_tolerances():
    assert tolerances("float32") == (1e-5, 1e-6)
    assert_close([1.0, 2.0], [1.0, 2.0 + 1e-11])
    with pytest.raises(AssertionError, match="first at index"):
        assert_close([1.0, 2.0], [1.0, 2.001])
    assert_close(np.float32([1.0]), np.float32([1.000001]))
    assert_close([float("nan")], [float("nan")])
    with pytest.raises(AssertionError, match="shape"):
        assert_close([1.0], [1.0, 2.0])
    assert_close_bounded([100.0], [100.0 + 100 * 1e-5 * 9], k=100, dtype="float32")


def test_gradcheck_catches_a_wrong_gradient():
    x = np.array([0.5, -1.25, 2.0])

    def f(v):
        return float(np.sum(v**3))

    gradcheck(f, [x], [3 * x**2])
    with pytest.raises(AssertionError, match="gradient differs"):
        gradcheck(f, [x], [2 * x**2])
