import tinyllm_rs


def test_hand_example():
    # WHY: the extension module imports and calls into Rust: 2 + 3 = 5.
    # KIND: unit
    assert tinyllm_rs.add(2, 3) == 5


def test_kahan_through_python():
    # WHY: a Python list crosses into Rust and the compensated sum comes back.
    # KIND: differential
    assert tinyllm_rs.kahan([1.0] + [1e-16] * 10) == 1.0 + 1e-15
