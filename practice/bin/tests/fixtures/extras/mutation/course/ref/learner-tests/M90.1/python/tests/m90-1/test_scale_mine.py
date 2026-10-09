from tinyllm.demo.scale import scale


def test_scales_each_element():
    assert scale([1.0, -2.0, 0.5], 2.0) == [2.0, -4.0, 1.0]


def test_does_not_modify_input():
    xs = [1.0, 2.0]
    scale(xs, 3.0)
    assert xs == [1.0, 2.0]
