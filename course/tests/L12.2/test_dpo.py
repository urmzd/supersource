import numpy as np
from tinyllm.post.dpo import dpo_loss, ipo_loss


def test_hand_dpo_loss():
    got = dpo_loss([1.2], [0.0], [0.2], [0.0], beta=0.1)
    assert abs(got - 0.6443966600735709) < 1e-12


def test_equal_policy_reference_is_log_two():
    assert abs(dpo_loss([0.3], [-0.2], [0.3], [-0.2]) - np.log(2)) < 1e-12


def test_ipo_variant():
    assert abs(ipo_loss([1.2], [0.0], [0.2], [0.0], beta=0.1) - 16.0) < 1e-12
