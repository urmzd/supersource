import numpy as np
from tinyllm.post.dpo import dpo_loss, ipo_loss


def test_hand_dpo_loss():
    # WHY: this numerical example pins the reference-adjusted preference margin and beta scale.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L12.2 section 3, Worked example by hand
    got = dpo_loss([1.2], [0.0], [0.2], [0.0], beta=0.1)
    assert abs(got - 0.6443966600735709) < 1e-12


def test_equal_policy_reference_is_log_two():
    # WHY: equal policy and reference odds must cancel, leaving the logistic loss at log two.
    # KIND: boundary
    # CHAPTER: L12.2 section 4, The interface
    assert abs(dpo_loss([0.3], [-0.2], [0.3], [-0.2]) - np.log(2)) < 1e-12


def test_ipo_variant():
    # WHY: IPO uses a squared margin target of one over two beta, distinct from DPO's logistic loss.
    # KIND: unit
    # CATCHES: s01, s02
    # CHAPTER: L12.2 section 4, The interface
    assert abs(ipo_loss([1.2], [0.0], [0.2], [0.0], beta=0.1) - 16.0) < 1e-12
