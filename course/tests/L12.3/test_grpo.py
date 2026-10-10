import numpy as np
from tinyllm.post.grpo import group_advantages, grpo_loss


def test_hand_grpo_objective():
    # Ratios are [1, 1]; advantages [-1, 1], so the mean clipped objective is 0.
    assert abs(grpo_loss([0, 0], [0, 0], [-1, 1], [0, 0], clip=0.2)) < 1e-12


def test_zero_advantage_has_zero_policy_gradient():
    eps = 1e-6
    fn = lambda x: grpo_loss([x, x], [0, 0], [0, 0], [0, 0])
    assert abs((fn(eps) - fn(-eps)) / (2 * eps)) < 1e-12


def test_grouped_rewards_are_centered():
    a = group_advantages(np.array([0.0, 1.0, 1.0]))
    assert abs(float(a.mean())) < 1e-12
    assert a[0] < 0 < a[1]
