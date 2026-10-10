import numpy as np
from tinyllm.post.grpo import group_advantages, grpo_loss


def test_hand_grpo_objective():
    # WHY: ratios outside the clip interval must use the clipped objective for both signs.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: L12.3 section 3, Worked example by hand
    got = grpo_loss([np.log(2), np.log(0.5)], [0, 0], [1, -1], [0, 0], clip=0.2)
    assert abs(got - (-0.2)) < 1e-12


def test_zero_advantage_has_zero_policy_gradient():
    # WHY: equal rewards produce zero advantages and therefore no policy update.
    # KIND: boundary
    # CHAPTER: L12.3 section 4, The interface
    eps = 1e-6
    fn = lambda x: grpo_loss([x, x], [0, 0], [0, 0], [0, 0])
    assert abs((fn(eps) - fn(-eps)) / (2 * eps)) < 1e-12


def test_grouped_rewards_are_centered():
    # WHY: normalization centers rewards within the prompt group before they scale updates.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L12.3 section 4, The interface
    a = group_advantages(np.array([0.0, 1.0, 1.0]))
    assert abs(float(a.mean())) < 1e-12
    assert a[0] < 0 < a[1]
