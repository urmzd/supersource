import numpy as np
from tinyllm.post.distill import forward_kl, reverse_kl


def test_hand_forward_kl():
    # WHY: the hand-computed asymmetric distributions distinguish teacher-to-student KL direction.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L12.4 section 3, Worked example by hand
    student = np.log([[0.5, 0.5]])
    teacher = np.log([[0.8, 0.2]])
    assert abs(forward_kl(student, teacher) - 0.1927447570217575) < 1e-12


def test_reverse_kl_modes():
    # WHY: reversing KL weights changes the answer on an asymmetric teacher and student pair.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: L12.4 section 4, The interface
    student = np.log([[0.5, 0.5]])
    teacher = np.log([[0.8, 0.2]])
    assert abs(reverse_kl(student, teacher) - 0.22314355131420976) < 1e-12
    assert abs(forward_kl(teacher, teacher)) < 1e-12


def test_on_policy_sampling():
    # WHY: stable normalization must handle extreme logits without producing NaN or infinity.
    # KIND: boundary
    # CHAPTER: L12.4 section 4, The interface
    a = forward_kl([[1000, -1000]], [[1000, -1000]])
    assert np.isfinite(a) and abs(a) < 1e-12
