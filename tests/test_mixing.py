import numpy as np

from gate_distill import GateConfig, combined_advantages, gate_lambda, regime


def test_lambda_teachable_high_and_unlearnable_zero():
    assert gate_lambda(0.0, 0.9) > 0.99
    assert gate_lambda(0.0, 0.1) == 0.0  # teacher below tau


def test_lambda_small_when_student_competitive():
    assert gate_lambda(0.5, 0.5) < 0.05
    assert gate_lambda(0.9, 0.95) < gate_lambda(0.2, 0.95)


def test_lambda_vectorised_and_bounded():
    ps, pt = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    lam = gate_lambda(ps, pt)
    assert lam.shape == (11, 11) and lam.min() >= 0 and lam.max() <= 1


def test_combined_advantage_limits():
    a_rl = np.array([1.0, -1.0])
    a_opd = np.array([[0.5, 0.5], [-0.5, 0.0]])
    cfg = GateConfig(alpha=1.0)
    assert np.allclose(combined_advantages(a_rl, a_opd, 0.0, cfg), [[1, 1], [-1, -1]])
    assert np.allclose(combined_advantages(a_rl, a_opd, 1.0, cfg), a_opd)


def test_regimes():
    assert regime(0.0, 0.0) == "unlearnable"
    assert regime(0.0, 0.75) == "teachable"
    assert regime(0.4, 0.75) == "explorable"
    assert regime(1.0, 1.0) == "saturated"
