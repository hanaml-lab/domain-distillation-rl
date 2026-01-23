import numpy as np

from gate_distill import grpo_advantages, opd_token_advantages
from gate_distill.advantages import group_never_informative_prob


def test_grpo_zero_when_group_uniform():
    assert np.all(grpo_advantages(np.ones(8)) == 0)
    assert np.all(grpo_advantages(np.zeros(8)) == 0)


def test_grpo_standardised():
    a = grpo_advantages(np.array([1, 0, 0, 1, 1, 0, 0, 0.0]))
    assert abs(a.mean()) < 1e-9
    assert abs((a**2).sum() - 8) < 1e-3  # Proposition 1: ||A||^2 = G for a mixed group


def test_opd_clip_and_prefix():
    lt = np.array([[0.0, -10.0, -1.0, -1.0]])
    ls = np.array([[-1.0, 0.0, -1.0, -3.0]])
    a = opd_token_advantages(lt, ls, clip=5.0, prefix_cap=3)
    assert np.allclose(a, [[1.0, -5.0, 0.0, 0.0]])


def test_never_informative_probability():
    assert group_never_informative_prob(0.0, 8) == 1.0
    assert abs(group_never_informative_prob(0.5, 2) - 0.5) < 1e-12


def test_monte_carlo_matches_q():
    rng = np.random.default_rng(0)
    p, G = 0.1, 8
    zero = np.mean([np.all(grpo_advantages((rng.random(G) < p).astype(float)) == 0) for _ in range(20000)])
    assert abs(zero - group_never_informative_prob(p, G)) < 0.01
