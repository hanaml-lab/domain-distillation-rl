"""Verifier (GRPO) and teacher (reverse-KL OPD) advantage estimators."""
from __future__ import annotations

import numpy as np


def grpo_advantages(rewards: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Group-relative advantages A_i = (r_i - mean) / (std + eps) for one group of G rollouts.

    Returns zeros when every reward in the group is equal (Proposition 1 of the paper).
    """
    r = np.asarray(rewards, dtype=float)
    sd = r.std()
    if sd < 1e-12:
        return np.zeros_like(r)
    return (r - r.mean()) / (sd + eps)


def opd_token_advantages(
    logp_teacher: np.ndarray,
    logp_student: np.ndarray,
    clip: float = 5.0,
    prefix_cap: int | None = None,
) -> np.ndarray:
    """Per-token teacher advantage a_t = log pi_T(y_t) - log pi_theta(y_t), clipped, zeroed after the prefix cap.

    Shapes: (G, T). Treated as a constant (stop-gradient) by the caller.
    """
    a = np.clip(np.asarray(logp_teacher, float) - np.asarray(logp_student, float), -clip, clip)
    if prefix_cap is not None and prefix_cap < a.shape[-1]:
        a = a.copy()
        a[..., prefix_cap:] = 0.0
    return a


def group_never_informative_prob(p: float, group_size: int) -> float:
    """q_G(p) = p^G + (1-p)^G, the probability that a GRPO group has zero gradient."""
    return p**group_size + (1 - p) ** group_size
