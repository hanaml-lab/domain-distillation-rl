"""The GATE per-prompt mixing weight and combined advantage (Eqs. 3-4 of the paper)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateConfig:
    beta: float = 8.0     # temperature of the sigmoid
    gamma: float = 1.0    # weight of the verifier-reward standard deviation
    tau: float = 0.15     # teacher-competence threshold on p_T
    alpha: float = 0.5    # scale of the teacher advantage
    clip: float = 5.0     # per-token log-ratio clip
    prefix_cap: int | None = 1024
    skip_below: float = 0.05  # skip teacher scoring when lambda is below this


def gate_lambda(p_student: float | np.ndarray, p_teacher: float | np.ndarray, cfg: GateConfig = GateConfig()):
    """lambda(x) = 1{p_T >= tau} * sigmoid(beta * ([p_T - p_S]_+ - gamma * sqrt(p_S (1 - p_S))))."""
    ps = np.clip(np.asarray(p_student, float), 0.0, 1.0)
    pt = np.clip(np.asarray(p_teacher, float), 0.0, 1.0)
    gap = np.clip(pt - ps, 0.0, 1.0)
    z = cfg.beta * (gap - cfg.gamma * np.sqrt(ps * (1.0 - ps)))
    lam = 1.0 / (1.0 + np.exp(-z))
    lam = np.where(pt >= cfg.tau, lam, 0.0)
    return float(lam) if lam.ndim == 0 else lam


def combined_advantages(a_rl: np.ndarray, a_opd: np.ndarray, lam: float, cfg: GateConfig = GateConfig()) -> np.ndarray:
    """A_hat[i, t] = (1 - lam) * A_rl[i] + lam * alpha * a_opd[i, t].  a_rl: (G,), a_opd: (G, T)."""
    a_rl = np.asarray(a_rl, float)[:, None]
    return (1.0 - lam) * a_rl + lam * cfg.alpha * np.asarray(a_opd, float)


def regime(p_student: float, p_teacher: float, cfg: GateConfig = GateConfig()) -> str:
    """Name the routing regime of a prompt: teachable, explorable, saturated or unlearnable."""
    if p_teacher < cfg.tau and p_student <= 0.0:
        return "unlearnable"
    if p_student >= 1.0:
        return "saturated"
    if p_student <= 0.0:
        return "teachable"
    return "explorable"
