"""Solve-rate estimates: an online EMA table for the student and a cached table for the teacher."""
from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np


class SolveRateTable:
    """Exponential moving average of per-prompt solve rates, updated from each group's rewards."""

    def __init__(self, rate: float = 0.5, default: float = 0.0):
        self.rate = rate
        self.default = default
        self._p: dict[Hashable, float] = {}
        self._saturated_visits: dict[Hashable, int] = {}

    def init(self, key: Hashable, rewards: Iterable[float]) -> None:
        self._p[key] = float(np.mean(list(rewards)))

    def update(self, key: Hashable, rewards: Iterable[float]) -> float:
        r = float(np.mean(list(rewards)))
        old = self._p.get(key)
        new = r if old is None else (1 - self.rate) * old + self.rate * r
        self._p[key] = new
        self._saturated_visits[key] = self._saturated_visits.get(key, 0) + 1 if r >= 1.0 else 0
        return new

    def get(self, key: Hashable) -> float:
        return self._p.get(key, self.default)

    def saturated(self, key: Hashable, visits: int = 2) -> bool:
        return self._saturated_visits.get(key, 0) >= visits

    def __contains__(self, key: Hashable) -> bool:
        return key in self._p


def coverage_rho(student_pass: np.ndarray, teacher_pass: np.ndarray) -> float:
    """Teacher-conditional coverage rho_k.

    student_pass, teacher_pass: boolean arrays (n_prompts,) of pass@k > 0 for each model.
    Returns P[student pass@k > 0 | teacher pass@k > 0].
    """
    s = np.asarray(student_pass, bool)
    t = np.asarray(teacher_pass, bool)
    if t.sum() == 0:
        return float("nan")
    return float((s & t).sum() / t.sum())


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
