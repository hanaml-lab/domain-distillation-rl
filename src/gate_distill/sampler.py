"""Gap-aware prompt sampler: drops unlearnable prompts, replays saturated ones at a low rate."""
from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from .estimators import SolveRateTable
from .mixing import GateConfig


class GapAwareSampler:
    def __init__(self, keys: Sequence[Hashable], p_teacher: dict, p_student: SolveRateTable,
                 cfg: GateConfig = GateConfig(), replay: float = 0.05, seed: int = 0):
        self.keys = list(keys)
        self.p_teacher = p_teacher
        self.p_student = p_student
        self.cfg = cfg
        self.replay = replay
        self.rng = np.random.default_rng(seed)

    def pools(self):
        active, saturated, dropped = [], [], []
        for k in self.keys:
            pt, ps = self.p_teacher[k], self.p_student.get(k)
            if pt < self.cfg.tau and ps <= 0.0:
                dropped.append(k)
            elif self.p_student.saturated(k):
                saturated.append(k)
            else:
                active.append(k)
        return active, saturated, dropped

    def sample(self, batch_size: int) -> list:
        active, saturated, _ = self.pools()
        n_rep = int(round(self.replay * batch_size)) if saturated else 0
        n_act = batch_size - n_rep if active else 0
        out = []
        if n_act:
            out += [active[i] for i in self.rng.choice(len(active), n_act, replace=len(active) < n_act)]
        if n_rep:
            out += [saturated[i] for i in self.rng.choice(len(saturated), n_rep, replace=len(saturated) < n_rep)]
        return out
