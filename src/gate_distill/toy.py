"""A tabular toy setting in which the four routing regimes of the paper coexist.

Each prompt x has a target sequence of length L over a vocabulary of size V; the verifier returns 1 iff the
sampled sequence equals the target. Policies are factorised softmax tables theta[x, t, v], so log-probabilities,
exact solve rates and policy gradients are available in closed form and everything runs on CPU in seconds.

Prompt types (fractions set in ToyConfig):
  easy        student already solves often; teacher solves.        -> explorable / saturated
  teachable   student puts ~0 mass on the target at one position;   -> teachable (GRPO has no signal)
              teacher solves.
  misled      teacher is confidently wrong at one position;         -> OPD harmful; GRPO helps
              student solves sometimes.
  impossible  neither model solves.                                 -> unlearnable
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyConfig:
    n_prompts: int = 64
    length: int = 4
    vocab: int = 6
    frac: tuple = (0.35, 0.35, 0.2, 0.1)  # easy, teachable, misled, impossible
    seed: int = 0


class ToyTask:
    TYPES = ("easy", "teachable", "misled", "impossible")

    def __init__(self, cfg: ToyConfig = ToyConfig()):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        N, L, V = cfg.n_prompts, cfg.length, cfg.vocab
        self.target = rng.integers(0, V, size=(N, L))
        counts = np.floor(np.array(cfg.frac) * N).astype(int)
        counts[0] += N - counts.sum()
        self.kind = np.repeat(np.arange(4), counts)
        rng.shuffle(self.kind)
        self.student0 = rng.normal(0, 0.3, size=(N, L, V))
        self.teacher = rng.normal(0, 0.3, size=(N, L, V))
        for x in range(N):
            k, tgt = self.kind[x], self.target[x]
            wrong = (tgt + 1 + rng.integers(0, V - 1, size=L)) % V
            pos = rng.integers(0, L)
            # teacher: confident on the target, except where noted
            self.teacher[x, np.arange(L), tgt] += 4.0
            # student: moderately on the target
            self.student0[x, np.arange(L), tgt] += 2.2
            if k == 1:  # teachable: student has ~no mass on the target at `pos`
                self.student0[x, pos, tgt[pos]] -= 7.0
            elif k == 2:  # misled: teacher confidently wrong at `pos`
                self.teacher[x, pos, wrong[pos]] += 7.0
            elif k == 3:  # impossible: both put ~no mass on the target at `pos`
                self.student0[x, pos, tgt[pos]] -= 7.0
                self.teacher[x, pos, tgt[pos]] -= 11.0

    @staticmethod
    def log_softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    def solve_rate(self, theta: np.ndarray, x: int | None = None) -> np.ndarray | float:
        lp = self.log_softmax(theta)
        idx = np.arange(self.cfg.length)
        if x is not None:
            return float(np.exp(lp[x, idx, self.target[x]].sum()))
        return np.exp(np.stack([lp[i, idx, self.target[i]].sum() for i in range(self.cfg.n_prompts)]))

    def sample(self, theta_x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        p = np.exp(self.log_softmax(theta_x))  # (L, V)
        cum = p.cumsum(-1)
        u = rng.random((n, self.cfg.length, 1))
        return (u > cum[None]).sum(-1).clip(max=self.cfg.vocab - 1)  # (n, L)

    def reward(self, x: int, ys: np.ndarray) -> np.ndarray:
        return (ys == self.target[x][None]).all(-1).astype(float)

    def token_logp(self, theta_x: np.ndarray, ys: np.ndarray) -> np.ndarray:
        lp = self.log_softmax(theta_x)
        return lp[np.arange(self.cfg.length)[None], ys]  # (n, L)
