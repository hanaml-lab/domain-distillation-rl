"""End-to-end training loop (Algorithm 1) on the toy task, for GATE and its baselines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .advantages import grpo_advantages, opd_token_advantages
from .estimators import SolveRateTable
from .mixing import GateConfig, combined_advantages, gate_lambda
from .sampler import GapAwareSampler
from .toy import ToyTask

METHODS = ("grpo", "opd", "kdrl", "gate")


@dataclass
class TrainConfig:
    method: str = "gate"
    steps: int = 300
    batch: int = 16
    group: int = 8
    lr: float = 1.0
    k_teacher: int = 4
    k_student: int = 8
    kdrl_lambda: float = 0.5
    seed: int = 0
    gate: GateConfig = field(default_factory=lambda: GateConfig(prefix_cap=None))
    log_every: int = 25


def train(task: ToyTask, cfg: TrainConfig) -> dict:
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}; choose from {METHODS}")
    rng = np.random.default_rng(cfg.seed)
    theta = task.student0.copy()
    N, L, V = theta.shape
    keys = list(range(N))

    # offline teacher solve rates from K_T samples; initial student rates from K_S samples
    p_teacher = {x: float(task.reward(x, task.sample(task.teacher[x], cfg.k_teacher, rng)).mean()) for x in keys}
    ps = SolveRateTable(rate=0.5)
    for x in keys:
        ps.init(x, task.reward(x, task.sample(theta[x], cfg.k_student, rng)))
    sampler = GapAwareSampler(keys, p_teacher, ps, cfg.gate, seed=cfg.seed) if cfg.method == "gate" else None

    history = {"step": [], "acc": [], "teacher_calls": 0, "lambda_mean": []}
    for step in range(cfg.steps + 1):
        if step % cfg.log_every == 0 or step == cfg.steps:
            history["step"].append(step)
            history["acc"].append(float(task.solve_rate(theta).mean()))
        if step == cfg.steps:
            break
        batch = sampler.sample(cfg.batch) if sampler else list(rng.choice(N, cfg.batch))
        grad = np.zeros_like(theta)
        lams = []
        for x in batch:
            ys = task.sample(theta[x], cfg.group, rng)
            r = task.reward(x, ys)
            a_rl = grpo_advantages(r)
            if cfg.method == "grpo":
                lam = 0.0
            elif cfg.method == "opd":
                lam = 1.0
            elif cfg.method == "kdrl":
                lam = cfg.kdrl_lambda
            else:
                p_s = ps.update(x, r)
                lam = gate_lambda(p_s, p_teacher[x], cfg.gate)
            lams.append(lam)
            if lam >= cfg.gate.skip_below:
                lp_s = task.token_logp(theta[x], ys)
                lp_t = task.token_logp(task.teacher[x], ys)
                a_opd = opd_token_advantages(lp_t, lp_s, cfg.gate.clip, cfg.gate.prefix_cap)
                history["teacher_calls"] += 1
            else:
                a_opd = np.zeros((cfg.group, L))
            if cfg.method == "opd":
                adv = cfg.gate.alpha * a_opd
            else:
                adv = combined_advantages(a_rl, a_opd, lam, cfg.gate)  # (G, L)
            # grad of sum_t log pi(y_t) wrt theta[x, t, :] = onehot(y_t) - softmax
            p = np.exp(task.log_softmax(theta[x]))  # (L, V)
            onehot = np.eye(V)[ys]  # (G, L, V)
            grad[x] += (adv[..., None] * (onehot - p[None])).sum(0) / cfg.group
        theta += cfg.lr * grad  # per-prompt step of size lr
        history["lambda_mean"].append(float(np.mean(lams)))
    history["final_acc"] = history["acc"][-1]
    history["p_teacher"] = p_teacher
    history["by_kind"] = {k: float(task.solve_rate(theta)[task.kind == i].mean()) for i, k in enumerate(task.TYPES)}
    return history
