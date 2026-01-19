"""GATE: gap-adaptive mixing of on-policy distillation and verifiable rewards."""
from .advantages import grpo_advantages, opd_token_advantages
from .estimators import SolveRateTable, coverage_rho, pass_at_k
from .mixing import GateConfig, combined_advantages, gate_lambda, regime
from .sampler import GapAwareSampler

__all__ = [
    "GateConfig", "GapAwareSampler", "SolveRateTable", "combined_advantages", "coverage_rho",
    "gate_lambda", "grpo_advantages", "opd_token_advantages", "pass_at_k", "regime",
]
__version__ = "0.1.0"
