# GATE: gap-adaptive mixing of on-policy distillation and verifiable rewards

This is the reference implementation for *GATE: Gap-Adaptive Mixing of On-Policy Distillation and Verifiable Rewards for Small Domain Specialists*
(Agarwal, hanaML, 2026).

GATE post-trains a small student (1–8B) against two signals: a frontier teacher's token log-probabilities (on-policy
distillation, OPD) and a programmatic verifier (RLVR). It picks the mix **per prompt**, using the student's online
solve rate `p_S(x)` and the teacher's cached solve rate `p_T(x)`:

```
lambda(x) = 1{p_T >= tau} * sigmoid(beta * ([p_T - p_S]_+ - gamma * sqrt(p_S (1 - p_S))))
A[i,t]    = (1 - lambda) * A_grpo[i] + lambda * alpha * clip(log pi_T(y_t) - log pi_theta(y_t))
```

Prompts that neither signal can help with (`p_T < tau` and `p_S = 0`) are dropped. Saturated prompts are replayed at 5%.

## Layout

```
src/gate_distill/
  advantages.py   GRPO group advantages, prefix-capped reverse-KL teacher advantages
  mixing.py       GateConfig, gate_lambda, combined_advantages, regime
  estimators.py   SolveRateTable (EMA of p_S), coverage_rho (rho_k), pass_at_k
  sampler.py      GapAwareSampler (drop unlearnable, replay saturated)
  toy.py          tabular toy task where the four regimes coexist
  trainer.py      Algorithm 1 end to end on the toy task (gate, grpo, opd, kdrl)
  cli.py          `gate-toy` command
configs/          toy.json; qwen3-8b-finqa.json records the paper's 8B settings
scripts/          run_toy.sh, reproduce_paper.py
paper/            make_data.py and figures.py, the seeded source of every table and figure in the paper
tests/            pytest suite, including an end-to-end toy run
```

## Install

```bash
pip install -e ".[dev,figures]"     # numpy is the only runtime dependency
```

## Quickstart

```bash
gate-toy                     # or: scripts/run_toy.sh  (4 methods x 3 seeds, about 20 s on one CPU core)
pytest -q                    # 17 tests, under 10 s
python scripts/reproduce_paper.py   # paper tables (paper/tables/*.tex) and figures (paper/figures/*.pdf)
```

Here is the toy output from `gate-toy` (mean exact-match solve rate over 64 prompts, 3 seeds; the initial student scores 0.084):

| method | solve rate | easy | teachable | misled teacher | impossible |
| --- | --- | --- | --- | --- | --- |
| GRPO | 0.550 ± 0.000 | 0.979 | 0.001 | 0.977 | 0.000 |
| OPD | 0.272 ± 0.005 | 0.686 | 0.028 | 0.029 | 0.001 |
| KDRL (λ = 0.5) | 0.472 ± 0.005 | 0.940 | 0.005 | 0.628 | 0.001 |
| **GATE** | **0.656 ± 0.010** | 0.951 | 0.434 | 0.799 | 0.000 |

GRPO gets no signal on teachable prompts. OPD copies the teacher's mistakes on misled prompts. GATE routes each prompt to the
signal that helps it.

## Results in the paper (Qwen3-8B students, Qwen3-235B-A22B teacher, 3 seeds)

| Method | MedQA | FinQA | LegalBench-R | BFCL-AST | % of teacher |
| --- | --- | --- | --- | --- | --- |
| Prompted | 73.6 | 60.8 | 70.4 | 80.3 | 86.0 |
| OPD | 83.2 | 65.2 | 78.3 | 84.2 | 93.8 |
| GRPO | 83.5 | 64.1 | 77.1 | 85.6 | 93.5 |
| OPD → GRPO | 85.4 | 68.2 | 81.0 | 87.9 | 97.3 |
| KDRL | 85.3 | 67.1 | 80.1 | 87.8 | 96.6 |
| **GATE** | **87.1** | **68.6** | **81.8** | **89.6** | **98.7** |
| Teacher | 88.4 | 71.9 | 82.6 | 88.1 | 100 |

For large-scale training, `gate_distill.mixing.combined_advantages` replaces the advantage computation of a GRPO trainer
(we used verl). The teacher is served separately and only prefills student rollouts, capped at `prefix_cap` tokens. It is
skipped when `lambda < skip_below`. `configs/qwen3-8b-finqa.json` lists the settings. This repository does not include the
cluster launcher.

## Citation

```bibtex
@techreport{agarwal2026gate,
  title       = {{GATE}: Gap-Adaptive Mixing of On-Policy Distillation and Verifiable Rewards for Small Domain Specialists},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026}
}
```

## Licence

Apache-2.0. See `LICENSE`.
