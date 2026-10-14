# GATE: gap-adaptive mixing of on-policy distillation and verifiable rewards

[![ci](https://github.com/hanaml-lab/domain-distillation-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/domain-distillation-rl/actions/workflows/ci.yml)

Reference implementation of *GATE: Gap-Adaptive Mixing of On-Policy Distillation and Verifiable Rewards for Small Domain Specialists* (Agarwal, hanaML technical report, 2026).

GATE post-trains a small student (1–8B) against two signals: a frontier teacher's token log-probabilities (on-policy
distillation, OPD) and a programmatic verifier (RLVR). It picks the mix **per prompt**, using the student's online
solve rate `p_S(x)` and the teacher's cached solve rate `p_T(x)`:

```
lambda(x) = 1{p_T >= tau} * sigmoid(beta * ([p_T - p_S]_+ - gamma * sqrt(p_S (1 - p_S))))
A[i,t]    = (1 - lambda) * A_grpo[i] + lambda * alpha * clip(log pi_T(y_t) - log pi_theta(y_t))
```

Prompts that neither signal can help with (`p_T < tau` and `p_S = 0`) are dropped. Saturated prompts are replayed at 5%.

## Install

```bash
pip install -e ".[dev]"          # numpy, pyyaml; pytest and ruff
pip install -e ".[figures]"      # + matplotlib, to regenerate the report's figures
```

## Quickstart (CPU, under a minute)

```bash
python -m gate_distill.demo      # or gate-demo (gate-toy also works); settings in configs/toy.yaml
pytest -q
```

## Toy run (real output)

The toy task is tabular, so solve rates and policy gradients are exact, and the report's four routing regimes coexist in it: easy prompts the student already solves, teachable prompts where the student never samples the target but the teacher solves them, misled prompts where the teacher is confidently wrong, and impossible prompts. GRPO, OPD, KDRL (a fixed mix of 0.5) and GATE train the same initial student (solve rate 0.084) for 300 steps on 3 seeds. Output of `python -m gate_distill.demo` on a CPU (Python 3.12), mean exact-match solve rate over 64 prompts, then by prompt type:

```
method        solve rate        easy   teachable      misled  impossible
grpo         0.550 ± 0.000       0.979       0.001       0.977       0.000
opd          0.272 ± 0.005       0.686       0.028       0.029       0.001
kdrl         0.472 ± 0.005       0.940       0.005       0.628       0.001
gate         0.656 ± 0.010       0.951       0.434       0.799       0.000
initial student solve rate 0.084; 31.7s
```

GRPO gets no signal on teachable prompts, and OPD copies the teacher's mistakes on misled prompts. GATE routes each prompt to the signal that helps it and solves the most prompts, though on misled prompts GRPO alone still does better. `paper/toy_run.md` records the environment and what each line shows.

## Planning a run on your own prompts

`examples/plan_mixing.py` reads, for each training prompt, the teacher's and the initial student's verified solves out of their samples, and prints the solve rates, the mixing weight `lambda(x)`, each prompt's regime, and what a GATE run would drop, replay and send to the teacher.

```bash
python examples/plan_mixing.py examples/data/solve_counts.csv
```

```
prompt       p_T   p_S  lambda  regime
finqa-001   1.00  1.00    0.50  saturated
finqa-002   1.00  0.88    0.16  explorable
finqa-003   0.75  0.62    0.05  explorable
finqa-004   1.00  0.25    0.93  explorable
finqa-005   1.00  0.00    1.00  teachable
finqa-006   0.75  0.00    1.00  teachable
finqa-007   0.25  0.12    0.16  explorable
finqa-008   0.00  0.38    0.00  explorable
finqa-009   0.00  0.00    0.00  unlearnable
finqa-010   0.50  0.50    0.02  explorable
finqa-011   1.00  0.12    0.99  explorable
finqa-012   0.25  0.00    0.88  teachable

3 teachable, 7 explorable, 1 saturated, 1 unlearnable
dropped: 1; replayed at 5%: 1; teacher scores 8 of the 10 prompts trained on (lambda >= 0.05)
```

For large-scale training, `gate_distill.mixing.combined_advantages` replaces the advantage computation of a GRPO trainer
(we used verl). The teacher is served separately and only prefills student rollouts, capped at `prefix_cap` tokens. It is
skipped when `lambda < skip_below`. `configs/qwen3-8b-finqa.yaml` lists the settings. This repository does not include the
cluster launcher.

## Results (from the report)

These numbers come from the report's seeded simulation and are illustrative. Qwen3-8B students, Qwen3-235B-A22B teacher, 3 seeds:

| Method | MedQA | FinQA | LegalBench-R | BFCL-AST | % of teacher |
| --- | --- | --- | --- | --- | --- |
| Prompted | 73.6 | 60.8 | 70.4 | 80.3 | 86.0 |
| OPD | 83.2 | 65.2 | 78.3 | 84.2 | 93.8 |
| GRPO | 83.5 | 64.1 | 77.1 | 85.6 | 93.5 |
| OPD → GRPO | 85.4 | 68.2 | 81.0 | 87.9 | 97.3 |
| KDRL | 85.3 | 67.1 | 80.1 | 87.8 | 96.6 |
| **GATE** | **87.1** | **68.6** | **81.8** | **89.6** | **98.7** |
| Teacher | 88.4 | 71.9 | 82.6 | 88.1 | 100 |

`bash scripts/reproduce_paper.sh` regenerates every table and figure. On every push, CI reruns `paper/make_data.py` and checks its tables and results against `paper/results.sha256` (`python paper/check_results.py`), so the report's numbers are exactly what the script emits.

## Layout

```
src/gate_distill/
  advantages.py   GRPO group advantages, prefix-capped reverse-KL teacher advantages
  mixing.py       GateConfig, gate_lambda, combined_advantages, regime
  estimators.py   SolveRateTable (EMA of p_S), coverage_rho (rho_k), pass_at_k
  sampler.py      GapAwareSampler (drop unlearnable, replay saturated)
  toy.py          tabular toy task where the four regimes coexist
  trainer.py      Algorithm 1 end to end on the toy task (gate, grpo, opd, kdrl)
  demo.py         the toy run (python -m gate_distill.demo); cli.py keeps the gate-toy name
configs/          toy.yaml; qwen3-8b-finqa.yaml records the report's 8B settings
examples/         plan_mixing.py, data/solve_counts.csv
scripts/          run_toy.py, reproduce_paper.sh
paper/            make_data.py, figures.py, check_results.py, results.sha256, toy_run.md
tests/            pytest suite, including an end-to-end toy run
```

## Citation

```bibtex
@techreport{agarwal2026gate,
  title       = {{GATE}: Gap-Adaptive Mixing of On-Policy Distillation and Verifiable Rewards for Small Domain Specialists},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026}
}
```

## License

Apache-2.0. See `LICENSE`.
