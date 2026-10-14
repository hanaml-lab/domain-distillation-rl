# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that the method works end to end on a problem with a known answer; it does not reproduce the report's
experiments.

- Command: `python -m gate_distill.demo` (settings in `configs/toy.yaml`)
- Code: commit `3594605`
- Environment: Python 3.12.13, numpy 2.5.4, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 33 s

## The toy setting

A tabular task in which the report's four routing regimes coexist (`src/gate_distill/toy.py`). Each of 64 prompts
has a target sequence of 4 tokens over a vocabulary of 6, and the verifier returns 1 only for the exact target.
Policies are factorised softmax tables, so solve rates, log-probabilities and policy gradients are exact. 35% of
the prompts are easy (the student already solves them often), 35% teachable (the student puts almost no mass on
the target at one position, the teacher solves them), 20% misled (the teacher is confidently wrong at one position)
and 10% impossible (neither model solves them). Each method trains the same initial student for 300 steps
(batch 16, group 8) on 3 seeds: GRPO on the verifier reward alone, on-policy distillation (OPD) on the teacher's
token advantages alone, KDRL with a fixed mix of 0.5, and GATE with its per-prompt weight. The table gives the mean
exact-match solve rate over all prompts with its standard deviation over seeds, then the mean by prompt type.

## Output

```
method        solve rate        easy   teachable      misled  impossible
grpo         0.550 ± 0.000       0.979       0.001       0.977       0.000
opd          0.272 ± 0.005       0.686       0.028       0.029       0.001
kdrl         0.472 ± 0.005       0.940       0.005       0.628       0.001
gate         0.656 ± 0.010       0.951       0.434       0.799       0.000
initial student solve rate 0.084; 31.7s
```

## What it shows

- GATE solves the most prompts (0.656, against 0.550 for GRPO, 0.472 for KDRL and 0.272 for OPD), and the
  breakdown shows why.
- GRPO gets no signal on teachable prompts (0.001): the student never samples their target, so every reward in a
  group is zero. GATE routes those prompts to the teacher and solves 43% of them.
- OPD copies the teacher's mistakes on misled prompts (0.029). GATE's teacher-competence gate limits that damage
  (0.799), though GRPO alone does better there (0.977).
- A fixed mix (KDRL) inherits part of each failure: 0.005 on teachable prompts and 0.628 on misled ones.
- No method solves impossible prompts, and GATE drops them from sampling.
