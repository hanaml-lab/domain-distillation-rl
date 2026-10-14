"""Plan a GATE run on your own prompts before training.

    python examples/plan_mixing.py examples/data/solve_counts.csv

The CSV has one row per training prompt: ``prompt_id``, then the teacher's and the student's verified solves out of
their samples (``teacher_solved``, ``teacher_samples``, ``student_solved``, ``student_samples``), for example from
k = 4 teacher samples and k = 8 samples of the initial student. For each prompt the script prints the solve rates,
GATE's mixing weight lambda(x) between the teacher's token advantages and the verifiable reward, and the prompt's
regime. Prompts that neither signal can help (the teacher is below tau and the student never solves them) are
dropped; prompts the student already solves are replayed at a low rate; teacher scoring is skipped when lambda is
below skip_below, which is where a GATE run saves teacher calls.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from gate_distill import GateConfig, gate_lambda, regime


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--tau", type=float, default=GateConfig.tau, help="teacher-competence threshold on p_T")
    ap.add_argument("--beta", type=float, default=GateConfig.beta)
    ap.add_argument("--gamma", type=float, default=GateConfig.gamma)
    args = ap.parse_args(argv)
    cfg = GateConfig(beta=args.beta, gamma=args.gamma, tau=args.tau)
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    p_t = np.array([int(r["teacher_solved"]) / int(r["teacher_samples"]) for r in rows])
    p_s = np.array([int(r["student_solved"]) / int(r["student_samples"]) for r in rows])
    lam = gate_lambda(p_s, p_t, cfg)
    regimes = [regime(s, t, cfg) for s, t in zip(p_s, p_t)]
    print(f"{'prompt':<10}{'p_T':>6}{'p_S':>6}{'lambda':>8}  regime")
    for r, t, s, w, g in zip(rows, p_t, p_s, lam, regimes):
        print(f"{r['prompt_id']:<10}{t:>6.2f}{s:>6.2f}{w:>8.2f}  {g}")
    counts = {g: regimes.count(g) for g in ("teachable", "explorable", "saturated", "unlearnable")}
    trained = [i for i, g in enumerate(regimes) if g in ("teachable", "explorable")]
    scored = sum(lam[i] >= cfg.skip_below for i in trained)
    print("\n" + ", ".join(f"{n} {g}" for g, n in counts.items()))
    print(f"dropped: {counts['unlearnable']}; replayed at 5%: {counts['saturated']}; "
          f"teacher scores {scored} of the {len(trained)} prompts trained on (lambda >= {cfg.skip_below})")
    return {"lambda": lam, "regimes": regimes, "teacher_scored": int(scored)}


if __name__ == "__main__":
    main()
