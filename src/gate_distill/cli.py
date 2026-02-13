"""Command line: run GATE and its baselines on the toy task and print a results table."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .mixing import GateConfig
from .toy import ToyConfig, ToyTask
from .trainer import TrainConfig, train


def run(config: dict, methods=None, seeds=None, steps=None) -> dict:
    task_cfg = dict(config["task"])
    task_cfg["frac"] = tuple(task_cfg["frac"])
    task = ToyTask(ToyConfig(**task_cfg))
    gate = GateConfig(**config["gate"])
    out = {}
    for m in methods or config["methods"]:
        accs, kinds = [], []
        for s in seeds if seeds is not None else config["seeds"]:
            tc = TrainConfig(method=m, seed=s, gate=gate, **{**config["train"], **({"steps": steps} if steps else {})})
            h = train(task, tc)
            accs.append(h["final_acc"])
            kinds.append(h["by_kind"])
        out[m] = {"mean": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                  "init": float(task.solve_rate(task.student0).mean()),
                  "by_kind": {k: float(np.mean([d[k] for d in kinds])) for k in kinds[0]}}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "toy.json"))
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    t0 = time.time()
    res = run(cfg, a.methods, a.seeds, a.steps)
    if a.json:
        print(json.dumps(res, indent=1))
        return
    kinds = list(next(iter(res.values()))["by_kind"])
    print(f"{'method':<8}{'solve rate':>16}" + "".join(f"{k:>12}" for k in kinds))
    for m, r in res.items():
        print(f"{m:<8}{r['mean']:>10.3f} ± {r['std']:.3f}" + "".join(f"{r['by_kind'][k]:>12.3f}" for k in kinds))
    print(f"initial student solve rate {next(iter(res.values()))['init']:.3f}; {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
