from pathlib import Path

import yaml

from gate_distill.demo import run

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_from_the_yaml_config():
    cfg = yaml.safe_load((ROOT / "configs" / "toy.yaml").read_text())
    out = run(cfg, methods=["grpo", "gate"], seeds=[0], steps=60)
    assert set(out) == {"grpo", "gate"}
    assert out["gate"]["by_kind"]["teachable"] > out["grpo"]["by_kind"]["teachable"]
