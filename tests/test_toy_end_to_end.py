from gate_distill.toy import ToyConfig, ToyTask
from gate_distill.trainer import TrainConfig, train


def test_gate_beats_grpo_on_teachable_prompts():
    task = ToyTask(ToyConfig(n_prompts=32, seed=1))
    g = train(task, TrainConfig(method="grpo", steps=150, seed=0))
    a = train(task, TrainConfig(method="gate", steps=150, seed=0))
    assert a["acc"][-1] > a["acc"][0]
    # GRPO has no signal where the student never samples the target (Proposition 1)
    assert g["by_kind"]["teachable"] < 0.05
    assert a["by_kind"]["teachable"] > g["by_kind"]["teachable"] + 0.1
    assert a["final_acc"] > g["final_acc"]


def test_gate_skips_teacher_calls():
    task = ToyTask(ToyConfig(n_prompts=32, seed=1))
    o = train(task, TrainConfig(method="opd", steps=60, seed=0))
    a = train(task, TrainConfig(method="gate", steps=60, seed=0))
    assert a["teacher_calls"] < o["teacher_calls"]


def test_opd_is_harmed_by_wrong_teacher():
    task = ToyTask(ToyConfig(n_prompts=32, seed=1))
    o = train(task, TrainConfig(method="opd", steps=150, seed=0))
    g = train(task, TrainConfig(method="grpo", steps=150, seed=0))
    assert o["by_kind"]["misled"] < g["by_kind"]["misled"]  # Proposition 2
