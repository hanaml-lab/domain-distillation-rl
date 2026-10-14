import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_mixing_on_the_bundled_counts():
    out = _example("plan_mixing").main([str(ROOT / "examples" / "data" / "solve_counts.csv")])
    regimes, lam = out["regimes"], out["lambda"]
    # a prompt the student never solves but the teacher does is routed to the teacher
    assert regimes[4] == "teachable" and lam[4] > 0.99
    # a prompt neither solves is dropped, and the teacher gets no weight where it is below tau
    assert regimes[8] == "unlearnable" and lam[8] == 0.0 and lam[7] == 0.0
    assert regimes.count("saturated") == 1
