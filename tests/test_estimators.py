import numpy as np

from gate_distill import GapAwareSampler, SolveRateTable, coverage_rho, pass_at_k


def test_ema_and_saturation():
    t = SolveRateTable(rate=0.5)
    t.init("a", [0, 0, 0, 0])
    assert t.update("a", [1, 1, 1, 1]) == 0.5
    t.update("a", [1, 1]); assert t.saturated("a")
    t.update("a", [0, 1]); assert not t.saturated("a")


def test_coverage_rho():
    s = np.array([1, 0, 1, 0], bool); te = np.array([1, 1, 1, 0], bool)
    assert abs(coverage_rho(s, te) - 2 / 3) < 1e-12


def test_pass_at_k():
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-12


def test_sampler_drops_unlearnable_and_replays_saturated():
    ps = SolveRateTable()
    for k, v in {"u": 0.0, "t": 0.0, "e": 0.5}.items():
        ps.init(k, [v])
    ps.init("s", [1.0]); ps.update("s", [1, 1]); ps.update("s", [1, 1])
    pt = {"u": 0.0, "t": 0.75, "e": 0.75, "s": 1.0}
    smp = GapAwareSampler(["u", "t", "e", "s"], pt, ps, replay=0.25, seed=0)
    active, saturated, dropped = smp.pools()
    assert dropped == ["u"] and saturated == ["s"] and set(active) == {"t", "e"}
    b = smp.sample(8)
    assert "u" not in b and b.count("s") == 2
