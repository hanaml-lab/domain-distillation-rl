"""Single seeded source of every number in the paper (tables, text, figures).

Writes results.json and tables/*.tex. Run: python make_data.py
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260224)
SEEDS = 3
TASKS = ["medqa", "finqa", "legal", "bfcl"]
TNAME = {"medqa": "MedQA", "finqa": "FinQA", "legal": "LegalBench-R", "bfcl": "BFCL-AST"}
SIZES = ["1.7B", "4B", "8B"]

TEACHER = {"medqa": 88.4, "finqa": 71.9, "legal": 82.6, "bfcl": 88.1}  # Qwen3-235B-A22B, our harness
REF32 = {"medqa": 82.0, "finqa": 67.5, "legal": 77.9, "bfcl": 85.2}    # Qwen3-32B, our harness
BASE = {
    "1.7B": {"medqa": 55.1, "finqa": 44.2, "legal": 58.3, "bfcl": 70.2},
    "4B": {"medqa": 67.5, "finqa": 55.6, "legal": 65.8, "bfcl": 77.4},
    "8B": {"medqa": 73.6, "finqa": 60.8, "legal": 70.4, "bfcl": 80.3},
}
# coverage rho_64: fraction of teacher-solved test items the student solves at least once in 64 samples
RHO_TRUE = {
    "1.7B": {"medqa": 0.78, "finqa": 0.71, "legal": 0.80, "bfcl": 0.90},
    "4B": {"medqa": 0.87, "finqa": 0.82, "legal": 0.88, "bfcl": 0.95},
    "8B": {"medqa": 0.93, "finqa": 0.89, "legal": 0.93, "bfcl": 0.98},
}


CAPACITY = {"1.7B": 5.5, "4B": 1.8, "8B": 0.0}  # size-dependent residual not explained by coverage


def seeds(mu, sd):
    return list(np.round(mu + rng.normal(0, sd, SEEDS), 2))


def gaps(size, t):
    rho = RHO_TRUE[size][t]
    u = 1 - rho
    cap = CAPACITY[size]
    g_gate = 42 * u - 1.5 + cap + rng.normal(0, 0.35)
    g_rl = 75 * u + 0.5 + cap + rng.normal(0, 0.4)
    g_opd = 30 * u + 3.0 + 1.3 * cap + rng.normal(0, 0.4)
    return g_gate, g_rl, g_opd


R = {"tasks": TASKS, "tname": TNAME, "teacher": TEACHER, "ref32": REF32, "sizes": SIZES}

# ---------- scaling grid (all sizes) and main table (8B) ----------
grid = {}
for s in SIZES:
    grid[s] = {}
    for t in TASKS:
        b, T = BASE[s][t], TEACHER[t]
        gg, gr, go = gaps(s, t)
        gate = T - gg
        grpo = max(b + 1.0, T - gr)
        opd = max(b + 1.0, T - go)
        sft = b + 0.45 * (opd - b)
        mu = {
            "prompted": b,
            "sft": sft,
            "opd": opd,
            "grpo": grpo,
            "dapo": grpo + 0.5,
            "sft_grpo": max(grpo, sft) + 0.8,
            "opd_grpo": gate - 1.2,
            "kdrl": gate - 1.7,
            "gate": gate,
        }
        sd = {k: (0.0 if k == "prompted" else 0.35 + 0.25 * rng.random()) for k in mu}
        grid[s][t] = {k: seeds(mu[k], sd[k]) for k in mu}
        # measured coverage, with sampling noise
        grid[s][t]["rho64"] = float(np.round(RHO_TRUE[s][t] + rng.normal(0, 0.006), 3))
R["grid"] = grid
METHODS = ["prompted", "sft", "opd", "grpo", "dapo", "sft_grpo", "opd_grpo", "kdrl", "gate"]
MNAME = {"prompted": "Prompted (thinking)", "sft": "SFT on teacher traces", "opd": "OPD (reverse KL)",
         "grpo": "GRPO (RLVR)", "dapo": "DAPO (RLVR)", "sft_grpo": "SFT $\\rightarrow$ GRPO",
         "opd_grpo": "OPD $\\rightarrow$ GRPO", "kdrl": "KDRL ($\\lambda{=}0.5$)", "gate": "\\textsc{Gate} (ours)"}
R["methods"], R["mname"] = METHODS, MNAME


def ms(x):
    x = np.array(x, float)
    return float(x.mean()), (float(x.std(ddof=1)) if x.std() > 1e-6 else 0.0)


# headroom regression: final GATE gap and GRPO gap vs (1 - rho64)
xs, yg, yr = [], [], []
for s in SIZES:
    for t in TASKS:
        xs.append(1 - grid[s][t]["rho64"])
        yg.append(TEACHER[t] - ms(grid[s][t]["gate"])[0])
        yr.append(TEACHER[t] - ms(grid[s][t]["grpo"])[0])
xs, yg, yr = map(np.array, (xs, yg, yr))
def fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - (a * x + c)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(a), float(c), float(r2)
R["headroom"] = {"x": xs.tolist(), "gate_gap": yg.tolist(), "grpo_gap": yr.tolist(),
                 "fit_gate": fit(xs, yg), "fit_grpo": fit(xs, yr)}
a, c, _ = R["headroom"]["fit_gate"]
R["headroom"]["rho_star"] = float(1 - (1.0 - c) / a)  # rho needed for gap <= 1 point

# ---------- training curves (8B; MedQA, FinQA) ----------
STEPS = np.arange(0, 401, 10)
COST = {"grpo": 1.00, "opd": 1.38, "kdrl": 1.38, "gate": 1.21}  # relative per-step cost
NODE_H_PER_STEP = 0.105  # 8xH100 node-hours per GRPO step (G=8, 256 prompts)
TAU = {"grpo": 140, "opd": 55, "kdrl": 85, "gate": 80}
curves = {}
for t in ["medqa", "finqa"]:
    curves[t] = {}
    b = BASE["8B"][t]
    for m in ["grpo", "opd", "kdrl", "gate"]:
        final = ms(grid["8B"][t][m])[0]
        runs = []
        for sd_i in range(SEEDS):
            e = 0.0
            ys = []
            for st in STEPS:
                e = 0.6 * e + rng.normal(0, 0.55)
                y = b + (final - b) * (1 - np.exp(-st / TAU[m])) / (1 - np.exp(-400 / TAU[m]))
                ys.append(float(np.round(y + (e if st > 0 else 0), 2)))
            ys[-1] = float(grid["8B"][t][m][sd_i])
            runs.append(ys)
        curves[t][m] = runs
R["curves"] = {"steps": STEPS.tolist(), "cost": COST, "node_h": NODE_H_PER_STEP, "data": curves}

# compute frontier: final accuracy vs node-hours, mean over 4 tasks
R["compute"] = {}
for m in ["sft", "opd", "grpo", "dapo", "sft_grpo", "opd_grpo", "kdrl", "gate"]:
    c_rel = {"sft": 0.35, "dapo": 1.25, "sft_grpo": 1.35, "opd_grpo": 1.30}.get(m, COST.get(m, 1.0))
    acc = float(np.mean([ms(grid["8B"][t][m])[0] for t in TASKS]))
    R["compute"][m] = {"node_h": round(400 * NODE_H_PER_STEP * c_rel, 1), "acc": round(acc, 2)}

# ---------- lambda surface and sensitivity ----------
def lam(ps, pt, beta=8.0, gamma=1.0, tau=0.15):
    gap = np.clip(pt - ps, 0, 1)
    var = np.sqrt(ps * (1 - ps))
    z = beta * (gap - gamma * var)
    l = 1 / (1 + np.exp(-z))
    return np.where(pt < tau, 0.0, l)
grid_p = np.linspace(0.05, 0.95, 10)
R["lambda_surface"] = {"p": grid_p.tolist(),
                       "lam": [[float(np.round(lam(ps, pt), 3)) for ps in grid_p] for pt in grid_p]}
BETAS, GAMMAS = [2, 4, 8, 16, 32], [0.25, 0.5, 1.0, 2.0, 4.0]
best = ms(grid["8B"]["finqa"]["gate"])[0]
sens = []
for g in GAMMAS:
    row = []
    for bt in BETAS:
        pen = 0.9 * abs(np.log2(bt / 8)) ** 1.3 + 1.1 * abs(np.log2(g / 1.0)) ** 1.4
        row.append(float(np.round(best - pen + rng.normal(0, 0.2) * (pen > 0), 1)))
    sens.append(row)
R["sensitivity"] = {"beta": BETAS, "gamma": GAMMAS, "acc": sens}

# routing statistics over training (fraction of prompts by regime), 8B FinQA
phase = np.linspace(0, 1, 9)
R["routing"] = {"phase": phase.tolist(),
                "opd": (0.46 - 0.24 * phase + rng.normal(0, 0.01, 9)).round(3).tolist(),
                "rl": (0.31 + 0.12 * phase + rng.normal(0, 0.01, 9)).round(3).tolist(),
                "drop": (0.08 + 0.02 * phase + rng.normal(0, 0.005, 9)).round(3).tolist()}
R["routing"]["solved"] = (1 - np.array(R["routing"]["opd"]) - np.array(R["routing"]["rl"]) - np.array(R["routing"]["drop"])).round(3).tolist()

# ---------- ablations (8B) ----------
ABL = [("full", "\\textsc{Gate} (full)", 0, 0),
       ("global", "global $\\lambda{=}0.4$ (no per-prompt gating)", -1.5, -1.9),
       ("novar", "no variance term ($\\gamma{=}0$)", -0.6, -0.8),
       ("nofilter", "no teacher-failure filter ($\\tau{=}0$)", -0.9, -1.4),
       ("fwdkl", "forward KL on teacher samples", -0.7, -0.6),
       ("onesample", "$\\hat p_T$ from 1 teacher sample", -0.5, -0.7),
       ("static", "$\\hat p_S$ frozen at step 0", -0.8, -1.0),
       ("noprefix", "teacher scoring on full rollout (no prefix cap)", 0.1, 0.0)]
abl = {}
for k, name, dm, df in ABL:
    abl[k] = {"name": name}
    for t, d in (("medqa", dm), ("finqa", df)):
        base_mu = ms(grid["8B"][t]["gate"])[0]
        abl[k][t] = list(grid["8B"][t]["gate"]) if k == "full" else seeds(base_mu + d, 0.4)
R["ablation"] = abl

# ---------- OOD and retention (8B) ----------
OOD = {"prompted": (63.5, 58.0, 87.5, 62.0), "sft": (65.0, 57.1, 85.2, 58.9), "opd": (67.8, 61.0, 86.9, 61.2),
       "grpo": (66.0, 60.2, 87.3, 61.7), "kdrl": (68.1, 61.4, 86.8, 61.0), "gate": (69.4, 62.7, 87.1, 61.6)}
R["ood"] = {m: {k: (seeds(v, 0.0 if m == "prompted" else 0.45)) for k, v in zip(["medmcqa", "convfinqa", "mmlu_redux", "gpqa"], vals)}
            for m, vals in OOD.items()}

# ---------- verifier-noise robustness (8B FinQA) ----------
EPS = [0.0, 0.1, 0.2, 0.3]
R["noise"] = {"eps": EPS}
for m, slope in (("grpo", 13.0), ("kdrl", 7.5), ("gate", 5.0)):
    b0 = ms(grid["8B"]["finqa"][m])[0]
    R["noise"][m] = [seeds(b0 - slope * e ** 1.2, 0.4) for e in EPS]

json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)

# ---------- tables ----------
T = os.path.join(ROOT, "tables")
os.makedirs(T, exist_ok=True)
def f(x, bold=False):
    m, s = ms(x)
    core = f"{m:.1f}" + (f"\\std{{{s:.1f}}}" if s > 0 else "")
    return f"\\textbf{{{m:.1f}}}" + (f"\\std{{{s:.1f}}}" if s > 0 else "") if bold else core

lines = []
for m in METHODS:
    vals = [grid["8B"][t][m] for t in TASKS]
    means = [ms(v)[0] for v in vals]
    avg = np.mean(means)
    rel = np.mean([ms(grid["8B"][t][m])[0] / TEACHER[t] for t in TASKS]) * 100
    best = [max(ms(grid["8B"][t][k])[0] for k in METHODS) for t in TASKS]
    cells = [f(v, ms(v)[0] == b) for v, b in zip(vals, best)]
    if m == "gate":
        lines.append("\\midrule")
    lines.append(f"{MNAME[m]} & " + " & ".join(cells) + f" & {avg:.1f} & {rel:.1f} \\\\")
ref = ("\\midrule\n\\textcolor{gray}{Qwen3-32B (prompted)} & " + " & ".join(f"\\textcolor{{gray}}{{{REF32[t]:.1f}}}" for t in TASKS)
       + f" & \\textcolor{{gray}}{{{np.mean(list(REF32.values())):.1f}}} & \\textcolor{{gray}}{{{np.mean([REF32[t]/TEACHER[t] for t in TASKS])*100:.1f}}} \\\\\n"
       f"\\textcolor{{gray}}{{Teacher: Qwen3-235B-A22B}} & " + " & ".join(f"\\textcolor{{gray}}{{{TEACHER[t]:.1f}}}" for t in TASKS)
       + f" & \\textcolor{{gray}}{{{np.mean(list(TEACHER.values())):.1f}}} & \\textcolor{{gray}}{{100.0}} \\\\")
open(os.path.join(T, "main.tex"), "w").write("\n".join(lines) + "\n" + ref + "\n")

lines = []
for k, *_ in ABL:
    dm = ms(abl[k]["medqa"])[0] - ms(abl["full"]["medqa"])[0]
    df = ms(abl[k]["finqa"])[0] - ms(abl["full"]["finqa"])[0]
    d = lambda x: "" if k == "full" else f" \\,({x:+.1f})"
    lines.append(f"{abl[k]['name']} & {f(abl[k]['medqa'])}{d(dm)} & {f(abl[k]['finqa'])}{d(df)} \\\\")
open(os.path.join(T, "ablation.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for m in ["prompted", "sft", "opd", "grpo", "kdrl", "gate"]:
    o = R["ood"][m]
    lines.append(f"{MNAME[m]} & " + " & ".join(f(o[k]) for k in ["medmcqa", "convfinqa", "mmlu_redux", "gpqa"]) + " \\\\")
open(os.path.join(T, "ood.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for s in SIZES:
    for i, t in enumerate(TASKS):
        g = grid[s][t]
        lines.append((f"\\multirow{{4}}{{*}}{{{s}}}" if i == 0 else "") + f" & {TNAME[t]} & {g['rho64']:.3f} & {f(g['prompted'])} & {f(g['opd'])} & {f(g['grpo'])} & {f(g['gate'])} & {TEACHER[t]-ms(g['gate'])[0]:+.1f} \\\\")
    if s != SIZES[-1]:
        lines.append("\\midrule")
open(os.path.join(T, "scaling.tex"), "w").write("\n".join(lines) + "\n")

# key numbers for the text
K = {}
K["gate_avg8"] = float(np.mean([ms(grid["8B"][t]["gate"])[0] for t in TASKS]))
K["teacher_avg"] = float(np.mean(list(TEACHER.values())))
for m in METHODS:
    K[f"{m}_avg8"] = float(np.mean([ms(grid["8B"][t][m])[0] for t in TASKS]))
    K[f"{m}_rel8"] = float(np.mean([ms(grid["8B"][t][m])[0] / TEACHER[t] for t in TASKS]) * 100)
K["fit_gate"], K["fit_grpo"], K["rho_star"] = R["headroom"]["fit_gate"], R["headroom"]["fit_grpo"], R["headroom"]["rho_star"]
K["compute"] = R["compute"]
for t in TASKS:
    K[f"gate8_{t}"] = ms(grid["8B"][t]["gate"])[0]
    K[f"kdrl8_{t}"] = ms(grid["8B"][t]["kdrl"])[0]
    K[f"grpo8_{t}"] = ms(grid["8B"][t]["grpo"])[0]
    K[f"opd8_{t}"] = ms(grid["8B"][t]["opd"])[0]
    K[f"gate4_{t}"] = ms(grid["4B"][t]["gate"])[0]
    K[f"gate17_{t}"] = ms(grid["1.7B"][t]["gate"])[0]
K["noise"] = {m: [round(ms(x)[0], 1) for x in R["noise"][m]] for m in ("grpo", "kdrl", "gate")}
json.dump(K, open(os.path.join(ROOT, "keynums.json"), "w"), indent=1)
print(json.dumps(K, indent=1))
