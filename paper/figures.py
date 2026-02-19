"""All figures, drawn from results.json written by make_data.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results.json")))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "axes.grid": True, "lines.linewidth": 1.2, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"grpo": "#0072B2", "opd": "#009E73", "kdrl": "#E69F00", "gate": "#D55E00", "sft": "#999999", "dapo": "#56B4E9",
     "sft_grpo": "#CC79A7", "opd_grpo": "#F0E442", "teacher": "#000000"}
N = {"grpo": "GRPO", "opd": "OPD", "kdrl": "KDRL", "gate": "GATE (ours)", "sft": "SFT", "dapo": "DAPO",
     "sft_grpo": "SFT→GRPO", "opd_grpo": "OPD→GRPO"}
TN = R["tname"]
ci = lambda a: 1.96 * a.std(0, ddof=1) / np.sqrt(a.shape[0])

# 1. training curves vs compute
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.05))
st = np.array(R["curves"]["steps"])
for ax, t in zip(axs, ["medqa", "finqa"]):
    for m in ["grpo", "opd", "kdrl", "gate"]:
        a = np.array(R["curves"]["data"][t][m])
        x = st * R["curves"]["cost"][m] * R["curves"]["node_h"]
        mu = a.mean(0)
        ax.plot(x, mu, color=C[m], lw=1.5 if m == "gate" else 1.0, label=N[m])
        ax.fill_between(x, mu - ci(a), mu + ci(a), color=C[m], alpha=0.2, lw=0)
    ax.axhline(R["teacher"][t], ls="--", color="k", lw=0.7, label="Teacher (235B)")
    ax.set_title(TN[t]); ax.set_xlabel("Training compute (8×H100 node-hours)")
axs[0].set_ylabel("Accuracy (%)"); axs[0].legend(loc="lower right", ncol=1)
fig.savefig(os.path.join(OUT, "fig_curves.pdf")); plt.close(fig)

# 2. headroom predictor + compute Pareto
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.15))
H = R["headroom"]; x = np.array(H["x"])
mk = {"1.7B": "o", "4B": "s", "8B": "^"}
for i, s in enumerate(R["sizes"]):
    sl = slice(4 * i, 4 * i + 4)
    axs[0].scatter(x[sl], np.array(H["grpo_gap"])[sl], marker=mk[s], s=14, color=C["grpo"], alpha=0.8, lw=0)
    axs[0].scatter(x[sl], np.array(H["gate_gap"])[sl], marker=mk[s], s=14, color=C["gate"], lw=0, label=f"Qwen3-{s}")
xx = np.linspace(0, x.max() * 1.05, 50)
for key, col, lab in (("fit_grpo", C["grpo"], "GRPO"), ("fit_gate", C["gate"], "GATE")):
    a, c, r2 = H[key]
    axs[0].plot(xx, a * xx + c, color=col, lw=0.9, label=f"{lab} fit, $R^2$={r2:.2f}")
axs[0].axhline(0, color="k", lw=0.5)
axs[0].set_xlabel(r"Coverage deficit $1-\rho_{64}$"); axs[0].set_ylabel("Gap to teacher (pts)")
axs[0].legend(loc="upper left", fontsize=5.8)
for m, v in R["compute"].items():
    axs[1].scatter(v["node_h"], v["acc"], s=18, color=C[m], zorder=3, edgecolor="k" if m == "gate" else "none", lw=0.5)
    axs[1].annotate(N[m], (v["node_h"], v["acc"]), xytext=(3, -3 if m != "opd_grpo" else 3), textcoords="offset points", fontsize=5.8)
axs[1].axhline(np.mean(list(R["teacher"].values())), ls="--", color="k", lw=0.7)
axs[1].text(15, np.mean(list(R["teacher"].values())) + 0.25, "Teacher", fontsize=6)
axs[1].set_xlabel("Training compute (node-hours)"); axs[1].set_ylabel("Mean accuracy, 4 tasks (%)")
fig.savefig(os.path.join(OUT, "fig_headroom.pdf")); plt.close(fig)

# 3. scaling across student sizes
fig, axs = plt.subplots(1, 4, figsize=(5.5, 1.75), sharex=True)
sz = [1.7, 4, 8]
for ax, t in zip(axs, R["tasks"]):
    for m in ["prompted", "opd", "grpo", "gate"]:
        a = np.array([R["grid"][s][t][m] for s in R["sizes"]])
        mu, e = a.mean(1), 1.96 * a.std(1, ddof=1) / np.sqrt(a.shape[1])
        col = C.get(m, "#777777")
        ax.errorbar(sz, mu, yerr=e, color=col, marker="o", ms=2.3, lw=1.4 if m == "gate" else 0.9, capsize=1.5,
                    ls=":" if m == "prompted" else "-", label={"prompted": "Prompted"}.get(m, N.get(m)))
    ax.axhline(R["teacher"][t], ls="--", color="k", lw=0.7)
    ax.set_xscale("log"); ax.set_xticks(sz); ax.set_xticklabels(["1.7", "4", "8"]); ax.minorticks_off()
    ax.set_title(TN[t]); ax.set_xlabel("Student (B)")
axs[0].set_ylabel("Accuracy (%)"); axs[0].legend(loc="lower right", fontsize=5.5)
fig.savefig(os.path.join(OUT, "fig_scaling.pdf")); plt.close(fig)

# 4. lambda surface and sensitivity
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.2))
L = np.array(R["lambda_surface"]["lam"]); p = R["lambda_surface"]["p"]
im = axs[0].imshow(L, origin="lower", cmap="cividis", extent=[0, 1, 0, 1], vmin=0, vmax=1, aspect="auto")
axs[0].set_xlabel(r"Student solve rate $\hat p_S(x)$"); axs[0].set_ylabel(r"Teacher solve rate $\hat p_T(x)$")
axs[0].set_title(r"Mixing weight $\lambda(x)$"); axs[0].grid(False)
fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
S = np.array(R["sensitivity"]["acc"])
im2 = axs[1].imshow(S, origin="lower", cmap="viridis", aspect="auto")
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        axs[1].text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color="w" if S[i, j] < S.max() - 2 else "k")
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(R["sensitivity"]["beta"]); axs[1].set_yticks(range(5)); axs[1].set_yticklabels(R["sensitivity"]["gamma"])
axs[1].set_xlabel(r"Temperature $\beta$"); axs[1].set_ylabel(r"Variance weight $\gamma$"); axs[1].set_title("FinQA accuracy (8B)"); axs[1].grid(False)
fig.savefig(os.path.join(OUT, "fig_lambda.pdf")); plt.close(fig)

# 5. routing over training + verifier-noise robustness
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.05))
ro = R["routing"]; ph = np.array(ro["phase"]) * 100
axs[0].stackplot(ph, ro["opd"], ro["rl"], ro["solved"], ro["drop"], colors=[C["opd"], C["grpo"], "#BBBBBB", "#444444"], alpha=0.85,
                 labels=[r"OPD-dominant ($\lambda>0.5$)", r"RL-dominant ($\lambda\leq0.5$)", "Saturated ($\\hat p_S{=}1$)", r"Dropped ($\hat p_T<\tau$)"])
axs[0].set_xlabel("Training progress (%)"); axs[0].set_ylabel("Fraction of prompts"); axs[0].set_ylim(0, 1); axs[0].set_xlim(0, 100)
axs[0].legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.85); axs[0].set_title("Routing, FinQA (8B)")
eps = np.array(R["noise"]["eps"])
for m in ["grpo", "kdrl", "gate"]:
    a = np.array(R["noise"][m]); mu = a.mean(1); e = 1.96 * a.std(1, ddof=1) / np.sqrt(a.shape[1])
    axs[1].errorbar(eps, mu, yerr=e, color=C[m], marker="o", ms=2.5, capsize=1.5, label=N[m])
axs[1].set_xlabel(r"Verifier flip rate $\varepsilon$"); axs[1].set_ylabel("FinQA accuracy (%)"); axs[1].legend(loc="lower left"); axs[1].set_title("Verifier-noise robustness")
fig.savefig(os.path.join(OUT, "fig_routing.pdf")); plt.close(fig)

# 6. ablation bars
fig, ax = plt.subplots(figsize=(5.5, 1.45))
A = R["ablation"]; keys = list(A.keys())
w = 0.38; xi = np.arange(len(keys))
for k_i, (t, col) in enumerate((("medqa", C["gate"]), ("finqa", C["grpo"]))):
    full = np.mean(A["full"][t])
    d = np.array([np.mean(A[k][t]) - full for k in keys]); e = np.array([np.std(A[k][t], ddof=1) for k in keys])
    ax.bar(xi + (k_i - 0.5) * w, d, w, yerr=e, color=col, capsize=1.5, error_kw={"lw": 0.6}, label=TN[t])
ax.axhline(0, color="k", lw=0.5)
lab = ["full", "global λ", "γ=0", "τ=0", "fwd KL", "1-sample $\\hat p_T$", "frozen $\\hat p_S$", "no prefix cap"]
ax.set_xticks(xi); ax.set_xticklabels(lab, fontsize=6.5); ax.set_ylabel("Δ accuracy vs. full (pts)"); ax.legend(loc="lower left")
fig.savefig(os.path.join(OUT, "fig_ablation.pdf")); plt.close(fig)
print("ok", sorted(os.listdir(OUT)))
