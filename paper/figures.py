"""All figures, drawn from results.json written by make_data.py.

Drawn at the size they are printed in the arXiv preprint style (arxiv.sty, a 6.5 in text block): every figure is
FULL = 6.5 in wide, with 8 pt labels and titles and 7 pt ticks and legends. Each PDF is exactly its figsize (no bbox
cropping), so the paper includes it at its natural size and web/scripts/paper-figure.mjs measures the printed width.
Bands and error bars over the three training seeds are 95% t-intervals (2 degrees of freedom).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results.json")))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
FULL = 6.5  # the arXiv text width (inches)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],  # Times metrics, TrueType
    "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "axes.grid": True, "lines.linewidth": 1.2, "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.2,
    "axes.titlepad": 3, "axes.labelpad": 2, "xtick.major.pad": 2, "ytick.major.pad": 2,
})
C = {"grpo": "#0072B2", "opd": "#009E73", "kdrl": "#E69F00", "gate": "#D55E00", "sft": "#999999", "dapo": "#56B4E9",
     "sft_grpo": "#CC79A7", "opd_grpo": "#C9B400", "teacher": "#000000"}
N = {"grpo": "GRPO", "opd": "OPD", "kdrl": "KDRL", "gate": "GATE (ours)", "sft": "SFT", "dapo": "DAPO",
     "sft_grpo": r"SFT$\rightarrow$GRPO", "opd_grpo": r"OPD$\rightarrow$GRPO"}
TN = R["tname"]
T975 = {2: 4.303, 3: 3.182, 4: 2.776}  # Student t quantiles for a two-sided 95% interval, by degrees of freedom


def ci(a, axis=0):
    """Half-width of the 95% t-interval of the mean over seeds (along `axis`)."""
    n = a.shape[axis]
    return T975[n - 1] * a.std(axis, ddof=1) / np.sqrt(n)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# 1. training curves vs compute (two panels, full width)
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.45), layout="constrained")
st = np.array(R["curves"]["steps"])
for ax, t in zip(axs, ["medqa", "finqa"]):
    for m in ["grpo", "opd", "kdrl", "gate"]:
        a = np.array(R["curves"]["data"][t][m])
        x = st * R["curves"]["cost"][m] * R["curves"]["node_h"]
        mu = a.mean(0)
        ax.plot(x, mu, color=C[m], lw=1.6 if m == "gate" else 1.1, label=N[m])
        ax.fill_between(x, mu - ci(a), mu + ci(a), color=C[m], alpha=0.18, lw=0)
    ax.axhline(R["teacher"][t], ls="--", color="k", lw=0.8, label="Teacher (235B)")
    ax.set_title(TN[t]); ax.set_xticks([0, 10, 20, 30, 40, 50, 60]); ax.set_xlim(0, 60)
    ax.set_xlabel("Training compute (8×H100 node-hours)")
axs[0].set_ylabel("Test accuracy (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5)
save(fig, "fig_curves.pdf")

# 2. scaling across student sizes (four panels, full width)
fig, axs = plt.subplots(1, 4, figsize=(FULL, 2.05), sharex=True, layout="constrained")
sz = [1.7, 4, 8]
for ax, t in zip(axs, R["tasks"]):
    for m in ["prompted", "opd", "grpo", "gate"]:
        a = np.array([R["grid"][s][t][m] for s in R["sizes"]])
        mu = a.mean(1)
        e = np.zeros_like(mu) if m == "prompted" else ci(a, axis=1)
        col = C.get(m, "#777777")
        ax.errorbar(sz, mu, yerr=e, color=col, marker="o", ms=2.8, lw=1.5 if m == "gate" else 1.0, capsize=1.6,
                    ls=":" if m == "prompted" else "-", label={"prompted": "Prompted"}.get(m, N.get(m)))
    ax.axhline(R["teacher"][t], ls="--", color="k", lw=0.8, label="Teacher (235B)")
    ax.set_xscale("log"); ax.set_xticks(sz); ax.set_xticklabels(["1.7", "4", "8"]); ax.minorticks_off(); ax.set_xlim(1.45, 9.4)
    ax.set_title(TN[t]); ax.set_xlabel("Student size (B)")
axs[0].set_ylabel("Accuracy (%)")
h, l = axs[0].get_legend_handles_labels()
h, l = h[1:] + h[:1], l[1:] + l[:1]  # teacher line last
fig.legend(h, l, loc="outside upper center", ncol=5)
save(fig, "fig_scaling.pdf")

# 3. headroom predictor + compute frontier (two panels, full width)
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.55), layout="constrained")
H = R["headroom"]; x = np.array(H["x"])
mk = {"1.7B": "o", "4B": "s", "8B": "^"}
for i, s in enumerate(R["sizes"]):
    sl = slice(4 * i, 4 * i + 4)
    axs[0].scatter(x[sl], np.array(H["grpo_gap"])[sl], marker=mk[s], s=20, color=C["grpo"], alpha=0.8, lw=0)
    axs[0].scatter(x[sl], np.array(H["gate_gap"])[sl], marker=mk[s], s=20, color=C["gate"], lw=0)
xx = np.linspace(0, x.max() * 1.05, 50)
for key, m in (("fit_grpo", "grpo"), ("fit_gate", "gate")):
    a, c, r2 = H[key]
    axs[0].plot(xx, a * xx + c, color=C[m], lw=1.0)
axs[0].axhline(0, color="k", lw=0.5)
axs[0].set_xlabel(r"Coverage deficit $1-\rho_{64}$ of the untrained student"); axs[0].set_ylabel("Gap to teacher after training (pts)")
fit_h = [Line2D([], [], color=C[m], lw=1.0,
                label=f"{N[m].split()[0]} fit: $R^2$ = {H[k][2]:.2f}, held-out MAE {H['loto_' + m]['mae']:.1f}")
         for k, m in (("fit_grpo", "grpo"), ("fit_gate", "gate"))]
size_h = [Line2D([], [], ls="none", marker=mk[s], ms=4.5, color="#555555", label=f"Qwen3-{s}") for s in R["sizes"]]
axs[0].legend(handles=fit_h + size_h, loc="upper left", ncol=1, handletextpad=0.5)
OFF = {"sft": (5, -1, "left"), "opd": (5, -3, "left"), "grpo": (-5, -1, "right"), "dapo": (0, -10, "center"),
       "sft_grpo": (5, -1, "left"), "opd_grpo": (-5, 1, "right"), "kdrl": (5, -2, "left"), "gate": (-5, 2, "right")}
for m, v in R["compute"].items():
    axs[1].scatter(v["node_h"], v["acc"], s=24, color=C[m], zorder=3, edgecolor="k" if m == "gate" else "none", lw=0.5)
    dx, dy, ha = OFF[m]
    axs[1].annotate(N[m], (v["node_h"], v["acc"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=7)
tavg = np.mean(list(R["teacher"].values()))
axs[1].axhline(tavg, ls="--", color="k", lw=0.8)
axs[1].text(12, tavg + 0.2, "Teacher (235B)", fontsize=7, va="bottom")
axs[1].set_xlim(10, 69); axs[1].set_ylim(73.3, 83.8)
axs[1].set_xlabel("Training compute (8×H100 node-hours)"); axs[1].set_ylabel("Mean accuracy, four tasks (%)")
save(fig, "fig_headroom.pdf")

# 4. lambda surface and sensitivity (appendix, full width)
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.35), layout="constrained")
L = np.array(R["lambda_surface"]["lam"])
im = axs[0].imshow(L, origin="lower", cmap="cividis", extent=[0, 1, 0, 1], vmin=0, vmax=1, aspect="auto")
axs[0].set_xlabel(r"Student solve rate $\hat p_S(x)$"); axs[0].set_ylabel(r"Teacher solve rate $\hat p_T(x)$")
axs[0].set_title(r"Mixing weight $\lambda(x)$"); axs[0].grid(False)
fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.02)
S = np.array(R["sensitivity"]["acc"])
im2 = axs[1].imshow(S, origin="lower", cmap="viridis", aspect="auto")
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        axs[1].text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w" if S[i, j] < S.max() - 2 else "k")
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(R["sensitivity"]["beta"]); axs[1].set_yticks(range(5)); axs[1].set_yticklabels(R["sensitivity"]["gamma"])
axs[1].set_xlabel(r"Temperature $\beta$"); axs[1].set_ylabel(r"Variance weight $\gamma$"); axs[1].set_title("FinQA accuracy (8B)"); axs[1].grid(False)
for sp in axs[1].spines.values():
    sp.set_visible(False)
save(fig, "fig_lambda.pdf")

# 5. routing over training + verifier-noise robustness (two panels, full width)
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.45), layout="constrained")
ro = R["routing"]; ph = np.array(ro["phase"]) * 100
axs[0].stackplot(ph, ro["opd"], ro["rl"], ro["solved"], ro["drop"], colors=[C["opd"], C["grpo"], "#BBBBBB", "#444444"], alpha=0.85,
                 labels=[r"OPD-dominant ($\lambda>0.5$)", r"RL-dominant ($\lambda\leq0.5$)", r"Saturated ($\hat p_S{=}1$)", r"Dropped ($\hat p_T<\tau$)"])
axs[0].set_xlabel("Training progress (%)"); axs[0].set_ylabel("Fraction of prompts"); axs[0].set_ylim(0, 1); axs[0].set_xlim(0, 100)
axs[0].set_xticks([0, 25, 50, 75, 100]); axs[0].set_title("Routing, FinQA (8B)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=4, handlelength=1.0)
eps = np.array(R["noise"]["eps"])
for m in ["grpo", "kdrl", "gate"]:
    a = np.array(R["noise"][m]); mu = a.mean(1)
    axs[1].errorbar(eps, mu, yerr=ci(a, axis=1), color=C[m], marker="o", ms=2.8, capsize=1.6, lw=1.5 if m == "gate" else 1.0,
                    label=N[m].split()[0])
axs[1].set_xlabel(r"Verifier flip rate $\varepsilon$"); axs[1].set_ylabel("FinQA accuracy (%)"); axs[1].set_title("Verifier noise")
axs[1].set_xticks(eps); axs[1].set_ylim(55, 70); axs[1].set_yticks([56, 60, 64, 68]); axs[1].legend(loc="lower left", handlelength=1.2)
save(fig, "fig_routing.pdf")

# 6. ablation bars (appendix, full width)
fig, ax = plt.subplots(figsize=(FULL, 1.95), layout="constrained")
A = R["ablation"]; keys = list(A.keys())
w = 0.38; xi = np.arange(len(keys))
for k_i, (t, col) in enumerate((("medqa", C["gate"]), ("finqa", C["grpo"]))):
    full = np.mean(A["full"][t])
    d = np.array([np.mean(A[k][t]) - full for k in keys]); e = np.array([np.std(A[k][t], ddof=1) for k in keys])
    ax.bar(xi + (k_i - 0.5) * w, d, w, yerr=e, color=col, capsize=1.6, error_kw={"lw": 0.6}, label=TN[t])
ax.axhline(0, color="k", lw=0.5)
lab = ["full", "global λ", "γ = 0", "τ = 0", "forward KL", "1-sample $\\hat p_T$", "frozen $\\hat p_S$", "no prefix cap"]
ax.set_xticks(xi); ax.set_xticklabels(lab); ax.set_ylabel("Δ accuracy vs. full (pts)"); ax.legend(loc="lower right")
ax.grid(axis="x", visible=False)
save(fig, "fig_ablation.pdf")
print("ok", sorted(os.listdir(OUT)))
