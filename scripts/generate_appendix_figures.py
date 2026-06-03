"""
scripts/generate_appendix_figures.py
Genera figure per l'appendice Cacioli:
  fig_cacioli_scatter.pdf     — H_spectral vs WER_in + BLEU_out (2 pannelli, colored by domain)
  fig_cacioli_ablation.pdf    — bar chart metodi vs BLEU_out + E_sem_top
  fig_cacioli_raw_vs_hspec.pdf — confronto raw correlation vs H_spectral su PC e Cacioli
"""

import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

BASE    = Path(__file__).parent.parent
RES     = BASE / "scripts" / "cacioli_results_v2.json"
FIGDIR  = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
res      = json.loads(RES.read_text())
per_clip = res["per_clip"]

h      = np.array([c["h_spectral"] for c in per_clip])
wer_in = np.array([c["wer"]        for c in per_clip])   # min over 3 GT variants
wer_out= np.array([c["wer_out"]    for c in per_clip])
bleu   = np.array([c["bleu_out"]   for c in per_clip])
esem   = np.array([c["e_sem_top"]  for c in per_clip])
domains= [c["domain"]              for c in per_clip]

dom_map   = {"Play": "#2196F3", "Poetry": "#FF9800", "Blog": "#4CAF50"}
dom_colors= [dom_map[d] for d in domains]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: scatter H_spectral vs WER_in and BLEU_out
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
fig.subplots_adjust(wspace=0.35)

for ax, y, ylabel, target in zip(
    axes,
    [wer_in, esem],
    ["WER$_{in}$", "$E_{sem}^{top}$"],
    ["wer_in", "e_sem_top"]
):
    ax.scatter(h, y, c=dom_colors, alpha=0.55, s=22, linewidths=0)
    r, p = pearsonr(h, y)
    # regression line
    m, b = np.polyfit(h, y, 1)
    xr = np.linspace(h.min(), h.max(), 100)
    ax.plot(xr, m*xr + b, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel(r"$H_{\mathrm{spectral}}$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    sign = "+" if r >= 0 else ""
    ax.set_title(fr"$r={sign}{r:.3f}$, $p={p:.1e}$", fontsize=10)
    ax.tick_params(labelsize=9)

# cap WER_in axis at 1.5 (7 outliers above skew the scale)
axes[0].set_ylim(-0.05, 1.55)
axes[0].set_xlabel(r"$H_{\mathrm{spectral}}$", fontsize=11)

# legend
patches = [mpatches.Patch(color=c, label=d) for d, c in dom_map.items()]
axes[1].legend(handles=patches, fontsize=8, loc="upper left", framealpha=0.7)

fig.suptitle("Cacioli Corpus ($N=141$): $H_{\\mathrm{spectral}}$ vs. Quality Metrics "
             r"(WER$_{in}$ and $E_{sem}^{top}$)", fontsize=11, y=1.01)
fig.savefig(FIGDIR / "fig_cacioli_scatter.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_cacioli_scatter.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: ablation bar chart
# ─────────────────────────────────────────────────────────────────────────────
methods = [
    "SVM union\n(N=191)",
    "SVM Cacioli\n(N=141)",
    r"cPCA $\alpha$=1.5",
    "SVM-PC1",
    "Local PCA\nORACLE",
    "Global PCA",
    "Frozen PC\naxis",
    "Mean-PC1",
]
bleu_vals = [0.2950, 0.3138, 0.1464, 0.1743, -0.1482, -0.2180, 0.0715, 0.0459]
esem_vals = [0.3268, 0.3275, 0.1451, 0.2339, -0.0430, -0.2332, 0.2034, 0.0421]
p_bleu    = [3.8e-4, 1.5e-4, 8.3e-2, 3.9e-2, 7.9e-2,  9.4e-3,  4.0e-1, 5.9e-1]
p_esem    = [7.6e-5, 7.4e-5, 8.6e-2, 5.3e-3, 6.1e-1,  5.4e-3,  1.6e-2, 6.2e-1]

x     = np.arange(len(methods))
width = 0.38
fig, ax = plt.subplots(figsize=(10, 4))
bars1 = ax.bar(x - width/2, bleu_vals, width, label="BLEU$_{out}$",
               color="#1976D2", alpha=0.85)
bars2 = ax.bar(x + width/2, esem_vals, width, label="$E_{sem}^{top}$",
               color="#E64A19", alpha=0.85)

# significance markers
def sig_marker(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

for bar, p in zip(bars1, p_bleu):
    m = sig_marker(p)
    if m:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005 if bar.get_height() >= 0 else bar.get_height() - 0.025,
                m, ha="center", va="bottom", fontsize=8, color="#1976D2")

for bar, p in zip(bars2, p_esem):
    m = sig_marker(p)
    if m:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005 if bar.get_height() >= 0 else bar.get_height() - 0.025,
                m, ha="center", va="bottom", fontsize=8, color="#E64A19")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel("Pearson $r$", fontsize=11)
ax.set_title("Cacioli Ablation: Drift Axis Methods ($N=141$, ONLINE)", fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(-0.32, 0.42)
ax.tick_params(axis="y", labelsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_cacioli_ablation.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_cacioli_ablation.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: raw correlation vs H_spectral — PC vs Cacioli
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.5))

metrics    = ["WER$_{in}$", "WER$_{out}$", "BLEU$_{out}$", "$E_{sem}^{top}$"]
pc_raw     = [0.501, 0.457, 0.568, 0.569]   # H_spectral on PC (main paper)
ca_hspec   = [0.424, 0.189, 0.277, 0.434]   # H_spectral on Cacioli v2 (neutral prompt)
ca_raw_corr= [0.317, None, None, None]       # raw WER_min→BLEU on Cacioli

x = np.arange(len(metrics))
w = 0.28
ax.bar(x - w, pc_raw,   w, label="PC ($N=50$)",         color="#1565C0", alpha=0.85)
ax.bar(x,     ca_hspec, w, label="Cacioli $H_{\\mathrm{spectral}}$ ($N=141$)", color="#F57F17", alpha=0.85)

# raw ceiling line on Cacioli for BLEU_out
ax.annotate("", xy=(2 + w/2 + 0.05, 0.317), xytext=(2 - w - 0.05, 0.317),
            arrowprops=dict(arrowstyle="-", color="gray", lw=1.2, linestyle="dashed"))
ax.text(2 + w/2 + 0.08, 0.321, "raw WER$_{min}$→BLEU\nceiling Cacioli", fontsize=7.5,
        color="gray", va="bottom")

ax.set_xticks(x - w/2)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Pearson $r$", fontsize=11)
ax.set_title("$H_{\\mathrm{spectral}}$ Correlation: PARLA CHIARO vs. Cacioli", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.70)
ax.tick_params(axis="y", labelsize=9)
ax.axhline(0, color="black", lw=0.5)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_cacioli_vs_pc.pdf", bbox_inches="tight")
plt.close()
print("Saved fig_cacioli_vs_pc.pdf")

print("\nAll figures saved to figures/")
