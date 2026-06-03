"""
scripts/generate_figures.py
----------------------------
Genera le figure per la sezione Geometric Validation del paper:
  fig_projection_vs_wer.pdf  — scatter Δ_i vs WER_in  (r=+0.437)
  fig_hspectral_vs_wer.pdf   — scatter H_spectral vs WER_in  (r=+0.501)
  fig_pc1_variance.pdf       — histogram varianza locale PC1
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

# ── dati ──────────────────────────────────────────────────────────────────────
ie_records = json.loads((BASE / "daws/results/ie_study_ablation.json").read_text())
stems  = [r["filename"] for r in ie_records]
wer_in = np.array([r["wer"] for r in ie_records])
N = len(stems)

inputs_by_slot    = [[] for _ in range(6)]
responses_by_slot = [[] for _ in range(6)]
for stem in stems:
    d = json.loads((BASE / "daws/results/llm_cache" / f"{stem}.json").read_text())
    for slot in range(6):
        inputs_by_slot[slot].append(d["inputs"][slot])
        responses_by_slot[slot].append(d["responses"][slot])

# ── SBERT ─────────────────────────────────────────────────────────────────────
print("Loading SBERT...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
X_in  = sbert.encode([t for s in range(6) for t in inputs_by_slot[s]],
                     batch_size=32, show_progress_bar=False, normalize_embeddings=False)
X_out = sbert.encode([t for s in range(6) for t in responses_by_slot[s]],
                     batch_size=32, show_progress_bar=False, normalize_embeddings=False)

in_slots  = [X_in [k*N:(k+1)*N] for k in range(6)]
out_slots = [X_out[k*N:(k+1)*N] for k in range(6)]

# ── SVM drift axis ────────────────────────────────────────────────────────────
labels = np.array([0]*(3*N) + [1]*(3*N))
svm_i  = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_in,  labels)
svm_o  = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_out, labels)
w_in   = sk_normalize(svm_i.coef_)[0]
w_out  = sk_normalize(svm_o.coef_)[0]

pi_all = X_in  @ w_in;  pi_s = [pi_all[k*N:(k+1)*N] for k in range(6)]
po_all = X_out @ w_out; po_s = [po_all[k*N:(k+1)*N] for k in range(6)]

delta_in = np.array([np.mean([pi_s[s][i] for s in range(3,6)]) -
                     np.mean([pi_s[s][i] for s in range(3)])    for i in range(N)])

# ── H_spectral ────────────────────────────────────────────────────────────────
def _lap(x, s): return np.exp(-np.abs(x[:,None]-x[None,:])/s)
def _rs(A): s=A.sum(1,keepdims=True); s[s<1e-12]=1.; return A/s
def med_sig(x): p=np.abs(x[:,None]-x[None,:]); return float(np.median(p[np.triu_indices(len(x),k=1)]))
def spectral_H(pi, po, si, so):
    eigs = np.abs(np.linalg.eigvals(_rs(_lap(po,so)) @ _rs(_lap(pi,si))))
    tot = eigs.sum()
    if tot < 1e-12: return 0.
    e = eigs/tot; e = e[e>1e-12]
    return float(-np.sum(e*np.log(e)))

si, so = med_sig(pi_all), med_sig(po_all)
ai = [float(pi_s[s].mean()) for s in range(3)]
ao = [float(po_s[s].mean()) for s in range(3)]

H_spec = np.array([
    spectral_H(np.array(ai + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
               np.array(ao + [po_s[3][i], po_s[4][i], po_s[5][i]]), si, so)
    for i in range(N)])

H_MIN, H_MAX = 0.4320, 0.8452
H_risk = np.clip((H_spec - H_MIN) / (H_MAX - H_MIN), 0., 1.)

# ── local PC1 variance ────────────────────────────────────────────────────────
pc1_var_in = []
for i in range(N):
    pts = np.stack([in_slots[s][i] for s in range(6)])
    pts_c = pts - pts.mean(axis=0)
    _, sv, _ = np.linalg.svd(pts_c, full_matrices=False)
    var = sv**2; pc1_var_in.append(var[0]/var.sum())
pc1_var_in = np.array(pc1_var_in)

# ── stile comune ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
FIG_W = 3.3   # inches — single column

cmap = mcolors.LinearSegmentedColormap.from_list("risk", ["#2ecc71","#f39c12","#e74c3c"])

# ── Fig 1: Δ_i vs WER_in ─────────────────────────────────────────────────────
r1, p1 = pearsonr(delta_in, wer_in)
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.85))
sc = ax.scatter(delta_in, wer_in, c=H_risk, cmap=cmap, vmin=0, vmax=1,
                s=28, edgecolors="white", linewidths=0.4, zorder=3)
cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
cb.set_label("$H_{\\mathrm{risk}}$", fontsize=8)
m, b = np.polyfit(delta_in, wer_in, 1)
xs = np.linspace(delta_in.min(), delta_in.max(), 100)
ax.plot(xs, m*xs+b, color="#555", lw=1.2, ls="--", zorder=2)
ax.set_xlabel("Projection displacement $\\Delta_i$ on $\\mathbf{w}_{\\mathrm{in}}$")
ax.set_ylabel("WER$_{\\mathrm{in}}$")
ax.text(0.97, 0.05, f"$r = {r1:+.3f}$\n$p = {p1:.1e}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
fig.tight_layout()
fig.savefig(FIGURES / "fig_projection_vs_wer.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_projection_vs_wer.pdf")

# ── Fig 2: H_spectral vs WER_in ───────────────────────────────────────────────
r2, p2 = pearsonr(H_spec, wer_in)
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.85))
sc = ax.scatter(H_spec, wer_in, c=H_risk, cmap=cmap, vmin=0, vmax=1,
                s=28, edgecolors="white", linewidths=0.4, zorder=3)
cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
cb.set_label("$H_{\\mathrm{risk}}$", fontsize=8)
m, b = np.polyfit(H_spec, wer_in, 1)
xs = np.linspace(H_spec.min(), H_spec.max(), 100)
ax.plot(xs, m*xs+b, color="#555", lw=1.2, ls="--", zorder=2)
ax.axvline(H_MIN + 0.39*(H_MAX-H_MIN), color="#2ecc71", lw=1, ls=":", alpha=0.7, label="$\\tau_{\\mathrm{green}}$")
ax.axvline(H_MIN + 0.52*(H_MAX-H_MIN), color="#e74c3c", lw=1, ls=":", alpha=0.7, label="$\\tau_{\\mathrm{red}}$")
ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
ax.set_xlabel("$H_{\\mathrm{spectral}}$ (nats)")
ax.set_ylabel("WER$_{\\mathrm{in}}$")
ax.text(0.97, 0.05, f"$r = {r2:+.3f}$\n$p = {p2:.1e}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
fig.tight_layout()
fig.savefig(FIGURES / "fig_hspectral_vs_wer.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_hspectral_vs_wer.pdf")

# ── Fig 3: histogram PC1 variance ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.75))
ax.hist(pc1_var_in * 100, bins=10, color="#4a90d9", edgecolor="white",
        linewidth=0.6, zorder=3)
ax.axvline(pc1_var_in.mean()*100, color="#e74c3c", lw=1.5, ls="--")
ax.set_xlabel("Local PC1 variance explained (%)")
ax.set_ylabel("Count")
ax.text(0.97, 0.93,
        f"mean = {pc1_var_in.mean()*100:.0f}%\nstd = {pc1_var_in.std()*100:.0f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
fig.tight_layout()
fig.savefig(FIGURES / "fig_pc1_variance.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved fig_pc1_variance.pdf")

print("\nAll figures saved to", FIGURES)
