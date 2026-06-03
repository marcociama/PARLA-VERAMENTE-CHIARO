"""
scripts/mean_pc1_analysis.py
-----------------------------
Compute mean of local PC1 vectors on Cacioli N=141 as a global drift axis.
Reports:
  1. Mutual cosines between local PC1 vectors (alignment check)
  2. Cosines: mean-PC1 vs SVM-cacioli-only, vs frozen-PC axis
  3. H_spectral (ONLINE, no GT at runtime) with mean-PC1 axis → Pearson(WER_in, WER_out, BLEU_out, E_sem_top)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.svm import LinearSVC

BASE = Path(__file__).parent.parent
CACHE_DIR = BASE / "scripts" / "cacioli_llm_cache"
PC_CACHE  = BASE / "daws" / "results" / "llm_cache"
GEO_CFG   = BASE / "config" / "geometry_calibration.json"

# ── SBERT ────────────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")

def encode(texts):
    return sbert.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=False)

# ── Load Cacioli clips ────────────────────────────────────────────────────────
files = sorted(CACHE_DIR.glob("*.json"))
ca_data = [json.loads(f.read_text()) for f in files]
N_CA = len(ca_data)
print(f"Cacioli N={N_CA}")

# ── Encode all slots ──────────────────────────────────────────────────────────
print("Encoding Cacioli inputs/responses ...")
ca_inputs_flat  = [x for d in ca_data for x in d["inputs"]]
ca_outputs_flat = [x for d in ca_data for x in d["responses"]]
E_in  = encode(ca_inputs_flat).reshape(N_CA, 6, -1)   # (141, 6, 768)
E_out = encode(ca_outputs_flat).reshape(N_CA, 6, -1)

# ── Local PC1 per clip ────────────────────────────────────────────────────────
print("Computing local PC1 per clip ...")
D = E_in.shape[2]
local_pc1_in  = []
degenerate    = 0

for i in range(N_CA):
    X = E_in[i]  # (6, 768)
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    v = Vt[0]
    # Orient: W slots (3-5) projected higher than GT slots (0-2)
    gt_proj = Xc[:3] @ v
    w_proj  = Xc[3:] @ v
    if w_proj.mean() < gt_proj.mean():
        v = -v
    local_pc1_in.append(v)

local_pc1_in = np.array(local_pc1_in)  # (141, 768)

# ── Mutual cosines ────────────────────────────────────────────────────────────
print("Computing mutual cosines of local PC1 vectors ...")
# Sample N_SAMPLE random pairs
rng = np.random.default_rng(42)
N_SAMPLE = 1000
idx = rng.integers(0, N_CA, size=(N_SAMPLE, 2))
cosines = []
for a, b in idx:
    if a != b:
        c = float(local_pc1_in[a] @ local_pc1_in[b])
        cosines.append(c)
cosines = np.array(cosines)
print(f"Mutual cosine local PC1 — mean={cosines.mean():.4f}  std={cosines.std():.4f}  "
      f"median={np.median(cosines):.4f}")

# ── Mean PC1 as global axis ───────────────────────────────────────────────────
mean_pc1 = local_pc1_in.mean(axis=0)
mean_pc1 = mean_pc1 / np.linalg.norm(mean_pc1)

# ── SVM cacioli-only axis ─────────────────────────────────────────────────────
print("Fitting SVM cacioli-only ...")
X_in_ca  = E_in.reshape(N_CA * 6, D)
X_out_ca = E_out.reshape(N_CA * 6, D)
labels_ca = np.array([0]*3*N_CA + [1]*3*N_CA)
svm_ca = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_in_ca, labels_ca)
w_svm_ca = sk_normalize(svm_ca.coef_)[0]

# ── Load frozen PC axis ───────────────────────────────────────────────────────
geo = json.loads(GEO_CFG.read_text())
w_pc_frozen = np.array(geo["w_input_drift"])

# ── SVM union axis ────────────────────────────────────────────────────────────
print("Loading PC clips and fitting SVM union ...")
pc_files = sorted(PC_CACHE.glob("*.json"))
pc_data  = [json.loads(f.read_text()) for f in pc_files]
N_PC = len(pc_data)
pc_inputs_flat  = [x for d in pc_data for x in d["inputs"]]
pc_outputs_flat = [x for d in pc_data for x in d["responses"]]
E_in_pc  = encode(pc_inputs_flat).reshape(N_PC, 6, -1)
E_out_pc = encode(pc_outputs_flat).reshape(N_PC, 6, -1)

N_UNION = N_PC + N_CA
X_in_union  = np.concatenate([E_in_pc.reshape(N_PC*6, D),  E_in.reshape(N_CA*6, D)],  axis=0)
X_out_union = np.concatenate([E_out_pc.reshape(N_PC*6, D), E_out.reshape(N_CA*6, D)], axis=0)
labels_union = np.array([0]*3*N_UNION + [1]*3*N_UNION)
svm_union = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_in_union, labels_union)
w_svm_union = sk_normalize(svm_union.coef_)[0]

# ── Axis cosines ──────────────────────────────────────────────────────────────
def abs_cos(a, b):
    return abs(float(a @ b))

print("\nAxis alignment (absolute cosine — sign is arbitrary):")
print(f"  cos(mean-PC1, SVM-cacioli)  = {abs_cos(mean_pc1, w_svm_ca):.4f}")
print(f"  cos(mean-PC1, SVM-union)    = {abs_cos(mean_pc1, w_svm_union):.4f}")
print(f"  cos(mean-PC1, frozen-PC)    = {abs_cos(mean_pc1, w_pc_frozen):.4f}")
print(f"  cos(SVM-cacioli, SVM-union) = {abs_cos(w_svm_ca, w_svm_union):.4f}")
print(f"  cos(SVM-cacioli, frozen-PC) = {abs_cos(w_svm_ca, w_pc_frozen):.4f}")

# ── ONLINE H_spectral with mean-PC1 ──────────────────────────────────────────
# Anchors: re-project PC GT slots on mean-PC1 (same logic as union calibration)
pc_gt_slots_in  = E_in_pc[:, :3, :].reshape(N_PC*3, D)
pc_gt_slots_out = E_out_pc[:, :3, :].reshape(N_PC*3, D)
anchors_in  = (pc_gt_slots_in  @ mean_pc1).tolist()
anchors_out = (pc_gt_slots_out @ mean_pc1).tolist()

# Sigma from median pairwise on union projections
proj_in_union  = X_in_union  @ mean_pc1
proj_out_union = X_out_union @ mean_pc1

def median_pairwise_sigma(projs):
    from itertools import combinations
    idx = np.argsort(projs)
    # approximate on sorted values — adjacent differences
    diffs = np.abs(np.diff(np.sort(projs)))
    return float(np.median(diffs)) if len(diffs) > 0 else 1e-3

sigma_in  = float(np.std(proj_in_union))  * 0.5 + 1e-6
sigma_out = float(np.std(proj_out_union)) * 0.5 + 1e-6

def h_spectral_online(emb_in_6, emb_out_6, w_in, w_out, anch_in, anch_out, sig_in, sig_out):
    pi = emb_in_6  @ w_in   # (6,)
    po = emb_out_6 @ w_out  # (6,)
    pts_in  = np.array(anch_in  + [pi[3], pi[4], pi[5]])
    pts_out = np.array(anch_out + [po[3], po[4], po[5]])
    n = len(pts_in)

    def laplace_kernel(pts, sig):
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = np.exp(-abs(pts[i] - pts[j]) / sig)
        row_sum = M.sum(axis=1, keepdims=True)
        return M / (row_sum + 1e-12)

    Px = laplace_kernel(pts_in,  sig_in)
    Py = laplace_kernel(pts_out, sig_out)
    M  = Py @ Px
    eigvals = np.linalg.eigvals(M)
    eigvals = np.abs(eigvals.real)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.5
    eigvals /= eigvals.sum()
    return float(-np.sum(eigvals * np.log(eigvals + 1e-12)))

# Use 3 PC GT projections as anchors (same as union pipeline)
pc_gt_proj_in  = [float(e @ mean_pc1) for e in pc_gt_slots_in[:3]]
pc_gt_proj_out = [float(e @ mean_pc1) for e in pc_gt_slots_out[:3]]

print("\nComputing H_spectral (mean-PC1 axis) for Cacioli N=141 ...")
h_vals = []
for i in range(N_CA):
    h = h_spectral_online(
        E_in[i], E_out[i],
        mean_pc1, mean_pc1,
        pc_gt_proj_in, pc_gt_proj_out,
        sigma_in, sigma_out
    )
    h_vals.append(h)
h_vals = np.array(h_vals)

wer_in   = np.array([d["wer"]         for d in ca_data])
try:
    wer_out  = np.array([d["wer_out"]     for d in ca_data])
    bleu_out = np.array([d["bleu_out"]    for d in ca_data])
    e_sem    = np.array([d["e_sem_top"]   for d in ca_data])
    has_output_metrics = True
except KeyError:
    has_output_metrics = False
    print("  (wer_out/bleu_out/e_sem not in JSON — loading from cacioli_results.json)")

if not has_output_metrics:
    results_path = BASE / "scripts" / "cacioli_results.json"
    if results_path.exists():
        res = json.loads(results_path.read_text())
        per_clip = {c["filename"]: c for c in res["per_clip"]}
        filenames = [d["filename"] for d in ca_data]
        wer_out  = np.array([per_clip[f]["wer_out"]   for f in filenames])
        bleu_out = np.array([per_clip[f]["bleu_out"]  for f in filenames])
        e_sem    = np.array([per_clip[f]["e_sem_top"] for f in filenames])
        has_output_metrics = True

def pr(x, y, name):
    r, p = pearsonr(x, y)
    return f"  Pearson(H_mean-PC1, {name:12s}) = {r:+.4f}  (p={p:.2e})"

print("\n" + "="*65)
print("  Mean-PC1 global axis — Cacioli N=141")
print("="*65)
print(pr(h_vals, wer_in, "WER_in"))
if has_output_metrics:
    print(pr(h_vals, wer_out,  "WER_out"))
    print(pr(h_vals, bleu_out, "BLEU_out"))
    print(pr(h_vals, e_sem,    "E_sem_top"))
print("="*65)
print(f"  H range: min={h_vals.min():.4f}  max={h_vals.max():.4f}  mean={h_vals.mean():.4f}")
print()
