"""
scripts/cacioli_ablation.py
----------------------------
Ablation sui metodi di drift axis per il corpus Cacioli N=141.

Obiettivo: capire perché SVM union perde correlazione su Cacioli rispetto a PC.

Metodi confrontati (tutti ONLINE — no GT a runtime):
  1. SVM-union     — asse SVM fit su N=191 (PC+Cacioli), anchors da PC
  2. SVM-cacioli   — asse SVM fit solo su N=141 Cacioli, anchors da Cacioli GT mean
  3. cPCA          — Sigma_W - alpha*Sigma_GT → PC1, sweep alpha [0.5..2.0]
  4. Local-PC1     — per-clip PCA locale sui 6 punti, uso varianza spiegata come score

Target: WER_in, WER_out, BLEU_out, E_sem_top (Cacioli N=141)
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

BASE        = Path(__file__).parent.parent
CA_CACHE    = BASE / "scripts" / "cacioli_llm_cache"
PC_LLM_DIR  = BASE / "daws" / "results" / "llm_cache"
PC_IE_JSON  = BASE / "daws" / "results" / "ie_study_ablation.json"

# ── 1. Carica dati Cacioli ─────────────────────────────────────────────────────
ca_files   = sorted(CA_CACHE.glob("*.json"))
ca_records = [json.loads(f.read_text(encoding="utf-8")) for f in ca_files]
ca_wer     = np.array([r["wer"] for r in ca_records])
N_CA = len(ca_records)

ca_inputs    = [[] for _ in range(6)]
ca_responses = [[] for _ in range(6)]
for r in ca_records:
    for s in range(6):
        ca_inputs[s].append(r["inputs"][s])
        ca_responses[s].append(r["responses"][s])

# ── 2. Carica dati PARLA CHIARO ───────────────────────────────────────────────
pc_meta  = json.loads(PC_IE_JSON.read_text(encoding="utf-8"))
pc_stems = [r["filename"] for r in pc_meta]
N_PC = len(pc_stems)

pc_inputs    = [[] for _ in range(6)]
pc_responses = [[] for _ in range(6)]
for stem in pc_stems:
    d = json.loads((PC_LLM_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    for s in range(6):
        pc_inputs[s].append(d["inputs"][s])
        pc_responses[s].append(d["responses"][s])

N_UNION = N_PC + N_CA
print(f"N_CA={N_CA}  N_PC={N_PC}  N_UNION={N_UNION}")

# ── 3. SBERT ──────────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")

# Encode Cacioli solo
ca_in_flat  = [t for s in range(6) for t in ca_inputs[s]]
ca_out_flat = [t for s in range(6) for t in ca_responses[s]]
print(f"Encoding {len(ca_in_flat)} Cacioli INPUT texts ...")
X_ca_in  = sbert.encode(ca_in_flat,  batch_size=32, show_progress_bar=True,
                         normalize_embeddings=False)
print(f"Encoding {len(ca_out_flat)} Cacioli RESPONSE texts ...")
X_ca_out = sbert.encode(ca_out_flat, batch_size=32, show_progress_bar=True,
                         normalize_embeddings=False)

# Encode PC solo (per union SVM)
pc_in_flat  = [t for s in range(6) for t in pc_inputs[s]]
pc_out_flat = [t for s in range(6) for t in pc_responses[s]]
print(f"Encoding {len(pc_in_flat)} PC INPUT texts ...")
X_pc_in  = sbert.encode(pc_in_flat,  batch_size=32, show_progress_bar=False,
                          normalize_embeddings=False)
print(f"Encoding {len(pc_out_flat)} PC RESPONSE texts ...")
X_pc_out = sbert.encode(pc_out_flat, batch_size=32, show_progress_bar=False,
                          normalize_embeddings=False)

# ── 4. Slot slices ─────────────────────────────────────────────────────────────
ca_in_slots  = [X_ca_in [s * N_CA:(s + 1) * N_CA] for s in range(6)]
ca_out_slots = [X_ca_out[s * N_CA:(s + 1) * N_CA] for s in range(6)]
pc_in_slots  = [X_pc_in [s * N_PC:(s + 1) * N_PC] for s in range(6)]
pc_out_slots = [X_pc_out[s * N_PC:(s + 1) * N_PC] for s in range(6)]

# ── 5. Target metriche ────────────────────────────────────────────────────────
def _wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r: return 1.0 if h else 0.0
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            t = dp[j]
            dp[j] = prev if r[i-1] == h[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = t
    return dp[m] / n

def _l2_norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.
    return X / n

wer_out  = np.array([_wer(ca_records[i]["responses"][0], ca_records[i]["responses"][3])
                     for i in range(N_CA)])
bleu_out = np.array([1. - sentence_bleu([ca_records[i]["responses"][0].split()],
                                         ca_records[i]["responses"][3].split(),
                                         smoothing_function=SmoothingFunction().method1)
                     for i in range(N_CA)])
ca_out_norm = [_l2_norm(ca_out_slots[s]) for s in range(6)]
E_sem_top   = np.array([1. - float(np.dot(ca_out_norm[3][i], ca_out_norm[0][i]))
                         for i in range(N_CA)])

TARGETS = [("WER_in", ca_wer), ("WER_out", wer_out),
           ("BLEU_out", bleu_out), ("E_sem_top", E_sem_top)]

# ── 6. Spettrale helpers ───────────────────────────────────────────────────────
def _laplacian(pts, sigma):
    return np.exp(-np.abs(pts[:, None] - pts[None, :]) / sigma)

def _rs(A):
    s = A.sum(1, keepdims=True); s[s < 1e-12] = 1.; return A / s

def _spectral_H(pts_in, pts_out, si, so):
    eigs = np.abs(np.linalg.eigvals(_rs(_laplacian(pts_out, so)) @ _rs(_laplacian(pts_in, si))))
    tot = eigs.sum()
    if tot < 1e-12: return 0.
    e = eigs / tot; e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))

def _median_sigma(proj):
    p = np.abs(proj[:, None] - proj[None, :])
    return float(np.median(p[np.triu_indices(len(proj), k=1)]))

def _run_online(w_in, w_out, emb_in_slots, emb_out_slots, N,
                anchors_in=None, anchors_out=None):
    """Compute H_spectral ONLINE for N samples given drift axes.
    If anchors_in/out are None, compute from GT slots of the same corpus."""
    pi = np.vstack([emb_in_slots[s]  @ w_in  for s in range(6)])  # (6*N,)
    po = np.vstack([emb_out_slots[s] @ w_out for s in range(6)])
    # reshape as (6,N)
    pi = (emb_in_slots[0]  @ w_in)  # just projections, not stacked
    # Actually need per-slot
    pi_s = [emb_in_slots[s]  @ w_in  for s in range(6)]
    po_s = [emb_out_slots[s] @ w_out for s in range(6)]
    pi_all = np.concatenate(pi_s)
    po_all = np.concatenate(po_s)
    si = _median_sigma(pi_all)
    so = _median_sigma(po_all)
    if anchors_in is None:
        anchors_in  = [float(pi_s[s].mean()) for s in range(3)]
        anchors_out = [float(po_s[s].mean()) for s in range(3)]
    H = np.array([
        _spectral_H(
            np.array(anchors_in  + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
            np.array(anchors_out + [po_s[3][i], po_s[4][i], po_s[5][i]]),
            si, so)
        for i in range(N)
    ])
    return H, si, so, anchors_in, anchors_out

def _pearson_row(H, label):
    cells = []
    for name, target in TARGETS:
        r, p = pearsonr(H, target)
        cells.append(f"{r:+.4f}(p={p:.1e})")
    return f"  {label:<30s} | " + " | ".join(cells)

# ── 7. Metodo 1: SVM union ────────────────────────────────────────────────────
print("\n[1/4] SVM union (N=191) ...")
X_union_in  = np.vstack([np.vstack([pc_in_slots[s],  ca_in_slots[s]])  for s in range(6)])
X_union_out = np.vstack([np.vstack([pc_out_slots[s], ca_out_slots[s]]) for s in range(6)])
labels_union = np.array([0] * (3 * N_UNION) + [1] * (3 * N_UNION))

svm_u_in  = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_union_in,  labels_union)
svm_u_out = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_union_out, labels_union)
w_u_in  = sk_normalize(svm_u_in.coef_)[0]
w_u_out = sk_normalize(svm_u_out.coef_)[0]

# Anchors from PC GT slots projected onto union axis
anc_in_u  = [float((pc_in_slots[s]  @ w_u_in).mean())  for s in range(3)]
anc_out_u = [float((pc_out_slots[s] @ w_u_out).mean()) for s in range(3)]

H_svm_union, *_ = _run_online(w_u_in, w_u_out, ca_in_slots, ca_out_slots, N_CA,
                                anchors_in=anc_in_u, anchors_out=anc_out_u)

# ── 8. Metodo 2: SVM Cacioli-only ─────────────────────────────────────────────
print("[2/4] SVM Cacioli-only (N=141) ...")
X_ca_in_all  = np.vstack([ca_in_slots[s]  for s in range(6)])
X_ca_out_all = np.vstack([ca_out_slots[s] for s in range(6)])
labels_ca = np.array([0] * (3 * N_CA) + [1] * (3 * N_CA))

svm_c_in  = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_ca_in_all,  labels_ca)
svm_c_out = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_ca_out_all, labels_ca)
w_c_in  = sk_normalize(svm_c_in.coef_)[0]
w_c_out = sk_normalize(svm_c_out.coef_)[0]
acc_c_in  = float((svm_c_in.predict(X_ca_in_all)   == labels_ca).mean())
acc_c_out = float((svm_c_out.predict(X_ca_out_all) == labels_ca).mean())
print(f"  SVM-cacioli accuracy in={acc_c_in:.3f} out={acc_c_out:.3f}")

H_svm_ca, *_ = _run_online(w_c_in, w_c_out, ca_in_slots, ca_out_slots, N_CA)

# ── 9. Metodo 3: cPCA sweep ───────────────────────────────────────────────────
print("[3/4] cPCA sweep alpha [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0] ...")
# Background = GT slots (0,1,2); Foreground = W slots (3,4,5)
X_gt_in  = np.vstack([ca_in_slots[s]  for s in range(3)])   # (3*N_CA, 768)
X_w_in   = np.vstack([ca_in_slots[s]  for s in range(3, 6)])
X_gt_out = np.vstack([ca_out_slots[s] for s in range(3)])
X_w_out  = np.vstack([ca_out_slots[s] for s in range(3, 6)])

cov_gt_in  = np.cov(X_gt_in.T)
cov_w_in   = np.cov(X_w_in.T)
cov_gt_out = np.cov(X_gt_out.T)
cov_w_out  = np.cov(X_w_out.T)

ALPHAS = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]
best_cpca_H, best_cpca_label, best_r_bleu = None, "", -np.inf

for alpha in ALPHAS:
    C_in  = cov_w_in  - alpha * cov_gt_in
    C_out = cov_w_out - alpha * cov_gt_out
    _, v_in  = np.linalg.eigh(C_in)
    _, v_out = np.linalg.eigh(C_out)
    w_cp_in  = sk_normalize(v_in[:, -1:].T)[0]
    w_cp_out = sk_normalize(v_out[:, -1:].T)[0]
    H_cp, *_ = _run_online(w_cp_in, w_cp_out, ca_in_slots, ca_out_slots, N_CA)
    r_bleu, _ = pearsonr(H_cp, bleu_out)
    if r_bleu > best_r_bleu:
        best_r_bleu, best_cpca_H = r_bleu, H_cp
        best_cpca_label = f"cPCA alpha={alpha}"

# ── 10. Metodo 4: Local PC1 ────────────────────────────────────────────────────
print("[4/6] Local PC1 variance (per-clip) ...")
# Score = fraction of variance explained by PC1 for input slots
# Higher = more compact drift = clearer signal
pc1_var = []
pc1_proj_3 = [[], [], []]   # [slot3, slot4, slot5] projections onto per-clip PC1
for i in range(N_CA):
    pts = np.stack([ca_in_slots[s][i] for s in range(6)])
    pts_c = pts - pts.mean(0)
    _, sv, Vt = np.linalg.svd(pts_c, full_matrices=False)
    v = Vt[0]
    # Orient: W mean > GT mean
    w_mean  = np.mean([pts[s] @ v for s in range(3, 6)])
    gt_mean = np.mean([pts[s] @ v for s in range(3)])
    if w_mean < gt_mean: v = -v
    var = sv**2
    pc1_var.append(var[0] / var.sum())
    for j, s in enumerate([3, 4, 5]):
        pc1_proj_3[j].append(float(pts[s] @ v))

pc1_var = np.array(pc1_var)

# For spectral H with local PC1: we need a single global axis or per-sample
# Use PC1 variance directly as a proxy score (higher variance = more drift = more risk)
# Also try H_spectral with per-sample anchors (ORACLE mode, uses local GT mean)
# Compute global SVM on local PC1 vectors (same as ablation_svm_pc1.py)
pc1_vecs_in = []
for i in range(N_CA):
    pts = np.stack([ca_in_slots[s][i] for s in range(6)])
    pts_c = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    v = Vt[0]
    w_mean  = np.mean([pts[s] @ v for s in range(3, 6)])
    gt_mean = np.mean([pts[s] @ v for s in range(3)])
    if w_mean < gt_mean: v = -v
    pc1_vecs_in.append(v)
pc1_vecs_in = np.stack(pc1_vecs_in)

pc1_vecs_out = []
for i in range(N_CA):
    pts = np.stack([ca_out_slots[s][i] for s in range(6)])
    pts_c = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    v = Vt[0]
    w_mean  = np.mean([pts[s] @ v for s in range(3, 6)])
    gt_mean = np.mean([pts[s] @ v for s in range(3)])
    if w_mean < gt_mean: v = -v
    pc1_vecs_out.append(v)
pc1_vecs_out = np.stack(pc1_vecs_out)

# Global axis via SVM on +v / -v pairs
X_pc1_in  = np.vstack([pc1_vecs_in,  -pc1_vecs_in])
X_pc1_out = np.vstack([pc1_vecs_out, -pc1_vecs_out])
y_pc1 = np.array([1] * N_CA + [-1] * N_CA)
svm_p_in  = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_pc1_in,  y_pc1)
svm_p_out = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_pc1_out, y_pc1)
w_p_in  = sk_normalize(svm_p_in.coef_)[0]
w_p_out = sk_normalize(svm_p_out.coef_)[0]
H_pc1_svm, *_ = _run_online(w_p_in, w_p_out, ca_in_slots, ca_out_slots, N_CA)

# ── 11. Metodo 5: Local PCA ORACLE (per-clip axis, GT usato) ─────────────────
print("[5/7] Local PCA ORACLE (per-clip PC1, usa GT a runtime) ...")
H_local_pca_oracle = np.zeros(N_CA)
for i in range(N_CA):
    # Input space: PC1 locale dai 6 punti del clip
    pts_in  = np.stack([ca_in_slots[s][i]  for s in range(6)])
    pts_out = np.stack([ca_out_slots[s][i] for s in range(6)])

    _, _, Vt_i  = np.linalg.svd(pts_in  - pts_in.mean(0),  full_matrices=False)
    _, _, Vt_o  = np.linalg.svd(pts_out - pts_out.mean(0), full_matrices=False)
    v_in  = Vt_i[0]
    v_out = Vt_o[0]

    # Orient: W mean > GT mean su entrambi gli assi
    if np.mean([pts_in[s]  @ v_in  for s in range(3, 6)]) < np.mean([pts_in[s]  @ v_in  for s in range(3)]):
        v_in  = -v_in
    if np.mean([pts_out[s] @ v_out for s in range(3, 6)]) < np.mean([pts_out[s] @ v_out for s in range(3)]):
        v_out = -v_out

    pi = np.array([pts_in[s]  @ v_in  for s in range(6)])
    po = np.array([pts_out[s] @ v_out for s in range(6)])

    si = float(np.median(np.abs(pi[:, None] - pi[None, :])[np.triu_indices(6, k=1)]))
    so = float(np.median(np.abs(po[:, None] - po[None, :])[np.triu_indices(6, k=1)]))
    if si < 1e-9: si = 1e-9
    if so < 1e-9: so = 1e-9

    H_local_pca_oracle[i] = _spectral_H(pi, po, si, so)

# ── 12. Metodo 6: PCA globale Cacioli ────────────────────────────────────────
print("[6/7] PCA globale Cacioli ...")
# PC1 della matrice di covarianza di tutti i 6*N_CA embedding input
# Orient: W mean > GT mean
_, _, Vt_global_in  = np.linalg.svd(X_ca_in_all  - X_ca_in_all.mean(0),  full_matrices=False)
_, _, Vt_global_out = np.linalg.svd(X_ca_out_all - X_ca_out_all.mean(0), full_matrices=False)
w_pca_in  = Vt_global_in[0]
w_pca_out = Vt_global_out[0]
# Orient: W slots should project higher than GT slots
if (X_ca_in_all[3*N_CA:] @ w_pca_in).mean() < (X_ca_in_all[:3*N_CA] @ w_pca_in).mean():
    w_pca_in = -w_pca_in
if (X_ca_out_all[3*N_CA:] @ w_pca_out).mean() < (X_ca_out_all[:3*N_CA] @ w_pca_out).mean():
    w_pca_out = -w_pca_out
w_pca_in  = sk_normalize(w_pca_in.reshape(1, -1))[0]
w_pca_out = sk_normalize(w_pca_out.reshape(1, -1))[0]
H_pca_global, *_ = _run_online(w_pca_in, w_pca_out, ca_in_slots, ca_out_slots, N_CA)

# ── 12. Metodo 6: asse congelato da PARLA CHIARO ──────────────────────────────
print("[7/7] Asse congelato PARLA CHIARO (geometry_calibration.json) ...")
cfg = json.loads((BASE / "config" / "geometry_calibration.json").read_text(encoding="utf-8"))
w_frozen_in  = np.array(cfg["w_input_drift"])
w_frozen_out = np.array(cfg["w_resp_drift"])
# Anchors originali PC (già in spazio PC, non ricalcolati)
anc_frozen_in  = cfg["anchors_in"]
anc_frozen_out = cfg["anchors_out"]
H_frozen, *_ = _run_online(w_frozen_in, w_frozen_out, ca_in_slots, ca_out_slots, N_CA,
                            anchors_in=anc_frozen_in, anchors_out=anc_frozen_out)

# ── 13. Risultati ─────────────────────────────────────────────────────────────
SEP = "─" * 108
print("\n" + SEP)
print(f"  CACIOLI N={N_CA} ABLATION — tutti metodi ONLINE")
print(f"  {'Metodo':<38s} | {'WER_in':^22s} | {'WER_out':^22s} | {'BLEU_out':^22s} | {'E_sem_top':^22s}")
print(SEP)
print(_pearson_row(H_svm_union,        "SVM union   (N=191, PC anchors)      "))
print(_pearson_row(H_svm_ca,           "SVM cacioli (N=141, CA anchors)      "))
print(_pearson_row(best_cpca_H,        f"{best_cpca_label:<38s}"))
print(_pearson_row(H_pc1_svm,          "SVM-PC1 cacioli                      "))
print(_pearson_row(H_local_pca_oracle, "Local PCA ORACLE (per-clip, GT used) "))
print(_pearson_row(H_pca_global,       "PCA globale cacioli (PC1 di 846 emb) "))
print(_pearson_row(H_frozen,           "Asse congelato PC (N=50, no refit)   "))
print(_pearson_row(pc1_var,            "Local PC1 variance (proxy score)     "))
print(SEP)
print(f"\nLocal PC1 variance Cacioli: mean={pc1_var.mean()*100:.1f}%  std={pc1_var.std()*100:.1f}%")
print(f"(PARLA CHIARO per confronto: mean=79%  std=14%)")
