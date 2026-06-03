"""
scripts/benchmark_final.py
--------------------------
Benchmark definitivo PARLA CHIARO — 3 metodi UQ (N=50, LLM Mistral greedy T=0)

Metodi confrontati:
  1. Inv-Entropy H_k1  — offline upper bound (Song NeurIPS 2025, k=1, SBERT 768D)
  2. 1D Markov Spettrale ORACLE  — Laplace 1D, proiezioni GT reali per-sample
  3. 1D Markov Spettrale ONLINE  — Laplace 1D, ancoraggi congelati (NO GT a runtime)

Target di correlazione:
  - WER        Word Error Rate WhisperX vs ground truth
  - E_sem_top  1 - cos_sim(R_W1, R_GT1)  768D SBERT
  - E_sem_cross  1 - mean(cos_sim) su 9 coppie W×GT  768D SBERT

Al termine salva config/geometry_calibration.json con i parametri di calibrazione
del metodo ONLINE per il runtime del DAWSPipeline.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer

BASE       = Path(__file__).parent.parent
LLM_DIR    = BASE / "daws" / "results" / "llm_cache"
IE_JSON    = BASE / "daws" / "results" / "ie_study_ablation.json"
CFG_OUT    = BASE / "config" / "geometry_calibration.json"
DATASET_PATH = BASE / "outputs_geometrici" / "dataset_topologico_50.json"

# ── 1. Dati ────────────────────────────────────────────────────────────────────
ie_records = json.loads(IE_JSON.read_text(encoding="utf-8"))
stems      = [r["filename"] for r in ie_records]
wer_arr    = np.array([r["wer"]  for r in ie_records])
H_ie_k1    = np.array([r["H_k1"] for r in ie_records])
N          = len(stems)

inputs_by_slot    = [[] for _ in range(6)]
responses_by_slot = [[] for _ in range(6)]
for stem in stems:
    p = LLM_DIR / f"{stem}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for slot in range(6):
            inputs_by_slot[slot].append(d["inputs"][slot])
            responses_by_slot[slot].append(d["responses"][slot])

inputs_flat    = [t for slot in range(6) for t in inputs_by_slot[slot]]
responses_flat = [t for slot in range(6) for t in responses_by_slot[slot]]

# ── 2. SBERT ──────────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
print("Encoding 300 INPUT texts ...")
X_in  = sbert.encode(inputs_flat,    batch_size=32, show_progress_bar=False,
                     normalize_embeddings=False)
print("Encoding 300 OUTPUT texts ...")
X_out = sbert.encode(responses_flat, batch_size=32, show_progress_bar=False,
                     normalize_embeddings=False)

emb_out_slots = [X_out[k * N:(k + 1) * N] for k in range(6)]

# ── 3. E_sem targets (768D, invarianti al kernel) ─────────────────────────────
def _l2_norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n

emb_out_norm = [_l2_norm(emb_out_slots[s]) for s in range(6)]

E_sem_top = np.array([
    1.0 - float(np.dot(emb_out_norm[3][i], emb_out_norm[0][i]))
    for i in range(N)
])
E_sem_cross = np.array([
    1.0 - np.mean([float(np.dot(emb_out_norm[w][i], emb_out_norm[g][i]))
                   for w in [3, 4, 5] for g in [0, 1, 2]])
    for i in range(N)
])
print(f"E_sem_top:   mean={E_sem_top.mean():.4f}  "
      f"range=[{E_sem_top.min():.4f}, {E_sem_top.max():.4f}]")
print(f"E_sem_cross: mean={E_sem_cross.mean():.4f}  "
      f"range=[{E_sem_cross.min():.4f}, {E_sem_cross.max():.4f}]")

# ── 4. Direzione di drift ─────────────────────────────────────────────────────
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize

emb_in_slots  = [X_in [k * N:(k + 1) * N] for k in range(6)]
emb_out_slots_all = [X_out[k * N:(k + 1) * N] for k in range(6)]

# ── 4a. SVM raw (GT=0, W=1) su 300 embedding — metodo produzione ──────────────
labels = np.array([0] * (3 * N) + [1] * (3 * N))

svm_in  = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_in.fit(X_in,  labels)
w_drift_in_svm  = sk_normalize(svm_in.coef_)[0]
acc_in  = float((svm_in.predict(X_in)  == labels).mean())

svm_out = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_out.fit(X_out, labels)
w_drift_out_svm = sk_normalize(svm_out.coef_)[0]
acc_out = float((svm_out.predict(X_out) == labels).mean())
print(f"SVM-raw  accuracy — input: {acc_in:.3f}  |  output: {acc_out:.3f}")

# ── 4b. SVM su PC1 locali — ablation ──────────────────────────────────────────
# Per ogni sample: PCA locale sui 6 punti → v_i (sign: W>GT).
# SVM con classe +1={v_i} e classe -1={-v_i}: trova la direzione che massimizza
# il margine tra i vettori di drift locali e i loro opposti.
def _local_pc1_vectors(emb_slots, n_samples):
    vecs = []
    for i in range(n_samples):
        pts = np.stack([emb_slots[s][i] for s in range(6)])
        pts_c = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
        v = Vt[0]
        w_proj  = np.mean([pts[s] @ v for s in range(3, 6)])
        gt_proj = np.mean([pts[s] @ v for s in range(3)])
        if w_proj < gt_proj:
            v = -v
        vecs.append(v)
    return np.stack(vecs)   # (N, 768)

pc1_in  = _local_pc1_vectors(emb_in_slots,    N)  # (50, 768)
pc1_out = _local_pc1_vectors(emb_out_slots_all, N)  # (50, 768)

# train set: v_i (+1) e -v_i (-1)
X_pc1_in  = np.vstack([pc1_in,  -pc1_in ])
X_pc1_out = np.vstack([pc1_out, -pc1_out])
y_pc1     = np.array([1] * N + [-1] * N)

svm_pc1_in  = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_pc1_in.fit(X_pc1_in, y_pc1)
w_drift_in_pc1  = sk_normalize(svm_pc1_in.coef_)[0]

svm_pc1_out = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_pc1_out.fit(X_pc1_out, y_pc1)
w_drift_out_pc1 = sk_normalize(svm_pc1_out.coef_)[0]

cos_svm_vs_raw_in  = float(w_drift_in_pc1  @ w_drift_in_svm)
cos_svm_vs_raw_out = float(w_drift_out_pc1 @ w_drift_out_svm)
print(f"SVM-PC1  cos vs SVM-raw — input: {cos_svm_vs_raw_in:.4f}  "
      f"|  output: {cos_svm_vs_raw_out:.4f}")

# Assi produzione = SVM raw (invariato)
w_drift_in  = w_drift_in_svm
w_drift_out = w_drift_out_svm

proj_in  = X_in  @ w_drift_in
proj_out = X_out @ w_drift_out

proj_in_slots  = [proj_in [k * N:(k + 1) * N] for k in range(6)]
proj_out_slots = [proj_out[k * N:(k + 1) * N] for k in range(6)]

def _median_sigma(proj_all):
    pairs = np.abs(proj_all[:, None] - proj_all[None, :])
    return float(np.median(pairs[np.triu_indices(len(proj_all), k=1)]))

sigma_in  = _median_sigma(proj_in)
sigma_out = _median_sigma(proj_out)
print(f"sigma_in={sigma_in:.5f}  sigma_out={sigma_out:.5f}")

anchors_in  = [float(proj_in_slots[s].mean())  for s in range(3)]
anchors_out = [float(proj_out_slots[s].mean()) for s in range(3)]

# ── 5. Core 1D Markov Spettrale (Laplace k=1) ─────────────────────────────────
def _laplacian(pts, sigma):
    diff = pts[:, None] - pts[None, :]
    return np.exp(-np.abs(diff) / sigma)

def _row_stochastic(A):
    s = A.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return A / s

def _spectral_H(pts_in, pts_out, sig_in, sig_out):
    px   = _row_stochastic(_laplacian(pts_in,  sig_in))
    py   = _row_stochastic(_laplacian(pts_out, sig_out))
    eigs = np.abs(np.linalg.eigvals(py @ px))
    total = eigs.sum()
    if total < 1e-12:
        return 0.0
    e = eigs / total
    e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))

# ── 6. Calcolo metodi — SVM-raw (produzione) + SVM-PC1 (ablation) ─────────────

# Metodo 1: Inv-Entropy H_k1 (già caricato)

def _run_online(w_in, w_out, label):
    """Calcola ORACLE e ONLINE dato un asse di drift, ricalcolando sigma e anchors."""
    pi  = X_in  @ w_in
    po  = X_out @ w_out
    pi_s = [pi[k * N:(k + 1) * N] for k in range(6)]
    po_s = [po[k * N:(k + 1) * N] for k in range(6)]
    sig_i = _median_sigma(pi)
    sig_o = _median_sigma(po)
    anc_i = [float(pi_s[s].mean()) for s in range(3)]
    anc_o = [float(po_s[s].mean()) for s in range(3)]
    print(f"  [{label}] sigma_in={sig_i:.5f}  sigma_out={sig_o:.5f}")
    print(f"  [{label}] anchors_in ={[f'{a:.4f}' for a in anc_i]}")
    print(f"  [{label}] anchors_out={[f'{a:.4f}' for a in anc_o]}")
    oracle = np.array([
        _spectral_H(
            np.array([pi_s[s][i] for s in range(6)]),
            np.array([po_s[s][i] for s in range(6)]),
            sig_i, sig_o,
        ) for i in range(N)
    ])
    online = np.array([
        _spectral_H(
            np.array(anc_i + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
            np.array(anc_o + [po_s[3][i], po_s[4][i], po_s[5][i]]),
            sig_i, sig_o,
        ) for i in range(N)
    ])
    return oracle, online, sig_i, sig_o, anc_i, anc_o, pi_s, po_s

# SVM-raw (produzione)
H_oracle_svm, H_online_svm, sigma_in, sigma_out, anchors_in, anchors_out, proj_in_slots, proj_out_slots = \
    _run_online(w_drift_in_svm, w_drift_out_svm, "SVM-raw")

# SVM-PC1 (ablation)
H_oracle_pc1, H_online_pc1, *_ = _run_online(w_drift_in_pc1, w_drift_out_pc1, "SVM-PC1")

# alias per il resto del codice (calibrazione usa SVM-raw = produzione)
H_spectral_oracle = H_oracle_svm
H_spectral_online = H_online_svm

# ── 7. Tabella Pearson ────────────────────────────────────────────────────────
targets = [
    ("vs WER",         wer_arr),
    ("vs E_sem_top",   E_sem_top),
    ("vs E_sem_cross", E_sem_cross),
]

archs = [
    ("Inv-Entropy H_k1 (offline UB)",       H_ie_k1,          "sì"),
    ("1D Markov ORACLE  [SVM-raw]",          H_oracle_svm,     "sì"),
    ("1D Markov ORACLE  [SVM-PC1 ablation]", H_oracle_pc1,     "sì"),
    ("1D Markov ONLINE  [SVM-raw] ★PROD",    H_online_svm,     "no"),
    ("1D Markov ONLINE  [SVM-PC1 ablation]", H_online_pc1,     "no"),
]

W, C = 36, 26
SEP = "─" * (W + 3 + (C + 3) * 3 + 8)
print("\n" + SEP)
print("   BENCHMARK FINALE — 3 METODI × 3 TARGET (N=50)")
print(SEP)
print(f"{'Metodo':<{W}} | {'GT':^4} | "
      f"{'vs WER':^{C}} | {'vs E_sem_top':^{C}} | {'vs E_sem_cross':^{C}}")
print(SEP)

in_online = False
pearson_online_wer = None
for name, H_vals, gt_rt in archs:
    if gt_rt == "no" and not in_online:
        print(SEP)
        in_online = True
    cells = [f"{pearsonr(H_vals, t)[0]:+.4f} (p={pearsonr(H_vals, t)[1]:.2e})"
             for _, t in targets]
    print(f"{name:<{W}} | {gt_rt:^4} | "
          f"{cells[0]:^{C}} | {cells[1]:^{C}} | {cells[2]:^{C}}")
    if gt_rt == "no" and pearson_online_wer is None:
        pearson_online_wer = float(pearsonr(H_vals, wer_arr)[0])

print(SEP + "\n")

# ── 8. Calibrazione — soglie P33/P66 su H_spectral_online ────────────────────
p33 = float(np.percentile(H_spectral_online, 33))
p66 = float(np.percentile(H_spectral_online, 66))
print(f"H_spectral_online — mean={H_spectral_online.mean():.4f}  "
      f"std={H_spectral_online.std():.4f}  "
      f"range=[{H_spectral_online.min():.4f}, {H_spectral_online.max():.4f}]")
print(f"threshold_green (P33) = {p33:.5f}")
print(f"threshold_red   (P66) = {p66:.5f}")

# ── 9. Salva geometry_calibration.json ───────────────────────────────────────
cfg_out = {
    # Asse di drift output (768D) — compatibilità _drift_fig() dashboard
    "w_resp_drift":  w_drift_out.tolist(),
    # Asse di drift input (768D) — nuovo, serve per 1D Markov
    "w_input_drift": w_drift_in.tolist(),
    # Sigma mediana per i due spazi 1D
    "sigma_out":     sigma_out,
    "sigma_in":      sigma_in,
    # Ancoraggi congelati GT (slot 0/1/2)
    "anchors_out":   anchors_out,
    "anchors_in":    anchors_in,
    # Alias per _drift_fig() — mantiene compatibilità con dashboard
    "mu_R_GT1": anchors_out[0],
    "mu_R_GT2": anchors_out[1],
    "mu_R_GT3": anchors_out[2],
    # Soglie di rischio (P33/P66 di H_spectral_online su N=50)
    "threshold_green":    p33,
    "threshold_red":      p66,
    # Limiti empirici per normalizzazione severity score
    "h_spectral_min":     float(H_spectral_online.min()),
    "h_spectral_max":     float(H_spectral_online.max()),
    # Metriche di diagnostica
    "pearson_H_wer":           pearson_online_wer,
    "svm_train_acc_in":        acc_in,
    "svm_train_acc_out":       acc_out,
    "n_samples":               N,
}

CFG_OUT.parent.mkdir(parents=True, exist_ok=True)
CFG_OUT.write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
print(f"\nCalibrazione salvata → {CFG_OUT}")

# ── 10. Salva h_spectral per-sample in dataset_topologico_50.json ─────────────
h_spectral_by_stem = {stem: float(h) for stem, h in zip(stems, H_spectral_online)}

if DATASET_PATH.exists():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    updated = 0
    for record in dataset:
        stem = record.get("stem", "")
        if stem in h_spectral_by_stem:
            record["h_spectral"] = round(h_spectral_by_stem[stem], 6)
            updated += 1
    DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"dataset_topologico_50.json aggiornato — {updated}/{len(dataset)} record con h_spectral")
else:
    print(f"ATTENZIONE: {DATASET_PATH} non trovato, skip aggiornamento dataset.")
