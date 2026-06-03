"""
scripts/ablation_cpca.py
------------------------
Ablation: Contrastive PCA (Abid et al., Nature Commun. 2018) per derivare
l'asse di drift GT→W senza label supervisionate.

Idea:
  Sigma_W  = cov(W embeddings)   — varianza topic + drift dialettale
  Sigma_GT = cov(GT embeddings)  — varianza topic soltanto
  C(alpha) = Sigma_W - alpha * Sigma_GT  →  isola la varianza di drift

  PC1 di C(alpha) = direzione con più varianza in W che in GT.

Sweep alpha in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
Per ogni alpha: ORACLE + ONLINE, Pearson vs WER / E_sem_top / E_sem_cross.

Confronto finale con SVM-raw (produzione).

Note:
  - Covariance matrices rank-deficient: rank ≤ 150 in R^768.
  - Usiamo np.linalg.eigh (symmetric) per stabilità numerica.
  - N=50, SBERT paraphrase-multilingual-mpnet-base-v2, Mistral greedy T=0.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize

BASE    = Path(__file__).parent.parent
LLM_DIR = BASE / "daws" / "results" / "llm_cache"
IE_JSON = BASE / "daws" / "results" / "ie_study_ablation.json"

ALPHAS  = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

# ── 1. Dati ────────────────────────────────────────────────────────────────────
ie_records = json.loads(IE_JSON.read_text(encoding="utf-8"))
stems      = [r["filename"] for r in ie_records]
wer_arr    = np.array([r["wer"] for r in ie_records])
N          = len(stems)
print(f"N={N}")

inputs_by_slot    = [[] for _ in range(6)]
responses_by_slot = [[] for _ in range(6)]
for stem in stems:
    d = json.loads((LLM_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    for slot in range(6):
        inputs_by_slot[slot].append(d["inputs"][slot])
        responses_by_slot[slot].append(d["responses"][slot])

inputs_flat    = [t for slot in range(6) for t in inputs_by_slot[slot]]
responses_flat = [t for slot in range(6) for t in responses_by_slot[slot]]

# ── 2. SBERT ──────────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
X_in  = sbert.encode(inputs_flat,    batch_size=32,
                     show_progress_bar=False, normalize_embeddings=False)
X_out = sbert.encode(responses_flat, batch_size=32,
                     show_progress_bar=False, normalize_embeddings=False)

in_slots  = [X_in [k*N:(k+1)*N] for k in range(6)]
out_slots = [X_out[k*N:(k+1)*N] for k in range(6)]

# ── 3. Target semantici ────────────────────────────────────────────────────────
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n < 1e-12] = 1.0
    return X / n

out_norm    = [_l2(out_slots[s]) for s in range(6)]
E_sem_top   = np.array([1. - float(np.dot(out_norm[3][i], out_norm[0][i]))
                        for i in range(N)])
E_sem_cross = np.array([1. - np.mean([float(np.dot(out_norm[w][i], out_norm[g][i]))
                                      for w in [3,4,5] for g in [0,1,2]])
                        for i in range(N)])

# ── 4. Pipeline 1D Markov Spettrale ───────────────────────────────────────────
def _laplacian(pts, sigma):
    return np.exp(-np.abs(pts[:, None] - pts[None, :]) / sigma)

def _row_stoch(A):
    s = A.sum(axis=1, keepdims=True); s[s < 1e-12] = 1.
    return A / s

def spectral_H(pts_in, pts_out, si, so):
    eigs = np.abs(np.linalg.eigvals(
        _row_stoch(_laplacian(pts_out, so)) @ _row_stoch(_laplacian(pts_in, si))
    ))
    tot = eigs.sum()
    if tot < 1e-12: return 0.
    e = eigs / tot; e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))

def median_sigma(proj):
    pairs = np.abs(proj[:, None] - proj[None, :])
    return float(np.median(pairs[np.triu_indices(len(proj), k=1)]))

def run_pipeline(w_in, w_out):
    """ORACLE + ONLINE con sigma e anchors ricalcolati sull'asse dato."""
    pi  = X_in  @ w_in;  pi_s = [pi[k*N:(k+1)*N] for k in range(6)]
    po  = X_out @ w_out; po_s = [po[k*N:(k+1)*N] for k in range(6)]
    si  = median_sigma(pi);  so  = median_sigma(po)
    ai  = [float(pi_s[s].mean()) for s in range(3)]
    ao  = [float(po_s[s].mean()) for s in range(3)]
    H_oracle = np.array([
        spectral_H(np.array([pi_s[s][i] for s in range(6)]),
                   np.array([po_s[s][i] for s in range(6)]), si, so)
        for i in range(N)])
    H_online = np.array([
        spectral_H(np.array(ai + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
                   np.array(ao + [po_s[3][i], po_s[4][i], po_s[5][i]]), si, so)
        for i in range(N)])
    return H_oracle, H_online

def pearson_row(H):
    return [f"{pearsonr(H, t)[0]:+.4f} (p={pearsonr(H, t)[1]:.2e})"
            for t in [wer_arr, E_sem_top, E_sem_cross]]

# ── 5. Matrici di covarianza GT e W ───────────────────────────────────────────
X_GT_in  = np.vstack([in_slots[s]  for s in range(3)])    # (150, 768)
X_W_in   = np.vstack([in_slots[s]  for s in range(3, 6)]) # (150, 768)
X_GT_out = np.vstack([out_slots[s] for s in range(3)])
X_W_out  = np.vstack([out_slots[s] for s in range(3, 6)])

def cov(X):
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (len(X) - 1)   # (768, 768)

print("Computing covariance matrices (768×768) ...")
Sigma_GT_in  = cov(X_GT_in)
Sigma_W_in   = cov(X_W_in)
Sigma_GT_out = cov(X_GT_out)
Sigma_W_out  = cov(X_W_out)

# diagnostica: varianza totale (traccia)
print(f"  tr(Sigma_GT_in)  = {np.trace(Sigma_GT_in):.4f}  "
      f"tr(Sigma_W_in)  = {np.trace(Sigma_W_in):.4f}")
print(f"  tr(Sigma_GT_out) = {np.trace(Sigma_GT_out):.4f}  "
      f"tr(Sigma_W_out) = {np.trace(Sigma_W_out):.4f}")

# ── 6. Baseline SVM-raw ────────────────────────────────────────────────────────
print("\nComputing SVM-raw baseline ...")
labels = np.array([0]*(3*N) + [1]*(3*N))
svm_i = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_in,  labels)
svm_o = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_out, labels)
w_svm_in  = sk_normalize(svm_i.coef_)[0]
w_svm_out = sk_normalize(svm_o.coef_)[0]
H_or_svm, H_on_svm = run_pipeline(w_svm_in, w_svm_out)

# ── 7. cPCA sweep su alpha ─────────────────────────────────────────────────────
print("\ncPCA sweep ...")

results = []   # (alpha, H_oracle, H_online, w_in, w_out, eig_in, eig_out)

for alpha in ALPHAS:
    C_in  = Sigma_W_in  - alpha * Sigma_GT_in    # (768, 768) symmetric
    C_out = Sigma_W_out - alpha * Sigma_GT_out

    # eigh: eigenvalues in ascending order → largest = [-1]
    evals_in,  evecs_in  = np.linalg.eigh(C_in)
    evals_out, evecs_out = np.linalg.eigh(C_out)

    w_in  = evecs_in [:, -1].copy()   # dominant eigenvector
    w_out = evecs_out[:, -1].copy()

    # sign: W projects più positivo di GT
    if (X_W_in  @ w_in ).mean() < (X_GT_in  @ w_in ).mean(): w_in  = -w_in
    if (X_W_out @ w_out).mean() < (X_GT_out @ w_out).mean(): w_out = -w_out

    H_or, H_on = run_pipeline(w_in, w_out)
    results.append((alpha, H_or, H_on, w_in, w_out,
                    float(evals_in[-1]), float(evals_out[-1])))

# ── 8. Stampa tabella risultati ────────────────────────────────────────────────
W, C = 44, 28
SEP  = "─" * (W + 3 + (C+3)*3 + 2)

print("\n" + "═"*120)
print("cPCA ABLATION — sweep alpha")
print("═"*120)
print(f"\n{'Metodo':<{W}} | {'vs WER':^{C}} | {'vs E_sem_top':^{C}} | {'vs E_sem_cross':^{C}}")
print(SEP)

# SVM-raw reference
cells = pearson_row(H_or_svm)
print(f"{'ORACLE  SVM-raw  [produzione]':<{W}} | {cells[0]:^{C}} | {cells[1]:^{C}} | {cells[2]:^{C}}")
cells = pearson_row(H_on_svm)
print(f"{'ONLINE  SVM-raw  [produzione]':<{W}} | {cells[0]:^{C}} | {cells[1]:^{C}} | {cells[2]:^{C}}")

print(SEP)

for alpha, H_or, H_on, w_in, w_out, eig_in, eig_out in results:
    cos_in  = float(w_in  @ w_svm_in)
    cos_out = float(w_out @ w_svm_out)
    tag = f"cPCA α={alpha:.1f}  eig=[{eig_in:+.4f},{eig_out:+.4f}]  cos={cos_in:.3f}/{cos_out:.3f}"
    cells = pearson_row(H_or)
    print(f"{'ORACLE  '+tag:<{W}} | {cells[0]:^{C}} | {cells[1]:^{C}} | {cells[2]:^{C}}")
    cells = pearson_row(H_on)
    print(f"{'ONLINE  '+tag:<{W}} | {cells[0]:^{C}} | {cells[1]:^{C}} | {cells[2]:^{C}}")
    print()

print(SEP)

# ── 9. Best alpha per ONLINE WER ──────────────────────────────────────────────
best = max(results, key=lambda r: pearsonr(r[2], wer_arr)[0])
best_r = pearsonr(best[2], wer_arr)[0]
print(f"\nBest cPCA ONLINE vs WER: alpha={best[0]}  Pearson={best_r:+.4f}"
      f"  (SVM-raw: {pearsonr(H_on_svm, wer_arr)[0]:+.4f})")
