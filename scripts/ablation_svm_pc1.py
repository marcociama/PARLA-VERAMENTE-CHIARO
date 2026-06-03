"""
scripts/ablation_svm_pc1.py
---------------------------
Ablation isolata: SVM su direzioni PC1 locali vs SVM su embedding raw.

Per ogni asse testato il calcolo è completamente indipendente:
  1. Asse w_in, w_out  (input e output separati)
  2. sigma_in, sigma_out  ricalcolati su quell'asse
  3. anchors_in, anchors_out = media proiezioni GT (slot 0,1,2) su quell'asse
  4. ORACLE: usa proiezioni GT reali per-sample
  5. ONLINE:  usa anchors congelati + proiezioni W live (slot 3,4,5)

N=50, SBERT paraphrase-multilingual-mpnet-base-v2, Mistral greedy T=0
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize

BASE     = Path(__file__).parent.parent
LLM_DIR  = BASE / "daws" / "results" / "llm_cache"
IE_JSON  = BASE / "daws" / "results" / "ie_study_ablation.json"

# ── 1. Carica dati ─────────────────────────────────────────────────────────────
ie_records = json.loads(IE_JSON.read_text(encoding="utf-8"))
stems      = [r["filename"] for r in ie_records]
wer_arr    = np.array([r["wer"] for r in ie_records])
N          = len(stems)
print(f"Campioni: N={N}")

inputs_by_slot    = [[] for _ in range(6)]
responses_by_slot = [[] for _ in range(6)]
for stem in stems:
    p = LLM_DIR / f"{stem}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    for slot in range(6):
        inputs_by_slot[slot].append(d["inputs"][slot])
        responses_by_slot[slot].append(d["responses"][slot])

inputs_flat    = [t for slot in range(6) for t in inputs_by_slot[slot]]
responses_flat = [t for slot in range(6) for t in responses_by_slot[slot]]

# ── 2. SBERT encode ────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
print("Encoding inputs ...")
X_in  = sbert.encode(inputs_flat,    batch_size=32, show_progress_bar=False,
                     normalize_embeddings=False)
print("Encoding outputs ...")
X_out = sbert.encode(responses_flat, batch_size=32, show_progress_bar=False,
                     normalize_embeddings=False)

# slot views  shape (N, 768)
in_slots  = [X_in [k*N:(k+1)*N] for k in range(6)]   # 0-2=GT, 3-5=W
out_slots = [X_out[k*N:(k+1)*N] for k in range(6)]

# ── 3. Target E_sem ────────────────────────────────────────────────────────────
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n

out_norm = [_l2(out_slots[s]) for s in range(6)]
E_sem_top   = np.array([1.0 - float(np.dot(out_norm[3][i], out_norm[0][i]))
                        for i in range(N)])
E_sem_cross = np.array([1.0 - np.mean([float(np.dot(out_norm[w][i], out_norm[g][i]))
                                       for w in [3,4,5] for g in [0,1,2]])
                        for i in range(N)])

# ── 4. Utilità 1D Markov Spettrale ────────────────────────────────────────────
def _laplacian(pts, sigma):
    diff = pts[:, None] - pts[None, :]
    return np.exp(-np.abs(diff) / sigma)

def _row_stoch(A):
    s = A.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return A / s

def spectral_H(pts_in, pts_out, sig_in, sig_out):
    px   = _row_stoch(_laplacian(pts_in,  sig_in))
    py   = _row_stoch(_laplacian(pts_out, sig_out))
    eigs = np.abs(np.linalg.eigvals(py @ px))
    tot  = eigs.sum()
    if tot < 1e-12:
        return 0.0
    e = eigs / tot
    e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))

def median_sigma(proj_all):
    pairs = np.abs(proj_all[:, None] - proj_all[None, :])
    return float(np.median(pairs[np.triu_indices(len(proj_all), k=1)]))

# ── 5. Funzione che calcola ORACLE + ONLINE dato un asse (w_in, w_out) ─────────
def run_pipeline(w_in, w_out, name):
    # proiezioni 1D di tutti i 300 punti
    pi  = X_in  @ w_in    # (300,)
    po  = X_out @ w_out   # (300,)

    # split per slot
    pi_s = [pi[k*N:(k+1)*N] for k in range(6)]   # lista di 6 array (50,)
    po_s = [po[k*N:(k+1)*N] for k in range(6)]

    # sigma = mediana distanze assolute su tutti i 300 punti
    sig_i = median_sigma(pi)
    sig_o = median_sigma(po)

    # anchors = MEDIA delle proiezioni GT (slot 0, 1, 2) — separatamente per input e output
    anc_i = [float(pi_s[s].mean()) for s in range(3)]   # [mu_GT1_in, mu_GT2_in, mu_GT3_in]
    anc_o = [float(po_s[s].mean()) for s in range(3)]   # [mu_GT1_out, mu_GT2_out, mu_GT3_out]

    print(f"\n── {name} ──────────────────────────────────────────────")
    print(f"  sigma_in ={sig_i:.5f}   sigma_out={sig_o:.5f}")
    print(f"  anchors_in  (GT mu slot 0/1/2): {[f'{a:+.4f}' for a in anc_i]}")
    print(f"  anchors_out (GT mu slot 0/1/2): {[f'{a:+.4f}' for a in anc_o]}")
    print(f"  GT  in  proj range: [{np.concatenate([pi_s[s] for s in range(3)]).min():.4f}, "
          f"{np.concatenate([pi_s[s] for s in range(3)]).max():.4f}]")
    print(f"  W   in  proj range: [{np.concatenate([pi_s[s] for s in range(3,6)]).min():.4f}, "
          f"{np.concatenate([pi_s[s] for s in range(3,6)]).max():.4f}]")
    print(f"  GT  out proj range: [{np.concatenate([po_s[s] for s in range(3)]).min():.4f}, "
          f"{np.concatenate([po_s[s] for s in range(3)]).max():.4f}]")
    print(f"  W   out proj range: [{np.concatenate([po_s[s] for s in range(3,6)]).min():.4f}, "
          f"{np.concatenate([po_s[s] for s in range(3,6)]).max():.4f}]")

    # ORACLE: usa le proiezioni GT reali per-sample (slot 0,1,2 reali)
    H_oracle = np.array([
        spectral_H(
            np.array([pi_s[s][i] for s in range(6)]),
            np.array([po_s[s][i] for s in range(6)]),
            sig_i, sig_o,
        ) for i in range(N)
    ])

    # ONLINE: sostituisce GT con anchors congelati, usa W live (slot 3,4,5)
    H_online = np.array([
        spectral_H(
            np.array(anc_i + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
            np.array(anc_o + [po_s[3][i], po_s[4][i], po_s[5][i]]),
            sig_i, sig_o,
        ) for i in range(N)
    ])

    targets = [("WER", wer_arr), ("E_sem_top", E_sem_top), ("E_sem_cross", E_sem_cross)]
    print(f"\n  {'Modo':<8} | {'vs WER':^26} | {'vs E_sem_top':^26} | {'vs E_sem_cross':^26}")
    print(f"  {'─'*8}-+-{'─'*26}-+-{'─'*26}-+-{'─'*26}")
    for H, label in [(H_oracle, "ORACLE"), (H_online, "ONLINE")]:
        cells = [f"{pearsonr(H, t)[0]:+.4f} (p={pearsonr(H, t)[1]:.2e})"
                 for _, t in targets]
        print(f"  {label:<8} | {cells[0]:^26} | {cells[1]:^26} | {cells[2]:^26}")

    return H_oracle, H_online

# ── 6. Asse A: SVM raw su embedding (GT=0, W=1) ────────────────────────────────
print("\n" + "═"*70)
print("ASSE A — LinearSVC su embedding raw (GT=0, W=1)  [produzione]")
print("═"*70)

labels_raw = np.array([0]*(3*N) + [1]*(3*N))

svm_in_raw = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_in_raw.fit(X_in, labels_raw)
w_in_raw  = sk_normalize(svm_in_raw.coef_)[0]
acc_in    = float((svm_in_raw.predict(X_in) == labels_raw).mean())

svm_out_raw = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_out_raw.fit(X_out, labels_raw)
w_out_raw  = sk_normalize(svm_out_raw.coef_)[0]
acc_out    = float((svm_out_raw.predict(X_out) == labels_raw).mean())

print(f"SVM-raw train acc — in: {acc_in:.3f}  out: {acc_out:.3f}")
H_oracle_raw, H_online_raw = run_pipeline(w_in_raw, w_out_raw, "SVM-raw")

# ── 7. Asse B: SVM su PC1 locali ───────────────────────────────────────────────
print("\n" + "═"*70)
print("ASSE B — LinearSVC su PC1 locali (+v_i / -v_i)  [ablation]")
print("═"*70)

def local_pc1_vecs(slots, n):
    """PC1 locale per ogni sample dai 6 embedding (sign: W>GT)."""
    vecs = []
    for i in range(n):
        pts   = np.stack([slots[s][i] for s in range(6)])
        pts_c = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
        v = Vt[0]
        if np.mean([pts[s] @ v for s in range(3,6)]) < np.mean([pts[s] @ v for s in range(3)]):
            v = -v
        vecs.append(v)
    return np.stack(vecs)   # (N, 768)

pc1_in  = local_pc1_vecs(in_slots,  N)
pc1_out = local_pc1_vecs(out_slots, N)

cos_in_self  = float(np.mean([pc1_in[i]  @ pc1_in[j]
                               for i in range(N) for j in range(i+1, N)]))
cos_out_self = float(np.mean([pc1_out[i] @ pc1_out[j]
                               for i in range(N) for j in range(i+1, N)]))
print(f"Mutual cosine tra PC1 locali — in: {cos_in_self:.4f}  out: {cos_out_self:.4f}")

# dataset per SVM: +1 = v_i,  -1 = -v_i
X_svm_in  = np.vstack([pc1_in,  -pc1_in ])
X_svm_out = np.vstack([pc1_out, -pc1_out])
y_svm     = np.array([1]*N + [-1]*N)

svm_in_pc1 = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_in_pc1.fit(X_svm_in, y_svm)
w_in_pc1  = sk_normalize(svm_in_pc1.coef_)[0]

svm_out_pc1 = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_out_pc1.fit(X_svm_out, y_svm)
w_out_pc1  = sk_normalize(svm_out_pc1.coef_)[0]

cos_in  = float(w_in_pc1  @ w_in_raw)
cos_out = float(w_out_pc1 @ w_out_raw)
print(f"Coseno SVM-PC1 vs SVM-raw — in: {cos_in:.4f}  out: {cos_out:.4f}")

H_oracle_pc1, H_online_pc1 = run_pipeline(w_in_pc1, w_out_pc1, "SVM-PC1")

# ── 8. Asse C: Regressione lineare PC1 ────────────────────────────────────────
# Target y[k*N + i] = X[k*N + i] @ pc1[i]  (proiezione dell'embedding grezzo
# sulla PC1 locale del sample i). La regressione trova il w globale che meglio
# approssima queste proiezioni locali con un'unica direzione lineare.
print("\n" + "═"*70)
print("ASSE C — Regressione lineare su target PC1 locali  [ablation]")
print("═"*70)

# costruisce il vettore target nello stesso ordine di X_in / X_out
# riga k*N + i → embedding del slot k, sample i → target = emb @ pc1[i]
y_in_reg  = np.array([X_in [k*N + i] @ pc1_in [i]
                       for k in range(6) for i in range(N)])
y_out_reg = np.array([X_out[k*N + i] @ pc1_out[i]
                       for k in range(6) for i in range(N)])

# lstsq: X @ w ≈ y  →  w = pinv(X) y
w_in_reg,  _, _, _ = np.linalg.lstsq(X_in,  y_in_reg,  rcond=None)
w_out_reg, _, _, _ = np.linalg.lstsq(X_out, y_out_reg, rcond=None)
w_in_reg  = w_in_reg  / np.linalg.norm(w_in_reg)
w_out_reg = w_out_reg / np.linalg.norm(w_out_reg)

r2_in  = float(1 - np.var(X_in  @ w_in_reg  - y_in_reg)  / np.var(y_in_reg))
r2_out = float(1 - np.var(X_out @ w_out_reg - y_out_reg) / np.var(y_out_reg))
print(f"R² regressione — in: {r2_in:.4f}  out: {r2_out:.4f}")
print(f"Coseno Reg vs SVM-raw — in: {float(w_in_reg @ w_in_raw):.4f}  "
      f"out: {float(w_out_reg @ w_out_raw):.4f}")
print(f"Coseno Reg vs SVM-PC1 — in: {float(w_in_reg @ w_in_pc1):.4f}  "
      f"out: {float(w_out_reg @ w_out_pc1):.4f}")

H_oracle_reg, H_online_reg = run_pipeline(w_in_reg, w_out_reg, "Reg-PC1")

# ── 9. Asse D: direzione centroide GT→W normalizzata ─────────────────────────
# Per ogni sample i: v_i = normalize(mean(W_embs) - mean(GT_embs))
# Differente dalla PC1 locale: non influenzata dalla varianza intra-cluster.
# Differente dal displacement SVD precedente: normalizza prima di aggregare
# (i sample a basso drift non vengono sovrastati da quelli ad alto drift).
print("\n" + "═"*70)
print("ASSE D — direzione centroide GT→W normalizzata  [ablation]")
print("═"*70)

def centroid_diff_vecs(slots, n):
    """Normalizza (mean_W - mean_GT) per ogni sample."""
    vecs = []
    for i in range(n):
        c_gt = np.mean([slots[s][i] for s in range(3)],   axis=0)
        c_w  = np.mean([slots[s][i] for s in range(3, 6)], axis=0)
        d = c_w - c_gt
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            d = np.zeros_like(d)
        else:
            d = d / norm
        vecs.append(d)
    return np.stack(vecs)   # (N, 768)

cd_in  = centroid_diff_vecs(in_slots,  N)
cd_out = centroid_diff_vecs(out_slots, N)

cos_cd_in  = float(np.mean([cd_in[i]  @ cd_in[j]
                              for i in range(N) for j in range(i+1, N)]))
cos_cd_out = float(np.mean([cd_out[i] @ cd_out[j]
                              for i in range(N) for j in range(i+1, N)]))
print(f"Mutual cosine centroid-diff — in: {cos_cd_in:.4f}  out: {cos_cd_out:.4f}")
print(f"  (confronto PC1 locale   — in: {cos_in_self:.4f}  out: {cos_out_self:.4f})")

# D1: media sferica
w_in_cd_mean  = cd_in.mean(axis=0);  w_in_cd_mean  /= np.linalg.norm(w_in_cd_mean)
w_out_cd_mean = cd_out.mean(axis=0); w_out_cd_mean /= np.linalg.norm(w_out_cd_mean)
print(f"\nD1 — media sferica")
print(f"  cos vs SVM-raw — in: {float(w_in_cd_mean @ w_in_raw):.4f}  "
      f"out: {float(w_out_cd_mean @ w_out_raw):.4f}")
H_oracle_cd_mean, H_online_cd_mean = run_pipeline(w_in_cd_mean, w_out_cd_mean, "CD-mean")

# D2: Grassmanniana (SVD di V, dom. eigenvec di V^T V)
_, _, Vt_cd_in  = np.linalg.svd(cd_in,  full_matrices=False)
_, _, Vt_cd_out = np.linalg.svd(cd_out, full_matrices=False)
w_in_cd_svd  = Vt_cd_in[0]
w_out_cd_svd = Vt_cd_out[0]
# convezione segno: media centroid-diff proietta positiva
if float(cd_in.mean(axis=0)  @ w_in_cd_svd)  < 0: w_in_cd_svd  = -w_in_cd_svd
if float(cd_out.mean(axis=0) @ w_out_cd_svd) < 0: w_out_cd_svd = -w_out_cd_svd
print(f"\nD2 — Grassmanniana (SVD)")
print(f"  cos vs SVM-raw — in: {float(w_in_cd_svd @ w_in_raw):.4f}  "
      f"out: {float(w_out_cd_svd @ w_out_raw):.4f}")
H_oracle_cd_svd, H_online_cd_svd = run_pipeline(w_in_cd_svd, w_out_cd_svd, "CD-svd")

# D3: SVM su {v_i: +1} ∪ {-v_i: -1}
X_cd_in  = np.vstack([cd_in,  -cd_in ])
X_cd_out = np.vstack([cd_out, -cd_out])
svm_cd_in  = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_cd_in.fit(X_cd_in, y_svm)
w_in_cd_svm  = sk_normalize(svm_cd_in.coef_)[0]
svm_cd_out = LinearSVC(C=1.0, max_iter=10_000, dual=True)
svm_cd_out.fit(X_cd_out, y_svm)
w_out_cd_svm = sk_normalize(svm_cd_out.coef_)[0]
print(f"\nD3 — SVM su centroid-diff")
print(f"  cos vs SVM-raw — in: {float(w_in_cd_svm @ w_in_raw):.4f}  "
      f"out: {float(w_out_cd_svm @ w_out_raw):.4f}")
H_oracle_cd_svm, H_online_cd_svm = run_pipeline(w_in_cd_svm, w_out_cd_svm, "CD-svm")

# ── 10. Riepilogo comparativo ──────────────────────────────────────────────────
print("\n" + "═"*70)
print("RIEPILOGO COMPARATIVO")
print("═"*70)
W, C = 38, 28
SEP = "─" * (W + 3 + (C+3)*3 + 2)
print(SEP)
print(f"{'Metodo':<{W}} | {'vs WER':^{C}} | {'vs E_sem_top':^{C}} | {'vs E_sem_cross':^{C}}")
print(SEP)
for H, label in [
    (H_oracle_raw,     "ORACLE  SVM-raw"),
    (H_oracle_pc1,     "ORACLE  SVM-PC1"),
    (H_oracle_reg,     "ORACLE  Reg-PC1"),
    (H_oracle_cd_mean, "ORACLE  CD-mean"),
    (H_oracle_cd_svd,  "ORACLE  CD-svd"),
    (H_oracle_cd_svm,  "ORACLE  CD-svm"),
    (H_online_raw,     "ONLINE  SVM-raw  ★PROD"),
    (H_online_pc1,     "ONLINE  SVM-PC1"),
    (H_online_reg,     "ONLINE  Reg-PC1"),
    (H_online_cd_mean, "ONLINE  CD-mean"),
    (H_online_cd_svd,  "ONLINE  CD-svd"),
    (H_online_cd_svm,  "ONLINE  CD-svm"),
]:
    cells = [f"{pearsonr(H, t)[0]:+.4f} (p={pearsonr(H, t)[1]:.2e})"
             for t in [wer_arr, E_sem_top, E_sem_cross]]
    print(f"{label:<{W}} | {cells[0]:^{C}} | {cells[1]:^{C}} | {cells[2]:^{C}}")
print(SEP)
