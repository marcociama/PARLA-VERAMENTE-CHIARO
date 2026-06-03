"""
scripts/ablation_cpca_inventropy.py
------------------------------------
Ablation: cPCA + Inv-Entropy in sottospazio k-D (senza GT a runtime).

Idea:
  1. cPCA trova il sottospazio k-D dove la varianza di drift W > GT.
  2. Tutti gli embedding vengono proiettati in questo sottospazio.
  3. Le similarità coseno vengono calcolate in k-D (non 768D).
  4. Inv-Entropy (formula ufficiale) applicata alle matrici di affinità k-D.
  5. Per la modalità ONLINE: i GT vengono sostituiti con anchor congelati
     (centroidi delle proiezioni GT sui dati di calibrazione).

Sweep:
  alpha ∈ {0.9, 1.0, 1.2, 1.5, 2.0}   — contrasto cPCA
  k     ∈ {1, 2, 5, 10, 20}            — dimensioni del sottospazio

Baselines:
  - Inv-Entropy OFFLINE k=1 (768D, GT reali) — upper bound
  - 1D Markov Spettrale ONLINE SVM-raw ★     — produzione

Usa _row_stochastic dall'implementazione ufficiale (daws/pipeline/inv_entropy.py).
"""

import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from daws.pipeline.inv_entropy import _row_stochastic   # formula ufficiale

LLM_DIR = BASE / "daws" / "results" / "llm_cache"
IE_JSON = BASE / "daws" / "results" / "ie_study_ablation.json"

ALPHAS = [0.9, 1.0, 1.2, 1.5, 2.0]
KS     = [1, 2, 5, 10, 20]

# ── 1. Dati ────────────────────────────────────────────────────────────────────
ie_records = json.loads(IE_JSON.read_text(encoding="utf-8"))
stems      = [r["filename"] for r in ie_records]
wer_in     = np.array([r["wer"]  for r in ie_records])
H_ie_k1    = np.array([r["H_k1"] for r in ie_records])
N          = len(stems)

inputs_by_slot    = [[] for _ in range(6)]
responses_by_slot = [[] for _ in range(6)]
for stem in stems:
    d = json.loads((LLM_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    for slot in range(6):
        inputs_by_slot[slot].append(d["inputs"][slot])
        responses_by_slot[slot].append(d["responses"][slot])

# ── 2. WER_out ─────────────────────────────────────────────────────────────────
def _ed(a, b):
    m, n = len(a), len(b)
    dp = np.arange(n + 1)
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return int(dp[n])

def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    return _ed(r, h) / len(r) if r else (0. if not h else 1.)

wer_out = np.array([wer(responses_by_slot[0][i], responses_by_slot[3][i])
                    for i in range(N)])

# ── 3. SBERT ──────────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
inputs_flat    = [t for s in range(6) for t in inputs_by_slot[s]]
responses_flat = [t for s in range(6) for t in responses_by_slot[s]]
X_in  = sbert.encode(inputs_flat,    batch_size=32,
                     show_progress_bar=False, normalize_embeddings=False)
X_out = sbert.encode(responses_flat, batch_size=32,
                     show_progress_bar=False, normalize_embeddings=False)

in_slots  = [X_in [k*N:(k+1)*N] for k in range(6)]
out_slots = [X_out[k*N:(k+1)*N] for k in range(6)]

# ── 4. Target semantici ────────────────────────────────────────────────────────
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n < 1e-12] = 1.
    return X / n

out_norm    = [_l2(out_slots[s]) for s in range(6)]
E_sem_top   = np.array([1. - float(np.dot(out_norm[3][i], out_norm[0][i]))
                        for i in range(N)])
E_sem_cross = np.array([1. - np.mean([float(np.dot(out_norm[w][i], out_norm[g][i]))
                                      for w in [3,4,5] for g in [0,1,2]])
                        for i in range(N)])

TARGETS = [("WER_in", wer_in), ("WER_out", wer_out),
           ("E_sem_top", E_sem_top), ("E_sem_cross", E_sem_cross)]

# ── 5. Inv-Entropy core (su embedding pre-proiettati) ─────────────────────────
def sim_matrix_kd(embs: np.ndarray, k_exp: int = 4) -> np.ndarray:
    """Cosine similarity in k-D, raised to k_exp, clipped [0,1]."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.
    en = embs / norms
    return np.clip(en @ en.T, 0., 1.) ** k_exp

def inv_entropy_H(embs_in: np.ndarray, embs_out: np.ndarray) -> float:
    """Inv-Entropy (Eq.5 Song NeurIPS 2025) su embedding k-D."""
    Px = _row_stochastic(sim_matrix_kd(embs_in))
    Py = _row_stochastic(sim_matrix_kd(embs_out))
    d  = np.diag(Py @ Px)
    return float(-np.sum(d * np.log(d + 1e-12)))

# ── 6. cPCA matrici di covarianza ─────────────────────────────────────────────
print("Computing covariance matrices ...")
def cov(X):
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (len(X) - 1)

X_GT_in  = np.vstack([in_slots[s]  for s in range(3)])
X_W_in   = np.vstack([in_slots[s]  for s in range(3, 6)])
X_GT_out = np.vstack([out_slots[s] for s in range(3)])
X_W_out  = np.vstack([out_slots[s] for s in range(3, 6)])

S_GT_in, S_W_in   = cov(X_GT_in),  cov(X_W_in)
S_GT_out, S_W_out = cov(X_GT_out), cov(X_W_out)

# ── 7. Baseline 1: Inv-Entropy OFFLINE 768D (GT reali) ────────────────────────
print("Computing Inv-Entropy OFFLINE 768D baseline ...")
H_ie_offline = []
for i in range(N):
    ei = np.stack([in_slots[s][i]  for s in range(6)])
    eo = np.stack([out_slots[s][i] for s in range(6)])
    H_ie_offline.append(inv_entropy_H(ei, eo))
H_ie_offline = np.array(H_ie_offline)

# ── 8. Baseline 2: 1D Markov Spettrale ONLINE SVM-raw ─────────────────────────
print("Computing 1D Markov SVM-raw baseline ...")
labels = np.array([0]*(3*N) + [1]*(3*N))
svm_i  = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_in,  labels)
svm_o  = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_out, labels)
w_in_svm  = sk_normalize(svm_i.coef_)[0]
w_out_svm = sk_normalize(svm_o.coef_)[0]

def markov_H(pi, po, si, so):
    lp = lambda x, s: np.exp(-np.abs(x[:,None]-x[None,:])/s)
    rs = lambda A: A / np.maximum(A.sum(1,keepdims=True), 1e-12)
    eigs = np.abs(np.linalg.eigvals(rs(lp(po,so)) @ rs(lp(pi,si))))
    tot  = eigs.sum(); e = eigs/tot; e = e[e>1e-12]
    return float(-np.sum(e*np.log(e))) if tot > 1e-12 else 0.

def med_sig(x):
    p = np.abs(x[:,None]-x[None,:]); return float(np.median(p[np.triu_indices(len(x),k=1)]))

pi_all = X_in  @ w_in_svm;  pi_s = [pi_all[k*N:(k+1)*N] for k in range(6)]
po_all = X_out @ w_out_svm; po_s = [po_all[k*N:(k+1)*N] for k in range(6)]
si_svm, so_svm = med_sig(pi_all), med_sig(po_all)
ai = [float(pi_s[s].mean()) for s in range(3)]
ao = [float(po_s[s].mean()) for s in range(3)]

H_markov_online = np.array([
    markov_H(np.array(ai + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
             np.array(ao + [po_s[3][i], po_s[4][i], po_s[5][i]]),
             si_svm, so_svm) for i in range(N)])

# ── 9. cPCA + Inv-Entropy sweep ────────────────────────────────────────────────
print(f"Running cPCA+InvEntropy sweep ({len(ALPHAS)} alphas × {len(KS)} k values) ...")

sweep_results = {}   # (alpha, k) → (H_oracle, H_online)

for alpha in ALPHAS:
    # Contrastive matrix e sue proiezioni
    C_in  = S_W_in  - alpha * S_GT_in
    C_out = S_W_out - alpha * S_GT_out
    evals_in,  evecs_in  = np.linalg.eigh(C_in)   # ascending
    evals_out, evecs_out = np.linalg.eigh(C_out)

    for k in KS:
        # Top-k autovettori (ultimi k nella lista ascendente)
        P_in  = evecs_in [:, -k:]   # (768, k)
        P_out = evecs_out[:, -k:]   # (768, k)

        # Proiezioni k-D di tutti i 300 punti
        XI_proj = X_in  @ P_in    # (300, k)
        XO_proj = X_out @ P_out   # (300, k)

        in_p  = [XI_proj[s*N:(s+1)*N] for s in range(6)]   # (N, k) per slot
        out_p = [XO_proj[s*N:(s+1)*N] for s in range(6)]

        # Anchor GT congelati (media dei 3 slot GT nel sottospazio)
        anc_in  = [in_p[s].mean(axis=0)  for s in range(3)]   # 3 × k-D
        anc_out = [out_p[s].mean(axis=0) for s in range(3)]

        # ORACLE: usa proiezioni GT reali per-sample
        H_oracle = np.array([
            inv_entropy_H(
                np.stack([in_p[s][i]  for s in range(6)]),
                np.stack([out_p[s][i] for s in range(6)]),
            ) for i in range(N)
        ])

        # ONLINE: sostituisce GT con anchor congelati
        H_online = np.array([
            inv_entropy_H(
                np.stack(anc_in  + [in_p[s][i]  for s in range(3, 6)]),
                np.stack(anc_out + [out_p[s][i] for s in range(3, 6)]),
            ) for i in range(N)
        ])

        sweep_results[(alpha, k)] = (H_oracle, H_online)

# ── 10. Stampa risultati ──────────────────────────────────────────────────────
def prow(H):
    return [f"{pearsonr(H,t)[0]:+.4f}(p={pearsonr(H,t)[1]:.1e})" for _,t in TARGETS]

NM, NC = 36, 22
SEP = "─" * (NM + 8 + (NC+3)*len(TARGETS))

header = (f"{'Metodo':<{NM}} {'Modo':^7} | "
          + " | ".join(f"{n:^{NC}}" for n,_ in TARGETS))

print("\n" + "═"*len(header))
print("cPCA + Inv-Entropy  vs  baseline OFFLINE e 1D Markov ONLINE")
print("═"*len(header))
print(header)
print(SEP)

# Baselines
for H, label, mode in [
    (H_ie_offline,    "InvEnt OFFLINE 768D (GT reali)", "ORACLE"),
    (H_ie_offline,    "InvEnt OFFLINE 768D (GT reali)", "—"),
    (H_markov_online, "1D Markov SVM-raw ★ PROD",       "ONLINE"),
]:
    r = prow(H)
    print(f"{label:<{NM}} {mode:^7} | " + " | ".join(f"{c:^{NC}}" for c in r))
print(SEP)

# Sweep
for alpha in ALPHAS:
    first_alpha = True
    for k in KS:
        H_or, H_on = sweep_results[(alpha, k)]
        tag = f"cPCA α={alpha} k={k:>2}"
        for H, mode in [(H_or, "ORACLE"), (H_on, "ONLINE")]:
            r = prow(H)
            print(f"{tag:<{NM}} {mode:^7} | " + " | ".join(f"{c:^{NC}}" for c in r))
        print()
    print(SEP)

# ── 11. Best ONLINE per ogni target ───────────────────────────────────────────
print("\nBEST cPCA+InvEnt ONLINE per target:")
for tname, tvec in TARGETS:
    best_key  = max(sweep_results, key=lambda k: pearsonr(sweep_results[k][1], tvec)[0])
    best_H    = sweep_results[best_key][1]
    best_r    = pearsonr(best_H, tvec)[0]
    base_r    = pearsonr(H_markov_online, tvec)[0]
    print(f"  {tname:<14}: α={best_key[0]}  k={best_key[1]:>2}  "
          f"Pearson={best_r:+.4f}  (SVM-raw ONLINE: {base_r:+.4f})")
