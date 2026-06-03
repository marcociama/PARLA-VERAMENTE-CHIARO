"""
scripts/ablation_enriched.py
-----------------------------
Ablation arricchita: tutti i metodi testati × 4 target.

Target:
  WER_in    — WER(WhisperX W1, GT)           errore ASR input
  WER_out   — WER(R_W1, R_GT1)              errore clinico output (LLM, T=0)
  E_sem_top — 1 - cos_sim(R_W1, R_GT1)      divergenza semantica top-beam
  E_sem_cross — 1 - mean(cos_sim) 9 coppie  divergenza semantica media

Metodi:
  A  SVM-raw       (produzione)
  B  SVM-PC1
  C  Reg-PC1
  D1 CD-mean       (centroid-diff, media sferica)
  D3 CD-svm        (centroid-diff, SVM)
  E  cPCA α=1.0
  E  cPCA α=2.0
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

BASE    = Path(__file__).parent.parent
LLM_DIR = BASE / "daws" / "results" / "llm_cache"
IE_JSON = BASE / "daws" / "results" / "ie_study_ablation.json"

# ── 1. Dati ────────────────────────────────────────────────────────────────────
ie_records = json.loads(IE_JSON.read_text(encoding="utf-8"))
stems      = [r["filename"] for r in ie_records]
wer_in     = np.array([r["wer"] for r in ie_records])
N          = len(stems)

inputs_by_slot    = [[] for _ in range(6)]
responses_by_slot = [[] for _ in range(6)]
for stem in stems:
    d = json.loads((LLM_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    for slot in range(6):
        inputs_by_slot[slot].append(d["inputs"][slot])
        responses_by_slot[slot].append(d["responses"][slot])

# ── 2. WER_out = WER(R_W1, R_GT1) ─────────────────────────────────────────────
def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = np.arange(n + 1)
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return int(dp[n])

def wer(ref: str, hyp: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)

wer_out = np.array([
    wer(responses_by_slot[0][i], responses_by_slot[3][i])
    for i in range(N)
])

_smooth = SmoothingFunction().method1
def bleu_err(ref: str, hyp: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r or not h:
        return 1.0
    return 1.0 - sentence_bleu([r], h, smoothing_function=_smooth)

bleu_out = np.array([
    bleu_err(responses_by_slot[0][i], responses_by_slot[3][i])
    for i in range(N)
])

print(f"N={N}  WER_in  mean={wer_in.mean():.4f}  range=[{wer_in.min():.3f},{wer_in.max():.3f}]")
print(f"       WER_out mean={wer_out.mean():.4f}  range=[{wer_out.min():.3f},{wer_out.max():.3f}]")
print(f"       BLEU_out(err) mean={bleu_out.mean():.4f}  range=[{bleu_out.min():.3f},{bleu_out.max():.3f}]")
print(f"       Pearson(WER_in, WER_out) = {pearsonr(wer_in, wer_out)[0]:+.4f}")
print(f"       Pearson(WER_out, BLEU_out) = {pearsonr(wer_out, bleu_out)[0]:+.4f}")

# ── 3. SBERT ──────────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
inputs_flat    = [t for slot in range(6) for t in inputs_by_slot[slot]]
responses_flat = [t for slot in range(6) for t in responses_by_slot[slot]]
X_in  = sbert.encode(inputs_flat,    batch_size=32,
                     show_progress_bar=False, normalize_embeddings=False)
X_out = sbert.encode(responses_flat, batch_size=32,
                     show_progress_bar=False, normalize_embeddings=False)

in_slots  = [X_in [k*N:(k+1)*N] for k in range(6)]
out_slots = [X_out[k*N:(k+1)*N] for k in range(6)]

# ── 4. E_sem ──────────────────────────────────────────────────────────────────
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n < 1e-12] = 1.
    return X / n

out_norm    = [_l2(out_slots[s]) for s in range(6)]
E_sem_top   = np.array([1. - float(np.dot(out_norm[3][i], out_norm[0][i]))
                        for i in range(N)])
E_sem_cross = np.array([1. - np.mean([float(np.dot(out_norm[w][i], out_norm[g][i]))
                                      for w in [3,4,5] for g in [0,1,2]])
                        for i in range(N)])

TARGETS = [("WER_in",     wer_in),
           ("WER_out",    wer_out),
           ("BLEU_out",   bleu_out),
           ("E_sem_top",  E_sem_top),
           ("E_sem_cross",E_sem_cross)]

print(f"       Pearson(WER_in, E_sem_top)  = {pearsonr(wer_in,  E_sem_top)[0]:+.4f}")
print(f"       Pearson(WER_out,E_sem_top)  = {pearsonr(wer_out, E_sem_top)[0]:+.4f}")

# ── 5. Pipeline 1D Markov ─────────────────────────────────────────────────────
def _laplacian(pts, sigma):
    return np.exp(-np.abs(pts[:, None] - pts[None, :]) / sigma)

def _row_stoch(A):
    s = A.sum(axis=1, keepdims=True); s[s < 1e-12] = 1.
    return A / s

def spectral_H(pi, po, si, so):
    eigs = np.abs(np.linalg.eigvals(
        _row_stoch(_laplacian(po, so)) @ _row_stoch(_laplacian(pi, si))
    ))
    tot = eigs.sum()
    if tot < 1e-12: return 0.
    e = eigs / tot; e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))

def median_sigma(proj):
    p = np.abs(proj[:, None] - proj[None, :])
    return float(np.median(p[np.triu_indices(len(proj), k=1)]))

def run_pipeline(w_in, w_out):
    pi   = X_in  @ w_in;  pi_s = [pi[k*N:(k+1)*N] for k in range(6)]
    po   = X_out @ w_out; po_s = [po[k*N:(k+1)*N] for k in range(6)]
    si   = median_sigma(pi);  so = median_sigma(po)
    ai   = [float(pi_s[s].mean()) for s in range(3)]
    ao   = [float(po_s[s].mean()) for s in range(3)]
    H_or = np.array([spectral_H(
        np.array([pi_s[s][i] for s in range(6)]),
        np.array([po_s[s][i] for s in range(6)]), si, so) for i in range(N)])
    H_on = np.array([spectral_H(
        np.array(ai + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
        np.array(ao + [po_s[3][i], po_s[4][i], po_s[5][i]]), si, so) for i in range(N)])
    return H_or, H_on

# ── 6. Calcolo assi ────────────────────────────────────────────────────────────
print("Computing axes ...")
labels = np.array([0]*(3*N) + [1]*(3*N))
y_svm  = np.array([1]*N + [-1]*N)

# A — SVM-raw
svm_i = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_in,  labels)
svm_o = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_out, labels)
w_svm_in  = sk_normalize(svm_i.coef_)[0]
w_svm_out = sk_normalize(svm_o.coef_)[0]

# B — SVM-PC1
def local_pc1_vecs(slots, n):
    vecs = []
    for i in range(n):
        pts = np.stack([slots[s][i] for s in range(6)])
        pts_c = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
        v = Vt[0]
        if np.mean([pts[s] @ v for s in range(3,6)]) < np.mean([pts[s] @ v for s in range(3)]):
            v = -v
        vecs.append(v)
    return np.stack(vecs)

pc1_in  = local_pc1_vecs(in_slots,  N)
pc1_out = local_pc1_vecs(out_slots, N)
svm_pc1_i = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(
    np.vstack([pc1_in,  -pc1_in ]), y_svm)
svm_pc1_o = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(
    np.vstack([pc1_out, -pc1_out]), y_svm)
w_pc1_in  = sk_normalize(svm_pc1_i.coef_)[0]
w_pc1_out = sk_normalize(svm_pc1_o.coef_)[0]

# C — Reg-PC1
y_reg_in  = np.array([X_in [k*N+i] @ pc1_in [i] for k in range(6) for i in range(N)])
y_reg_out = np.array([X_out[k*N+i] @ pc1_out[i] for k in range(6) for i in range(N)])
w_reg_in,  *_ = np.linalg.lstsq(X_in,  y_reg_in,  rcond=None)
w_reg_out, *_ = np.linalg.lstsq(X_out, y_reg_out, rcond=None)
w_reg_in  /= np.linalg.norm(w_reg_in);  w_reg_out /= np.linalg.norm(w_reg_out)

# D — centroid-diff
def cd_vecs(slots, n):
    vecs = []
    for i in range(n):
        d = np.mean([slots[s][i] for s in range(3,6)], axis=0) \
          - np.mean([slots[s][i] for s in range(3)],   axis=0)
        nm = np.linalg.norm(d)
        vecs.append(d / nm if nm > 1e-12 else np.zeros_like(d))
    return np.stack(vecs)

cd_in  = cd_vecs(in_slots,  N)
cd_out = cd_vecs(out_slots, N)

w_cdmean_in  = cd_in.mean(axis=0);  w_cdmean_in  /= np.linalg.norm(w_cdmean_in)
w_cdmean_out = cd_out.mean(axis=0); w_cdmean_out /= np.linalg.norm(w_cdmean_out)

svm_cd_i = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(
    np.vstack([cd_in,  -cd_in ]), y_svm)
svm_cd_o = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(
    np.vstack([cd_out, -cd_out]), y_svm)
w_cdsvm_in  = sk_normalize(svm_cd_i.coef_)[0]
w_cdsvm_out = sk_normalize(svm_cd_o.coef_)[0]

# E — cPCA α=1.0 e α=2.0
def cov(X):
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (len(X) - 1)

X_GT_in  = np.vstack([in_slots[s]  for s in range(3)])
X_W_in   = np.vstack([in_slots[s]  for s in range(3,6)])
X_GT_out = np.vstack([out_slots[s] for s in range(3)])
X_W_out  = np.vstack([out_slots[s] for s in range(3,6)])
S_GT_in  = cov(X_GT_in);  S_W_in  = cov(X_W_in)
S_GT_out = cov(X_GT_out); S_W_out = cov(X_W_out)

def cpca_axis(alpha):
    _, evecs_i = np.linalg.eigh(S_W_in  - alpha * S_GT_in)
    _, evecs_o = np.linalg.eigh(S_W_out - alpha * S_GT_out)
    wi = evecs_i[:, -1].copy(); wo = evecs_o[:, -1].copy()
    if (X_W_in  @ wi).mean() < (X_GT_in  @ wi).mean(): wi = -wi
    if (X_W_out @ wo).mean() < (X_GT_out @ wo).mean(): wo = -wo
    return wi, wo

w_cpca10_in,  w_cpca10_out  = cpca_axis(1.0)
w_cpca20_in,  w_cpca20_out  = cpca_axis(2.0)

print("Running pipelines ...")

# ── 7. Run tutti i metodi ──────────────────────────────────────────────────────
methods = [
    ("SVM-raw   ★PROD", w_svm_in,     w_svm_out),
    ("SVM-PC1",         w_pc1_in,     w_pc1_out),
    ("Reg-PC1",         w_reg_in,     w_reg_out),
    ("CD-mean",         w_cdmean_in,  w_cdmean_out),
    ("CD-svm",          w_cdsvm_in,   w_cdsvm_out),
    ("cPCA α=1.0",      w_cpca10_in,  w_cpca10_out),
    ("cPCA α=2.0",      w_cpca20_in,  w_cpca20_out),
]

results = {}
for name, wi, wo in methods:
    H_or, H_on = run_pipeline(wi, wo)
    results[name] = (H_or, H_on)

# ── 8. Stampa tabella ─────────────────────────────────────────────────────────
NM = 22
NC = 24
NCOLS = len(TARGETS)
SEP = "─" * (NM + 9 + (NC+3)*NCOLS + 2)

def prow(H):
    return [f"{pearsonr(H, t)[0]:+.4f} (p={pearsonr(H, t)[1]:.1e})" for _, t in TARGETS]

header = f"{'Metodo':<{NM}} {'Modo':^7} | " + " | ".join(f"{n:^{NC}}" for n, _ in TARGETS)

print("\n" + "═"*(len(header)+2))
print("ABLATION ARRICCHITA — tutti i metodi × 4 target")
print("═"*(len(header)+2))
print("\n" + header)
print(SEP)

for name, (H_or, H_on) in results.items():
    for H, mode in [(H_or, "ORACLE"), (H_on, "ONLINE")]:
        row = prow(H)
        line = f"{name:<{NM}} {mode:^7} | " + " | ".join(f"{c:^{NC}}" for c in row)
        print(line)
    print(SEP)

# ── 9. Riepilogo ONLINE ────────────────────────────────────────────────────────
print("\nRIEPILOGO ONLINE — ordinato per WER_in")
print(SEP)
print(header)
print(SEP)
online_rows = [(name, results[name][1]) for name, *_ in methods]
online_rows.sort(key=lambda x: -pearsonr(x[1], wer_in)[0])
for name, H in online_rows:
    row = prow(H)
    print(f"{name:<{NM}} {'ONLINE':^7} | " + " | ".join(f"{c:^{NC}}" for c in row))
print(SEP)
