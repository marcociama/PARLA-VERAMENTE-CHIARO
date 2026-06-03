"""
scripts/comparison_table.py
----------------------------
Tabella comparativa finale:
  1D Markov Spettrale ORACLE  (GT reali a runtime)
  1D Markov Spettrale ONLINE  (anchor congelati — produzione)
  Inv-Entropy OFFLINE 768D    (GT reali a runtime — upper bound)

Metriche (tutte "higher = more damage/divergence"):
  WER_in      WER(GT_input,   W1_input)
  WER_out     WER(R_GT1,      R_W1)
  BLEU_in     1 - BLEU(GT_input,   W1_input)
  BLEU_out    1 - BLEU(R_GT1,      R_W1)
  E_sem_top   1 - cos(R_W1, R_GT1)            top-beam
  E_sem_cross 1 - mean_cos  9 coppie W×GT
"""

import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from daws.pipeline.inv_entropy import _row_stochastic

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

# ── 2. Metriche testuali ───────────────────────────────────────────────────────
def _ed(a, b):
    m, n = len(a), len(b)
    dp = np.arange(n + 1)
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]; dp[j] = prev if a[i-1]==b[j-1] else 1+min(prev,dp[j],dp[j-1]); prev=tmp
    return int(dp[n])

def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    return _ed(r, h) / len(r) if r else (0. if not h else 1.)

_smooth = SmoothingFunction().method1
def bleu_err(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    if not r or not h: return 1.
    return 1. - sentence_bleu([r], h, smoothing_function=_smooth)

wer_out  = np.array([wer(responses_by_slot[0][i], responses_by_slot[3][i]) for i in range(N)])
bleu_in  = np.array([bleu_err(inputs_by_slot[0][i],    inputs_by_slot[3][i])    for i in range(N)])
bleu_out = np.array([bleu_err(responses_by_slot[0][i], responses_by_slot[3][i]) for i in range(N)])

# ── 3. SBERT ──────────────────────────────────────────────────────────────────
print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
inputs_flat    = [t for s in range(6) for t in inputs_by_slot[s]]
responses_flat = [t for s in range(6) for t in responses_by_slot[s]]
X_in  = sbert.encode(inputs_flat,    batch_size=32, show_progress_bar=False, normalize_embeddings=False)
X_out = sbert.encode(responses_flat, batch_size=32, show_progress_bar=False, normalize_embeddings=False)

in_slots  = [X_in [k*N:(k+1)*N] for k in range(6)]
out_slots = [X_out[k*N:(k+1)*N] for k in range(6)]

# ── 4. E_sem ──────────────────────────────────────────────────────────────────
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n < 1e-12] = 1.; return X / n

out_norm    = [_l2(out_slots[s]) for s in range(6)]
E_sem_top   = np.array([1. - float(np.dot(out_norm[3][i], out_norm[0][i])) for i in range(N)])
E_sem_cross = np.array([1. - np.mean([float(np.dot(out_norm[w][i], out_norm[g][i]))
                                      for w in [3,4,5] for g in [0,1,2]]) for i in range(N)])

TARGETS = [
    ("WER_in",      wer_in),
    ("WER_out",     wer_out),
    ("BLEU_in",     bleu_in),
    ("BLEU_out",    bleu_out),
    ("E_sem_top",   E_sem_top),
    ("E_sem_cross", E_sem_cross),
]

# ── 5. 1D Markov Spettrale ─────────────────────────────────────────────────────
def _lap(x, s): return np.exp(-np.abs(x[:,None]-x[None,:])/s)
def _rs(A): s=A.sum(1,keepdims=True); s[s<1e-12]=1.; return A/s
def spectral_H(pi, po, si, so):
    eigs = np.abs(np.linalg.eigvals(_rs(_lap(po,so)) @ _rs(_lap(pi,si))))
    tot  = eigs.sum()
    if tot < 1e-12: return 0.
    e = eigs/tot; e = e[e>1e-12]
    return float(-np.sum(e*np.log(e)))
def med_sig(x): p=np.abs(x[:,None]-x[None,:]); return float(np.median(p[np.triu_indices(len(x),k=1)]))

labels = np.array([0]*(3*N)+[1]*(3*N))
svm_i  = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_in,  labels)
svm_o  = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_out, labels)
w_in   = sk_normalize(svm_i.coef_)[0]
w_out  = sk_normalize(svm_o.coef_)[0]

pi_all = X_in  @ w_in;  pi_s = [pi_all[k*N:(k+1)*N] for k in range(6)]
po_all = X_out @ w_out; po_s = [po_all[k*N:(k+1)*N] for k in range(6)]
si, so = med_sig(pi_all), med_sig(po_all)
ai = [float(pi_s[s].mean()) for s in range(3)]
ao = [float(po_s[s].mean()) for s in range(3)]

H_markov_oracle = np.array([
    spectral_H(np.array([pi_s[s][i] for s in range(6)]),
               np.array([po_s[s][i] for s in range(6)]), si, so) for i in range(N)])

H_markov_online = np.array([
    spectral_H(np.array(ai + [pi_s[3][i], pi_s[4][i], pi_s[5][i]]),
               np.array(ao + [po_s[3][i], po_s[4][i], po_s[5][i]]), si, so) for i in range(N)])

# ── 6. Inv-Entropy OFFLINE 768D ───────────────────────────────────────────────
def sim_kd(E):
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n<1e-12]=1.
    en = E/n; return np.clip(en@en.T, 0., 1.)**4

def inv_entropy_H(ei, eo):
    Px = _row_stochastic(sim_kd(ei)); Py = _row_stochastic(sim_kd(eo))
    d  = np.diag(Py@Px)
    return float(-np.sum(d*np.log(d+1e-12)))

H_ie_offline = np.array([
    inv_entropy_H(np.stack([in_slots[s][i]  for s in range(6)]),
                  np.stack([out_slots[s][i] for s in range(6)])) for i in range(N)])

# ── 7. Tabella ────────────────────────────────────────────────────────────────
METHODS = [
    ("1D Markov ORACLE", H_markov_oracle, "ORACLE", "GT reali a runtime"),
    ("1D Markov ONLINE", H_markov_online, "ONLINE", "anchor congelati ★PROD"),
    ("Inv-Entropy 768D", H_ie_offline,    "OFFLINE","GT reali a runtime"),
]

NC = 24
NM = 20
NMD = 26
SEP = "─" * (NM + NMD + 10 + (NC+3)*len(TARGETS))

header = (f"{'Metodo':<{NM}} {'Dettaglio':<{NMD}} {'Modo':^7} | "
          + " | ".join(f"{n:^{NC}}" for n,_ in TARGETS))

print("\n" + "═"*len(header))
print("CONFRONTO FINALE — 1D Markov vs Inv-Entropy × 6 target")
print("═"*len(header))
print(header)
print(SEP)

for name, H, mode, detail in METHODS:
    cells = [f"{pearsonr(H,t)[0]:+.4f} (p={pearsonr(H,t)[1]:.1e})" for _,t in TARGETS]
    print(f"{name:<{NM}} {detail:<{NMD}} {mode:^7} | " + " | ".join(f"{c:^{NC}}" for c in cells))

print(SEP)

# ── 8. Statistiche target ─────────────────────────────────────────────────────
print("\nStatistiche target:")
for name, t in TARGETS:
    print(f"  {name:<14}: mean={t.mean():.4f}  std={t.std():.4f}  "
          f"range=[{t.min():.3f}, {t.max():.3f}]")

print(f"\nCorrelazioni inter-target (Pearson):")
tnames = [n for n,_ in TARGETS]
tvecs  = [t for _,t in TARGETS]
for i in range(len(TARGETS)):
    for j in range(i+1, len(TARGETS)):
        r = pearsonr(tvecs[i], tvecs[j])[0]
        print(f"  {tnames[i]:<14} × {tnames[j]:<14}: {r:+.4f}")
