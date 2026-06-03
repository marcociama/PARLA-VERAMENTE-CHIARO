"""Cacioli-only calibration with v2 responses (neutral summarisation prompt)."""
import json, numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as sk_normalize

BASE  = Path(__file__).parent.parent
CACHE = BASE / "scripts/cacioli_llm_cache_v2"
RES   = BASE / "scripts/cacioli_results_v2.json"

files   = sorted(CACHE.glob("*.json"))
ca_data = [json.loads(f.read_text()) for f in files]
N = len(ca_data)
results  = json.loads(RES.read_text())
per_clip = {c["filename"]: c for c in results["per_clip"]}
fnames   = [d["filename"] for d in ca_data]
wer_in   = np.array([d["wer"]                  for d in ca_data])
wer_out  = np.array([per_clip[f]["wer_out"]   for f in fnames])
bleu_out = np.array([per_clip[f]["bleu_out"]  for f in fnames])
e_sem    = np.array([per_clip[f]["e_sem_top"] for f in fnames])

ca_inputs    = [[] for _ in range(6)]
ca_responses = [[] for _ in range(6)]
for r in ca_data:
    for s in range(6):
        ca_inputs[s].append(r["inputs"][s])
        ca_responses[s].append(r["responses"][s])

in_flat  = [t for s in range(6) for t in ca_inputs[s]]
out_flat = [t for s in range(6) for t in ca_responses[s]]

print("Loading SBERT ...")
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")
print("Encoding inputs ...")
X_in  = sbert.encode(in_flat,  batch_size=32, show_progress_bar=True,  normalize_embeddings=False)
print("Encoding responses ...")
X_out = sbert.encode(out_flat, batch_size=32, show_progress_bar=True, normalize_embeddings=False)

labels = np.array([0]*(3*N) + [1]*(3*N))
svm_i = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_in,  labels)
svm_o = LinearSVC(C=1., max_iter=10_000, dual=True).fit(X_out, labels)
w_in  = sk_normalize(svm_i.coef_)[0]
w_out = sk_normalize(svm_o.coef_)[0]
print(f"SVM acc — in:{svm_i.score(X_in,labels):.3f}  out:{svm_o.score(X_out,labels):.3f}")

def median_sigma(proj):
    pairs = np.abs(proj[:, None] - proj[None, :])
    return float(np.median(pairs[np.triu_indices(len(proj), k=1)]))

proj_in_all  = X_in  @ w_in
proj_out_all = X_out @ w_out
sigma_in  = median_sigma(proj_in_all)
sigma_out = median_sigma(proj_out_all)
print(f"sigma_in={sigma_in:.5f}  sigma_out={sigma_out:.5f}")

in_slots  = [X_in [s*N:(s+1)*N] for s in range(6)]
out_slots = [X_out[s*N:(s+1)*N] for s in range(6)]
anchors_in  = [float((in_slots[s]  @ w_in).mean())  for s in range(3)]
anchors_out = [float((out_slots[s] @ w_out).mean()) for s in range(3)]

def laplacian(pts, sig):
    M = np.exp(-np.abs(pts[:,None]-pts[None,:])/sig)
    s = M.sum(1, keepdims=True); s[s<1e-12] = 1.
    return M / s

def h_spec(emb_in6, emb_out6):
    pi = emb_in6 @ w_in; po = emb_out6 @ w_out
    pts_in  = np.array(anchors_in  + [pi[3], pi[4], pi[5]])
    pts_out = np.array(anchors_out + [po[3], po[4], po[5]])
    ev = np.abs(np.linalg.eigvals(
        laplacian(pts_out, sigma_out) @ laplacian(pts_in, sigma_in)).real)
    ev = ev[ev > 1e-12]; ev /= ev.sum()
    return float(-np.sum(ev * np.log(ev + 1e-12)))

ca_in_slots  = [in_slots[s]  for s in range(6)]
ca_out_slots = [out_slots[s] for s in range(6)]
h = np.array([h_spec(
    np.stack([ca_in_slots[s][i]  for s in range(6)]),
    np.stack([ca_out_slots[s][i] for s in range(6)])
) for i in range(N)])

def pr(x, y, nm):
    r, p = pearsonr(x, y)
    return f"  {nm:12s}: r={r:+.4f}  p={p:.2e}"

print("\n" + "="*60)
print("  Cacioli-ONLY v2 (SVM Cacioli, anchors GT Cacioli)")
print("="*60)
print(pr(h, wer_in,   "WER_in"))
print(pr(h, wer_out,  "WER_out"))
print(pr(h, bleu_out, "BLEU_out"))
print(pr(h, e_sem,    "E_sem_top"))
print("="*60)
print(f"  H: min={h.min():.4f}  max={h.max():.4f}  mean={h.mean():.4f}")
print()
print("  Union-calibrated v2 per confronto:")
print("  WER_in  : r=+0.4236  p=1.66e-07")
print("  WER_out : r=+0.1892  p=2.47e-02")
print("  BLEU_out: r=+0.2766  p=9.00e-04")
print("  E_sem   : r=+0.4338  p=7.72e-08")
