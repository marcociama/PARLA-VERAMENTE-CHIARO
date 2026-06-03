"""
scripts/cacioli_study.py
------------------------
Cacioli corpus validation study — N=141 Neapolitan clips

Phase 1 (compute-heavy, ~1-2h):
  For each clip in Neapolitan-Spoken-Corpus/audioData/*.m4a:
    1. Convert m4a → wav (16 kHz mono via ffmpeg)
    2. WhisperX + TTA: W1 (clean), W2 (noise seed=1), W3 (noise seed=2), alpha=0.005
    3. Parse 3 Italian GT variants from traduzioni_italiano/{stem}.txt (split by ";")
    4. Ollama Mistral T=0 seed=0 → 6 LLM responses: [rGT1, rGT2, rGT3, rW1, rW2, rW3]
    5. Save scripts/cacioli_llm_cache/{stem}.json (same structure as daws/results/llm_cache/)

Phase 2 (fast, ~5 min — run after Phase 1 completes):
  1. Load N=50 PARLA CHIARO + N=141 Cacioli → union N=191
  2. SBERT encode all 191×6 texts for both input and response spaces
  3. Re-fit LinearSVC on union → w_in_union, w_out_union
  4. Compute sigma_in, sigma_out on union projections (median pairwise)
  5. Re-compute GT anchors from PC slots using union drift axis
  6. H_spectral ONLINE for each Cacioli clip (anchors frozen to PC)
  7. Report Pearson(H_spectral, WER_in/WER_out/BLEU_out/E_sem_top) on N=141
  8. Save scripts/cacioli_results.json
  NOTE: config/geometry_calibration.json is NOT modified.

Usage:
  python scripts/cacioli_study.py            # auto-detect phase
  python scripts/cacioli_study.py --phase 1  # force Phase 1
  python scripts/cacioli_study.py --phase 2  # force Phase 2
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy.stats import pearsonr

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

CORPUS_DIR  = BASE / "Neapolitan-Spoken-Corpus"
AUDIO_DIR   = CORPUS_DIR / "audioData"
TRANSL_DIR  = CORPUS_DIR / "traduzioni_italiano"
TRANSC_CSV  = CORPUS_DIR / "transcripts.csv"
CACHE_DIR   = BASE / "scripts" / "cacioli_llm_cache"
ASR_CACHE   = BASE / "scripts" / "cacioli_asr_cache"
WAV_DIR     = BASE / "scripts" / "cacioli_wav"
PC_LLM_DIR  = BASE / "daws" / "results" / "llm_cache"
PC_IE_JSON  = BASE / "daws" / "results" / "ie_study_ablation.json"
CFG_PATH    = BASE / "config" / "geometry_calibration.json"
RESULTS_OUT = BASE / "scripts" / "cacioli_results.json"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

_CULTURAL_PROMPT = (
    "Sei un assistente culturale. Rispondi brevemente in italiano alla seguente "
    "frase in massimo 2 frasi.\n\nTesto: {text}\n\nRisposta:"
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _wer(ref: str, hyp: str) -> float:
    ref_t, hyp_t = ref.lower().split(), hyp.lower().split()
    if not ref_t:
        return 1.0 if hyp_t else 0.0
    n, m = len(ref_t), len(hyp_t)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if ref_t[i - 1] == hyp_t[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / n


def _ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 120, "seed": 0},
    }
    for wait in [0, 5, 10, 20]:
        if wait:
            print(f"  Ollama retry in {wait}s ...", flush=True)
            time.sleep(wait)
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=300)
            r.raise_for_status()
            return r.json().get("response", "").strip().split("\n")[0].strip()
        except requests.HTTPError as exc:
            if exc.response.status_code == 503 and wait < 20:
                continue
            raise
    raise RuntimeError("Ollama unreachable after 4 attempts")


def _parse_gt(stem: str) -> tuple[str, str, str]:
    txt = TRANSL_DIR / f"{stem}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Missing translation: {txt}")
    parts = [v.strip() for v in txt.read_text(encoding="utf-8").strip().split(";") if v.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], parts[1]
    p = parts[0] if parts else ""
    return p, p, p


def _load_metadata() -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with open(TRANSC_CSV, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if row[0] == "File Name":
                continue
            fname, domain = row[0].strip(), row[1].strip()
            nap_text = row[2].strip() if len(row) > 2 else ""
            meta[fname.replace(".m4a", "")] = {"neapolitan_text": nap_text, "domain": domain}
    return meta


def _convert_m4a(m4a_path: Path, wav_path: Path) -> None:
    """m4a → 16 kHz mono wav. Cacioli is already normalized per §B.3 of the paper."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(m4a_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True, check=True,
    )


# ── Phase 1: ASR + LLM inference ──────────────────────────────────────────────

def phase1() -> None:
    from daws.pipeline.asr import ASRPipeline

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ASR_CACHE.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    meta = _load_metadata()
    m4a_files = sorted(AUDIO_DIR.glob("*.m4a"))
    print(f"Found {len(m4a_files)} .m4a files in {AUDIO_DIR}")

    asr = ASRPipeline(language="it")
    done = skipped = 0

    for m4a in m4a_files:
        stem = m4a.stem
        out_path = CACHE_DIR / f"{stem}.json"
        if out_path.exists():
            skipped += 1
            continue

        idx = done + skipped + 1
        print(f"\n[{idx}/{len(m4a_files)}] {stem}", flush=True)

        # m4a → 16 kHz mono wav (Cacioli already normalized per §B.3)
        wav_path = WAV_DIR / f"{stem}.wav"
        if not wav_path.exists():
            print("  convert m4a → wav ...", flush=True)
            _convert_m4a(m4a, wav_path)

        # WhisperX + TTA
        asr_result = asr.process(str(wav_path), cache_path=str(ASR_CACHE / f"{stem}.json"))
        w1, w2, w3 = asr_result.top3[0], asr_result.top3[1], asr_result.top3[2]
        u_asr = float(asr_result.u_asr)
        print(f"  W1: '{w1[:70]}'", flush=True)

        # GT variants (split by ";")
        gt1, gt2, gt3 = _parse_gt(stem)

        # WER_in
        wer = _wer(gt1, w1)

        # LLM: 6 responses [rGT1, rGT2, rGT3, rW1, rW2, rW3]
        inputs = [gt1, gt2, gt3, w1, w2, w3]
        responses: list[str] = []
        for s, text in enumerate(inputs):
            label = "GT" if s < 3 else "W"
            print(f"  LLM slot {s} ({label}{(s%3)+1}): '{text[:50]}' ...", flush=True)
            resp = _ollama(_CULTURAL_PROMPT.format(text=text))
            responses.append(resp)
            print(f"    → '{resp[:70]}'", flush=True)

        meta_item = meta.get(stem, {"neapolitan_text": "", "domain": "unknown"})
        record = {
            "filename": stem,
            "neapolitan_text": meta_item["neapolitan_text"],
            "domain": meta_item["domain"],
            "gt_variants": [gt1, gt2, gt3],
            "inputs": inputs,
            "responses": responses,
            "wer": round(wer, 6),
            "u_asr": round(u_asr, 6),
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1

    total = done + skipped
    print(f"\nPhase 1 done: {done} processed, {skipped} skipped ({total}/{len(m4a_files)} total)")
    if total < len(m4a_files):
        print("Some clips missing — re-run to retry.")


# ── Phase 2: Union calibration + analysis ─────────────────────────────────────

def _bleu_err(ref: str, hyp: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    return 1.0 - sentence_bleu([ref.split()], hyp.split(),
                                smoothing_function=SmoothingFunction().method1)


def _laplacian(pts: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-np.abs(pts[:, None] - pts[None, :]) / sigma)


def _row_stochastic(A: np.ndarray) -> np.ndarray:
    s = A.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return A / s


def _spectral_H(pts_in: np.ndarray, pts_out: np.ndarray,
                sig_in: float, sig_out: float) -> float:
    eigs = np.abs(np.linalg.eigvals(
        _row_stochastic(_laplacian(pts_out, sig_out)) @
        _row_stochastic(_laplacian(pts_in,  sig_in))
    ))
    tot = eigs.sum()
    if tot < 1e-12:
        return 0.0
    e = eigs / tot
    e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))


def _median_sigma(proj: np.ndarray) -> float:
    pairs = np.abs(proj[:, None] - proj[None, :])
    return float(np.median(pairs[np.triu_indices(len(proj), k=1)]))


def _l2_norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n


def phase2() -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import normalize as sk_normalize

    # ── Load PARLA CHIARO N=50 ─────────────────────────────────────────────────
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

    # ── Load Cacioli N=141 ─────────────────────────────────────────────────────
    ca_files = sorted(CACHE_DIR.glob("*.json"))
    if not ca_files:
        print("ERROR: No Cacioli JSON found in cacioli_llm_cache/. Run Phase 1 first.")
        sys.exit(1)

    ca_records   = [json.loads(f.read_text(encoding="utf-8")) for f in ca_files]
    ca_stems     = [r["filename"]  for r in ca_records]
    ca_domains   = [r["domain"]    for r in ca_records]
    ca_wer       = np.array([r["wer"] for r in ca_records])
    N_CA = len(ca_records)

    ca_inputs    = [[] for _ in range(6)]
    ca_responses = [[] for _ in range(6)]
    for r in ca_records:
        for s in range(6):
            ca_inputs[s].append(r["inputs"][s])
            ca_responses[s].append(r["responses"][s])

    N_UNION = N_PC + N_CA
    print(f"N_PC={N_PC}  N_CA={N_CA}  N_UNION={N_UNION}")

    # ── SBERT — encode union (slot-major: 6 × N_UNION rows each) ──────────────
    # Row s*N_UNION + i → slot s, sample i (PC samples: i < N_PC, CA: i >= N_PC)
    union_in_flat  = [t for s in range(6) for t in pc_inputs[s]    + ca_inputs[s]]
    union_out_flat = [t for s in range(6) for t in pc_responses[s] + ca_responses[s]]

    print("Loading SBERT ...")
    sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="mps")

    print(f"Encoding {len(union_in_flat)} INPUT texts (union) ...")
    X_in  = sbert.encode(union_in_flat,  batch_size=32, show_progress_bar=True,
                          normalize_embeddings=False)
    print(f"Encoding {len(union_out_flat)} RESPONSE texts (union) ...")
    X_out = sbert.encode(union_out_flat, batch_size=32, show_progress_bar=True,
                          normalize_embeddings=False)

    # ── SVM on union N=191 — GT=0, W=1 ────────────────────────────────────────
    labels = np.array([0] * (3 * N_UNION) + [1] * (3 * N_UNION))

    print("Fitting SVM (input space) ...")
    svm_in  = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_in,  labels)
    print("Fitting SVM (output space) ...")
    svm_out = LinearSVC(C=1.0, max_iter=10_000, dual=True).fit(X_out, labels)

    w_in  = sk_normalize(svm_in.coef_)[0]
    w_out = sk_normalize(svm_out.coef_)[0]
    acc_in  = float((svm_in.predict(X_in)   == labels).mean())
    acc_out = float((svm_out.predict(X_out) == labels).mean())
    print(f"SVM union accuracy — in: {acc_in:.3f}  out: {acc_out:.3f}")

    # ── Sigma on all union projections ─────────────────────────────────────────
    proj_in_all  = X_in  @ w_in
    proj_out_all = X_out @ w_out
    sigma_in  = _median_sigma(proj_in_all)
    sigma_out = _median_sigma(proj_out_all)
    print(f"sigma_in={sigma_in:.5f}  sigma_out={sigma_out:.5f}")

    # ── GT anchors re-computed with union drift axis ───────────────────────────
    # Re-project PC GT slots (0,1,2) onto union w_in / w_out
    pc_in_slots  = [X_in [s * N_UNION : s * N_UNION + N_PC]  for s in range(6)]
    pc_out_slots = [X_out[s * N_UNION : s * N_UNION + N_PC]  for s in range(6)]
    proj_pc_in   = [pc_in_slots[s]  @ w_in  for s in range(6)]
    proj_pc_out  = [pc_out_slots[s] @ w_out for s in range(6)]
    anchors_in   = [float(proj_pc_in[s].mean())  for s in range(3)]
    anchors_out  = [float(proj_pc_out[s].mean()) for s in range(3)]
    print(f"Anchors in  (union axis): {[f'{a:.4f}' for a in anchors_in]}")
    print(f"Anchors out (union axis): {[f'{a:.4f}' for a in anchors_out]}")

    # ── H_spectral ONLINE for each Cacioli clip ────────────────────────────────
    ca_in_slots  = [X_in [s * N_UNION + N_PC : (s + 1) * N_UNION] for s in range(6)]
    ca_out_slots = [X_out[s * N_UNION + N_PC : (s + 1) * N_UNION] for s in range(6)]
    proj_ca_in   = [ca_in_slots[s]  @ w_in  for s in range(6)]
    proj_ca_out  = [ca_out_slots[s] @ w_out for s in range(6)]

    print("Computing H_spectral for N_CA clips ...")
    H_spec = np.array([
        _spectral_H(
            np.array(anchors_in  + [proj_ca_in[3][i],  proj_ca_in[4][i],  proj_ca_in[5][i]]),
            np.array(anchors_out + [proj_ca_out[3][i], proj_ca_out[4][i], proj_ca_out[5][i]]),
            sigma_in, sigma_out,
        )
        for i in range(N_CA)
    ])

    H_MIN = float(H_spec.min())
    H_MAX = float(H_spec.max())
    H_risk = np.clip((H_spec - H_MIN) / max(H_MAX - H_MIN, 1e-8), 0.0, 1.0)
    print(f"H_spectral: min={H_MIN:.4f}  max={H_MAX:.4f}  mean={H_spec.mean():.4f}")

    # ── Output metrics (Cacioli) ───────────────────────────────────────────────
    print("Computing WER_out, BLEU_out, E_sem_top ...")
    ca_out_norm = [_l2_norm(ca_out_slots[s]) for s in range(6)]

    wer_out = np.array([
        _wer(ca_records[i]["responses"][0], ca_records[i]["responses"][3])
        for i in range(N_CA)
    ])
    bleu_out = np.array([
        _bleu_err(ca_records[i]["responses"][0], ca_records[i]["responses"][3])
        for i in range(N_CA)
    ])
    E_sem_top = np.array([
        1.0 - float(np.dot(ca_out_norm[3][i], ca_out_norm[0][i]))
        for i in range(N_CA)
    ])

    # Also compute H_spectral for PC clips (union-calibrated) for comparison
    print("Computing H_spectral for N_PC clips (union-calibrated) ...")
    H_spec_pc = np.array([
        _spectral_H(
            np.array(anchors_in  + [proj_pc_in[3][i],  proj_pc_in[4][i],  proj_pc_in[5][i]]),
            np.array(anchors_out + [proj_pc_out[3][i], proj_pc_out[4][i], proj_pc_out[5][i]]),
            sigma_in, sigma_out,
        )
        for i in range(N_PC)
    ])
    pc_wer_arr = np.array([r["wer"] for r in pc_meta])
    r_pc, p_pc = pearsonr(H_spec_pc, pc_wer_arr)
    print(f"PC (union-calibrated): Pearson(H_spec, WER_in) = {r_pc:+.4f}  p={p_pc:.2e}")

    # ── Pearson report ─────────────────────────────────────────────────────────
    targets = [
        ("WER_in",    ca_wer),
        ("WER_out",   wer_out),
        ("BLEU_out",  bleu_out),
        ("E_sem_top", E_sem_top),
    ]

    print("\n" + "=" * 65)
    print(f"  CACIOLI N={N_CA} — 1D Markov ONLINE (union-calibrated N={N_UNION})")
    print("=" * 65)
    pearson_dict: dict[str, float] = {}
    for name, target in targets:
        r, p = pearsonr(H_spec, target)
        print(f"  Pearson(H_spectral, {name:10s}) = {r:+.4f}  (p={p:.2e})")
        pearson_dict[f"pearson_{name.lower()}"] = round(float(r), 4)
    print("=" * 65)

    # Domain breakdown
    print("\nDomain breakdown:")
    for domain in sorted(set(ca_domains)):
        idxs = [i for i, d in enumerate(ca_domains) if d == domain]
        print(f"  {domain:30s} N={len(idxs):3d}  "
              f"WER_in={ca_wer[np.array(idxs)].mean():.3f}  "
              f"H_spec={H_spec[np.array(idxs)].mean():.4f}")

    # ── Save results ────────────────────────────────────────────────────────────
    per_clip = [
        {
            "filename": ca_stems[i],
            "domain":   ca_domains[i],
            "wer":      round(float(ca_wer[i]),    4),
            "wer_out":  round(float(wer_out[i]),   4),
            "bleu_out": round(float(bleu_out[i]),  4),
            "e_sem_top":round(float(E_sem_top[i]), 4),
            "h_spectral":round(float(H_spec[i]),   6),
            "h_risk":   round(float(H_risk[i]),    4),
        }
        for i in range(N_CA)
    ]

    results = {
        "n_parla_chiaro": N_PC,
        "n_cacioli": N_CA,
        "n_union": N_UNION,
        "h_spectral_min": round(H_MIN, 6),
        "h_spectral_max": round(H_MAX, 6),
        "svm_union_acc_in":  round(acc_in,  4),
        "svm_union_acc_out": round(acc_out, 4),
        "sigma_in":  round(sigma_in,  6),
        "sigma_out": round(sigma_out, 6),
        "anchors_in_union":  [round(a, 6) for a in anchors_in],
        "anchors_out_union": [round(a, 6) for a in anchors_out],
        "pearson_pc_wer_union_calibrated": round(float(r_pc), 4),
        **pearson_dict,
        "per_clip": per_clip,
    }

    RESULTS_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults → {RESULTS_OUT}")
    print("config/geometry_calibration.json NOT modified (production pipeline unchanged)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cacioli corpus validation study")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None)
    args = parser.parse_args()

    if args.phase == 1:
        phase1()
    elif args.phase == 2:
        phase2()
    else:
        done  = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
        total = len(list(AUDIO_DIR.glob("*.m4a")))
        if done < total:
            print(f"Auto-detect: {done}/{total} clips cached → Phase 1")
            phase1()
        else:
            print(f"Auto-detect: {done}/{total} clips cached → Phase 2")
            phase2()


if __name__ == "__main__":
    main()
