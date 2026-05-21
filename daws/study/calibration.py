"""
study/calibration.py
--------------------
Calibration of cross-modal UQ weights and risk thresholds from Inv-Entropy study.

Step 1 — Load ie_study_ablation.json (50 samples, u_asr + wer).
         Join with LLM cache sd_w1.u_llm (real Semantic Density per W1).
         Fallback: H_k4 normalised as U_LLM proxy if sd_w1 not available.
Step 2 — Fit alpha, beta, gamma via scipy SLSQP:
         minimise MSE(U_pipeline, WER_norm) subject to alpha+beta+gamma=1, all≥0
Step 3 — Thresholds from empirical percentiles of U_pipeline (P33, P66).
Step 4 — Save config/thresholds.json.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_ABLATION_FILE = Path(__file__).parent.parent / "results" / "ie_study_ablation.json"
_STUDY_FILE    = Path(__file__).parent.parent / "results" / "inv_entropy_study.json"
_LLM_CACHE_DIR = Path(__file__).parent.parent / "results" / "llm_cache"
CONFIG_FILE    = Path(__file__).parent.parent.parent / "config" / "thresholds.json"

_P33, _P66 = 33, 66


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Optional[Path] = None):
    """
    Returns (u_asr_norm, u_llm, wer_norm, n, source).
    u_llm: real SD-based U_LLM from llm_cache/sd_w1, or H_k4 normalised as fallback.
    wer_norm: WER min-max normalised to [0,1] — calibration target.
    """
    if path is None:
        path = _ABLATION_FILE if _ABLATION_FILE.exists() else _STUDY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Study results not found: {path}\nRun --mode ie-study first.")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    u_asrs = np.array([r.get("u_asr", 0.0) for r in records])
    wers   = np.array([r.get("wer", 0.0) for r in records])

    # Try real SD u_llm from llm_cache/sd_w1
    u_llm_vals, source = [], "sd_w1"
    for r in records:
        stem = r.get("filename", "")
        cache_path = _LLM_CACHE_DIR / f"{stem}.json"
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            sd_w1 = cache.get("sd_w1", {})
            u_llm_vals.append(sd_w1.get("u_llm", float("nan")))
        else:
            u_llm_vals.append(float("nan"))

    u_llm_arr = np.array(u_llm_vals, dtype=float)
    n_valid = int(np.sum(~np.isnan(u_llm_arr)))

    if n_valid < len(records) // 2:
        # Fallback: use H_k4 normalised
        logger.warning(f"Only {n_valid}/{len(records)} sd_w1 entries — falling back to H_k4 proxy.")
        h_raw = np.array([r.get("H_k4", r.get("h_norm", 0.0)) for r in records])
        lo, hi = h_raw.min(), h_raw.max()
        u_llm_arr = (h_raw - lo) / (hi - lo) if hi > lo else np.zeros_like(h_raw)
        source = "H_k4_proxy"
    else:
        # Fill any remaining NaN with column mean
        col_mean = float(np.nanmean(u_llm_arr))
        u_llm_arr = np.where(np.isnan(u_llm_arr), col_mean, u_llm_arr)

    def _minmax(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    print(f"  U_LLM source: {source}  ({n_valid}/{len(records)} real SD values)")
    return _minmax(u_asrs), u_llm_arr, _minmax(wers), len(records)


def _u_pipeline(u_asr: np.ndarray, u_llm: np.ndarray,
                alpha: float, beta: float, gamma: float) -> np.ndarray:
    return alpha * u_asr + beta * u_llm + gamma * u_asr * u_llm


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def calibrate(
    results_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> dict:
    """
    Calibrate weights and thresholds. Saves config/thresholds.json.
    Returns the config dict.
    """
    from scipy.optimize import minimize

    rpath = Path(results_path) if results_path else None
    cpath = Path(config_path) if config_path else CONFIG_FILE

    u_asrs, h_norm, wer_norm, n = _load(rpath)
    print(f"Calibrating on {n} recordings.")

    # ── Step 3: scipy SLSQP — minimise MSE(U_pipeline, WER_norm) ─────────────
    # Target = WER_norm (ground truth).  U_LLM proxy = h_norm.
    # U_pipeline = α·u_asr + β·h_norm + γ·u_asr·h_norm  subject to α+β+γ=1

    def objective(w):
        u_pipe = w[0] * u_asrs + w[1] * h_norm + w[2] * u_asrs * h_norm
        return float(np.mean((u_pipe - wer_norm) ** 2))

    opt = minimize(
        objective,
        x0=[0.33, 0.34, 0.33],
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints={"type": "eq", "fun": lambda w: w[0] + w[1] + w[2] - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if opt.success:
        alpha, beta, gamma = float(opt.x[0]), float(opt.x[1]), float(opt.x[2])
    else:
        logger.warning(f"SLSQP did not converge ({opt.message}) — using defaults.")
        alpha, beta, gamma = 0.3, 0.5, 0.2

    print(f"  alpha (U_ASR):       {alpha:.4f}")
    print(f"  beta  (U_LLM):       {beta:.4f}")
    print(f"  gamma (interact.):   {gamma:.4f}")
    print(f"  SLSQP converged:     {opt.success}  MSE={opt.fun:.6f}")

    # ── Step 4: thresholds from empirical percentiles of U_pipeline ───────────
    u_pipes = _u_pipeline(u_asrs, h_norm, alpha, beta, gamma)
    tg = float(np.percentile(u_pipes, _P33))
    tr = float(np.percentile(u_pipes, _P66))

    # Guarantee tg < tr with minimum gap
    if tg >= tr:
        tg, tr = min(tg, tr) - 0.01, max(tg, tr) + 0.01
    tg = float(np.clip(tg, 0.05, 0.50))
    tr = float(np.clip(tr, tg + 0.05, 0.95))

    pct_b = float((h_norm > np.percentile(h_norm, _P66)).mean()) * 100
    pct_a = 100.0 - pct_b

    print(f"  threshold_green:     {tg:.4f}  (P{_P33} of U_pipeline)")
    print(f"  threshold_red:       {tr:.4f}  (P{_P66} of U_pipeline)")
    print(f"  Case A (robust.):    {pct_a:.1f}%")
    print(f"  Case B (amplif.):    {pct_b:.1f}%")

    # ── Step 5: save ──────────────────────────────────────────────────────────
    cfg = {
        "alpha": round(alpha, 4),
        "beta":  round(beta,  4),
        "gamma": round(gamma, 4),
        "threshold_green": round(tg, 4),
        "threshold_red":   round(tr, 4),
        "calibration_method": "scipy_SLSQP",
        "calibration_mse": round(float(opt.fun), 6),
        "inv_entropy_p33": round(float(np.percentile(h_norm, _P33)), 4),
        "inv_entropy_p66": round(float(np.percentile(h_norm, _P66)), 4),
        "pct_case_a_robustness":    round(pct_a, 1),
        "pct_case_b_amplification": round(pct_b, 1),
        "n_samples": n,
    }

    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"  Saved → {cpath}")
    return cfg
