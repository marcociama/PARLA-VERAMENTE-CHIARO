"""
daws/study/threshold_calibration.py
-------------------------------------
Derives H_risk triage thresholds for the 1D Markov Spettrale pipeline via
3-class ROC analysis on the PARLA CHIARO calibration corpus (N=50).

Method
------
Ground truth: combined damage label  damage = WER + |ΔPC1|
  - GREEN label boundary:  damage < P33(damage)
  - RED   label boundary:  damage > P66(damage)

GREEN threshold: H_risk* that achieves sensitivity ≥ 0.90 on (yellow+red) vs green
                 at maximum specificity — minimises clinical false negatives.
RED   threshold: H_risk* that maximises Youden's J = sensitivity + specificity − 1
                 on red vs (green+yellow) — best overall discrimination.

Bootstrap 95% CI computed over 2000 resamples (seed=42).

Results (2026-05-24, N=50)
--------------------------
  AUC green-vs-rest = 0.7576   _THRESH_GREEN = 0.39   CI [0.15, 0.47]
  AUC red-vs-rest   = 0.8503   _THRESH_RED   = 0.52   CI [0.47, 0.70]

  Corpus distribution with calibrated thresholds:
    GREEN=11 (22%)   YELLOW=14 (28%)   RED=25 (50%)
  Note: corpus is 100% dialectal — in mixed-population deployment
  GREEN% would increase significantly.

Usage
-----
    conda run -n parlachiaro python daws/study/threshold_calibration.py
"""

import json
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "outputs_geometrici" / "dataset_topologico_50.json"

H_MIN = 0.4320
H_MAX = 0.8452
H_RNG = H_MAX - H_MIN


def load_corpus() -> tuple[np.ndarray, np.ndarray]:
    data   = json.loads(DATA_PATH.read_text())
    h_risk = []
    damage = []
    for d in data:
        h_sp = float(d.get("h_spectral", 0.0))
        wer  = float(d.get("wer", 0.0))
        dpc1 = float(d.get("pca_local", {}).get("delta_pc1", 0.0))
        h_risk.append(float(np.clip((h_sp - H_MIN) / H_RNG, 0.0, 1.0)))
        damage.append(wer + dpc1)
    return np.array(h_risk), np.array(damage)


def roc_curve_manual(labels: np.ndarray, scores: np.ndarray):
    thresholds = np.sort(np.unique(scores))[::-1]
    tpr_list, fpr_list = [], []
    pos, neg = labels.sum(), (1 - labels).sum()
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tpr_list.append(tp / pos if pos > 0 else 0.0)
        fpr_list.append(fp / neg if neg > 0 else 0.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_trapz(fpr, tpr) -> float:
    idx = np.argsort(fpr)
    return float(np.trapezoid(tpr[idx], fpr[idx]))


def calibrate(h_risk: np.ndarray, damage: np.ndarray, n_boot: int = 2000, seed: int = 42):
    p33 = np.percentile(damage, 33)
    p66 = np.percentile(damage, 66)
    labels3 = np.where(damage < p33, 0, np.where(damage < p66, 1, 2))

    print(f"Damage  P33={p33:.4f}  P66={p66:.4f}")
    print(f"Labels  GREEN={( labels3==0).sum()}  YELLOW={(labels3==1).sum()}  RED={(labels3==2).sum()}\n")

    # ── GREEN threshold (sens >= 0.90 criterion) ──────────────────────────
    lab_g = (labels3 >= 1).astype(int)
    fpr_g, tpr_g, thr_g = roc_curve_manual(lab_g, h_risk)
    auc_g = auc_trapz(fpr_g, tpr_g)
    mask = tpr_g >= 0.90
    if mask.any():
        best_g = int(np.where(mask)[0][np.argmax((1 - fpr_g)[mask])])
    else:
        best_g = int(np.argmax(tpr_g - fpr_g))
    thresh_green = float(thr_g[best_g])

    print(f"GREEN threshold (sens≥0.90, max spec):")
    print(f"  AUC={auc_g:.4f}  H_risk*={thresh_green:.4f}  "
          f"sens={tpr_g[best_g]:.3f}  spec={1-fpr_g[best_g]:.3f}\n")

    # ── RED threshold (Youden's J) ────────────────────────────────────────
    lab_r = (labels3 == 2).astype(int)
    fpr_r, tpr_r, thr_r = roc_curve_manual(lab_r, h_risk)
    auc_r = auc_trapz(fpr_r, tpr_r)
    best_r = int(np.argmax(tpr_r - fpr_r))
    thresh_red = float(thr_r[best_r])

    print(f"RED threshold (Youden's J):")
    print(f"  AUC={auc_r:.4f}  H_risk*={thresh_red:.4f}  "
          f"sens={tpr_r[best_r]:.3f}  spec={1-fpr_r[best_r]:.3f}\n")

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    boot_g, boot_r = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels3), len(labels3))
        lg, lr, hr = lab_g[idx], lab_r[idx], h_risk[idx]
        try:
            fgb, tgb, thgb = roc_curve_manual(lg, hr)
            m = tgb >= 0.90
            bi = int(np.where(m)[0][np.argmax((1 - fgb)[m])]) if m.any() else int(np.argmax(tgb - fgb))
            boot_g.append(float(thgb[bi]))
        except Exception:
            pass
        try:
            frb, trb, thrb = roc_curve_manual(lr, hr)
            boot_r.append(float(thrb[int(np.argmax(trb - frb))]))
        except Exception:
            pass

    ci_g = np.percentile(boot_g, [2.5, 97.5])
    ci_r = np.percentile(boot_r, [2.5, 97.5])
    print(f"Bootstrap 95% CI (n={n_boot}, seed={seed}):")
    print(f"  GREEN: [{ci_g[0]:.4f}, {ci_g[1]:.4f}]")
    print(f"  RED:   [{ci_r[0]:.4f}, {ci_r[1]:.4f}]\n")

    # ── Corpus distribution ───────────────────────────────────────────────
    n_g = (h_risk < thresh_green).sum()
    n_y = ((h_risk >= thresh_green) & (h_risk < thresh_red)).sum()
    n_r = (h_risk >= thresh_red).sum()
    print(f"Corpus distribution with calibrated thresholds (N={len(h_risk)}):")
    print(f"  GREEN={n_g} ({100*n_g/len(h_risk):.0f}%)  "
          f"YELLOW={n_y} ({100*n_y/len(h_risk):.0f}%)  "
          f"RED={n_r} ({100*n_r/len(h_risk):.0f}%)\n")

    print("─" * 50)
    print(f"_THRESH_GREEN = {thresh_green:.2f}  (round: {round(thresh_green, 2)})")
    print(f"_THRESH_RED   = {thresh_red:.2f}  (round: {round(thresh_red, 2)})")
    print("─" * 50)

    return thresh_green, thresh_red, auc_g, auc_r, ci_g, ci_r


if __name__ == "__main__":
    print("=" * 50)
    print("PARLA CHIARO — H_risk Threshold Calibration")
    print("=" * 50 + "\n")
    h_risk, damage = load_corpus()
    calibrate(h_risk, damage)
