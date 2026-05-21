"""
study/llm_characterization.py
------------------------------
NeurIPS-style characterization plots for the Inv-Entropy study.

Four panels (matching the micropaper structure):

  Plot 1 — Scatter WER vs Inv-Entropy (main result)
            Colour: green if IE <= p66, red if IE > p66.
            Shows cross-modal error propagation.

  Plot 2 — Inv-Entropy distribution (histogram + KDE)
            Vertical lines at empirical percentiles 33 and 66.
            These percentiles become the calibrated risk thresholds.

  Plot 3 — Caso A vs Caso B: input cluster separation vs output cluster separation
            Points above y=x diagonal → Mistral amplified the error (Case B).
            Points below → Mistral absorbed the error (Case A).
            Cluster separation = 1 - cosine(centroid_GT, centroid_Whisper).

  Plot 4 — ROC curve for calibration
            Binary label: output_divergence > 0.3 → high risk.
            y_score = inv_entropy.
            Optimal threshold maximises F1.

Inputs: inv_entropy_study.json  (from --mode ie-study)
        LaBSE embeddings for Plot 3 (computed on the fly)

Usage:
    python main.py --mode characterize
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
IE_RESULTS = RESULTS_DIR / "inv_entropy_study.json"
FIGURE_PATH = RESULTS_DIR / "llm_characterization.pdf"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _setup_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })
    return plt


# ---------------------------------------------------------------------------
# Cluster separation (for Plot 3)
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def _compute_cluster_separations(
    records: list[dict],
) -> tuple[list[float], list[float]]:
    """
    For each recording compute:
      input_sep  = cosine distance between centroid([GT1,GT2,GT3]) and centroid([W1,W2,W3])
      output_sep = cosine distance between centroid([R_GT1,R_GT2,R_GT3]) and
                                          centroid([R_W1,R_W2,R_W3])

    output_sep requires the 6 LLM responses stored in the study results.
    If llm_responses are missing, falls back to h_norm as proxy.
    """
    from daws.pipeline.inv_entropy import LaBSEEmbedder

    # Check whether responses are stored
    has_responses = "llm_responses" in records[0] if records else False
    if not has_responses:
        logger.warning(
            "llm_responses not in study results — run ie-study with store_responses=True "
            "or rerun. Using h_norm as output_sep proxy."
        )

    embedder = LaBSEEmbedder()

    input_seps, output_seps = [], []

    for rec in records:
        gt = [rec.get(f"gt{i+1}", rec.get("gt1", "")) for i in range(3)]
        w = [rec.get(f"w{i+1}", rec.get("w1", "")) for i in range(3)]

        # Deduplicate (some recordings have only one GT variant)
        gt = [g if g else gt[0] for g in gt]
        w = [wi if wi else w[0] for wi in w]

        try:
            all_texts = gt + w
            embs = embedder.similarity_matrix(all_texts)  # Not needed; just embed
            from sentence_transformers import SentenceTransformer
            # Use LaBSE directly via the embedder
            e_all = embedder._model.encode(all_texts, normalize_embeddings=True)
            centroid_gt = e_all[:3].mean(axis=0)
            centroid_w = e_all[3:].mean(axis=0)
            input_sep = _cosine_distance(centroid_gt, centroid_w)
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
            input_sep = rec.get("wer", 0.0)

        if has_responses:
            resp = rec.get("llm_responses", [])
            if len(resp) == 6:
                try:
                    e_resp = embedder._model.encode(resp, normalize_embeddings=True)
                    centroid_r_gt = e_resp[:3].mean(axis=0)
                    centroid_r_w = e_resp[3:].mean(axis=0)
                    output_sep = _cosine_distance(centroid_r_gt, centroid_r_w)
                except Exception:
                    output_sep = rec.get("h_norm", 0.0)
            else:
                output_sep = rec.get("h_norm", 0.0)
        else:
            # Use h_norm as proxy for output divergence
            output_sep = rec.get("h_norm", rec.get("u_inv_entropy", 0.0))

        input_seps.append(input_sep)
        output_seps.append(output_sep)

    return input_seps, output_seps


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_all(records: list[dict], output_path: Path):
    """Generate all four characterization panels and save to PDF."""
    plt = _setup_style()
    import matplotlib.pyplot as mpl_plt
    import matplotlib.gridspec as gridspec

    wers = np.array([r["wer"] for r in records])
    h_norms = np.array([r.get("h_norm", r.get("u_inv_entropy", 0.0)) for r in records])
    dialects = [r.get("dialect", "") for r in records]

    p33 = float(np.percentile(h_norms, 33))
    p66 = float(np.percentile(h_norms, 66))

    fig = mpl_plt.figure(figsize=(7.0, 5.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Plot 1: Scatter WER vs Inv-Entropy ───────────────────────────────────
    colors = ["#e74c3c" if h > p66 else "#2ecc71" for h in h_norms]
    ax1.scatter(wers, h_norms, c=colors, s=22, alpha=0.75, linewidths=0)
    ax1.axhline(p66, color="#e74c3c", linestyle="--", linewidth=0.9, label=f"p66 = {p66:.3f}")
    ax1.axhline(p33, color="#f39c12", linestyle="--", linewidth=0.9, label=f"p33 = {p33:.3f}")

    # Linear regression
    if len(wers) >= 3:
        coef = np.polyfit(wers, h_norms, 1)
        xfit = np.linspace(wers.min(), wers.max(), 100)
        ax1.plot(xfit, np.polyval(coef, xfit), color="gray", linewidth=0.8, alpha=0.7)
        corr = float(np.corrcoef(wers, h_norms)[0, 1])
        ax1.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax1.transAxes, fontsize=7)

    ax1.set_xlabel("WER  (WhisperX vs GT)")
    ax1.set_ylabel("Inv-Entropy H̃")
    ax1.set_title("(1) Dialect error propagation")
    ax1.legend(frameon=False, fontsize=7)

    # ── Plot 2: Inv-Entropy distribution ─────────────────────────────────────
    try:
        import seaborn as sns
        sns.histplot(h_norms, kde=True, ax=ax2, color="#2980b9", alpha=0.6,
                     line_kws={"linewidth": 1.5})
    except ImportError:
        ax2.hist(h_norms, bins=15, color="#2980b9", alpha=0.6, density=True)

    ax2.axvline(p33, color="#f39c12", linestyle="--", linewidth=1.2,
                label=f"p33 = {p33:.3f}  (green/yellow)")
    ax2.axvline(p66, color="#e74c3c", linestyle="--", linewidth=1.2,
                label=f"p66 = {p66:.3f}  (yellow/red)")
    ax2.set_xlabel("Inv-Entropy H̃")
    ax2.set_ylabel("Density")
    ax2.set_title("(2) Empirical H̃ distribution")
    ax2.legend(frameon=False, fontsize=7)

    # ── Plot 3: Input cluster sep vs Output cluster sep ───────────────────────
    input_seps, output_seps = _compute_cluster_separations(records)
    input_seps = np.array(input_seps)
    output_seps = np.array(output_seps)

    # Points above diagonal → Case B (amplification)
    above = output_seps > input_seps
    ax3.scatter(input_seps[above], output_seps[above],
                color="#e74c3c", s=20, alpha=0.75, linewidths=0,
                label=f"Case B — amplif. ({above.sum()}/{len(above)})")
    ax3.scatter(input_seps[~above], output_seps[~above],
                color="#2ecc71", s=20, alpha=0.75, linewidths=0,
                label=f"Case A — robust. ({(~above).sum()}/{len(above)})")

    lim = max(input_seps.max(), output_seps.max()) * 1.05
    ax3.plot([0, lim], [0, lim], "--", color="gray", linewidth=0.8, label="y = x")
    ax3.set_xlim(0, lim); ax3.set_ylim(0, lim)
    ax3.set_xlabel("Input cluster separation")
    ax3.set_ylabel("Output cluster separation")
    ax3.set_title("(3) Error propagation (A vs B)")
    ax3.legend(frameon=False, fontsize=7)

    # ── Plot 4: ROC curve ─────────────────────────────────────────────────────
    # Binary label: output_sep > 0.3 → high risk
    output_div = output_seps
    y_true = (output_div > 0.3).astype(int)
    y_score = h_norms

    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        try:
            from sklearn.metrics import roc_curve, auc, f1_score

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            f1s = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0) for t in thresholds]
            best_idx = int(np.argmax(f1s))
            optimal_thresh = float(thresholds[best_idx])

            ax4.plot(fpr, tpr, color="#2980b9", linewidth=1.5, label=f"ROC (AUC={roc_auc:.2f})")
            ax4.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
            ax4.scatter([fpr[best_idx]], [tpr[best_idx]], color="#e74c3c", s=50, zorder=5,
                        label=f"Optimal F1 thresh={optimal_thresh:.3f}")
            ax4.set_xlabel("False Positive Rate")
            ax4.set_ylabel("True Positive Rate")
            ax4.set_title("(4) ROC — calibrated threshold")
            ax4.legend(frameon=False, fontsize=7)
            ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

        except ImportError:
            ax4.text(0.5, 0.5, "sklearn required for ROC", ha="center", va="center",
                     transform=ax4.transAxes, fontsize=8)
            optimal_thresh = p66
            roc_auc = float("nan")
    else:
        ax4.text(0.5, 0.5, "Insufficient class balance\nfor ROC", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=8)
        optimal_thresh = p66
        roc_auc = float("nan")

    pct_b = float(above.mean()) * 100

    fig.suptitle(
        f"DAWS — Mistral-7B Inv-Entropy Characterization  (n={len(records)}, "
        f"Case B={pct_b:.0f}%)",
        fontsize=9, y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    mpl_plt.close(fig)
    logger.info(f"Figure saved → {output_path}")
    print(f"Figure saved → {output_path}")

    return {
        "p33": p33, "p66": p66,
        "corr_wer_h": float(np.corrcoef(wers, h_norms)[0, 1]) if len(wers) >= 3 else float("nan"),
        "pct_case_b": pct_b,
        "optimal_thresh": optimal_thresh if "optimal_thresh" in dir() else p66,
        "roc_auc": roc_auc if "roc_auc" in dir() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_characterization(
    results_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Load ie-study results and generate the 4-panel characterization figure.

    Args:
        results_path: path to inv_entropy_study.json (default: daws/results/)
        output_path:  output figure path (PDF)

    Returns:
        Dict with p33, p66, correlation, pct_case_b, optimal_thresh, roc_auc
    """
    rpath = Path(results_path) if results_path else IE_RESULTS
    opath = Path(output_path) if output_path else FIGURE_PATH

    if not rpath.exists():
        raise FileNotFoundError(
            f"Study results not found: {rpath}\n"
            f"Run --mode ie-study first."
        )

    with open(rpath, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {rpath}")
    return plot_all(records, opath)
