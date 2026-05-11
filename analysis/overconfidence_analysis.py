"""
overconfidence_analysis.py
--------------------------
Novel contribution di PARLA CHIARO:

Ipotesi: quando Whisper trascrive dialetto napoletano producendo errori sistematici,
l'LLM risponde con semantic entropy *più bassa* (overconfidence) rispetto a quando
riceve la trascrizione corretta umana.

Questo è controintuitivo ma atteso: l'input distorto collassa il LLM su un'unica
interpretazione (quella "più vicina" in italiano standard), mentre l'input corretto
preserva l'ambiguità originale che l'LLM onestamente riconosce.

Quadrante di interesse (quello che vuoi flaggare nel DAWS):
  - SE_distorta < SE_corretta  AND  risposta_distorta != ground_truth
  → overconfidence vera: il modello è sicuro ma sbaglia

Design dell'analisi:
  Per ogni clip del corpus Cacioli 2026:
  1. Input A = trascrizione umana corretta
  2. Input B = trascrizione Whisper distorta  
  3. Genera risposta LLM per A e per B
  4. Calcola SE(A) e SE(B) via semantic entropy pipeline
  5. Confronto: SE(B) vs SE(A) — se sistematicamente SE(B) < SE(A) → overconfidence
  6. Test statistico: Wilcoxon signed-rank test (paired, non-parametric)
  7. Visualizzazione: scatter plot con diagonal reference + overconfidence quadrant
"""

import json
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from scipy import stats

from core.semantic_entropy import SemanticUncertaintyPipeline, ClaimUncertainty

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClipAnalysis:
    clip_id: str
    domain: str                    # play / poetry / blog (da Cacioli Tabella 1)
    human_transcript: str          # trascrizione umana di riferimento
    whisper_transcript: str        # output Whisper (distorto)
    
    # Risposta LLM per i due input
    llm_response_correct: str = ""
    llm_response_distorted: str = ""
    
    # Semantic entropy per i due input
    se_correct: float = 0.0        # SE su trascrizione corretta
    se_distorted: float = 0.0     # SE su trascrizione distorta
    n_clusters_correct: int = 0
    n_clusters_distorted: int = 0
    
    # Ground truth per valutare accuracy (opzionale)
    ground_truth: Optional[str] = None
    is_correct_on_distorted: Optional[bool] = None

    @property
    def delta_se(self) -> float:
        """SE_corretta - SE_distorta. Positivo → overconfidence su input distorto."""
        return self.se_correct - self.se_distorted

    @property
    def is_overconfident(self) -> bool:
        """
        Overconfidence vera: il modello è più sicuro sull'input distorto E sbaglia.
        """
        if self.is_correct_on_distorted is None:
            return self.delta_se > 0.1   # fallback: solo delta SE
        return self.delta_se > 0.1 and not self.is_correct_on_distorted


@dataclass
class OverconfidenceReport:
    n_clips: int
    n_overconfident: float          # % clip in quadrante overconfidence
    mean_se_correct: float
    mean_se_distorted: float
    mean_delta_se: float
    std_delta_se: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    effect_size_r: float            # r = Z / sqrt(N), da Wilcoxon
    domain_breakdown: dict[str, float]  # delta_se medio per dominio


# ──────────────────────────────────────────────────────────────────────────────
# Corpus loader (Cacioli 2026 — HuggingFace)
# ──────────────────────────────────────────────────────────────────────────────

class CacioliCorpusLoader:
    """
    Carica il Neapolitan-Spoken-Corpus da HuggingFace.
    
    Dataset: anonymous-nsc-author/Neapolitan-Spoken-Corpus
    Struttura attesa: {clip_id, domain, transcript, whisper_output, audio_path}
    
    Nota: il corpus ha 141 clip da un singolo speaker (limite dichiarato nel paper).
    Questo limita la generalizzabilità ma è sufficiente per la diagnostic evaluation.
    """

    DATASET_NAME = "anonymous-nsc-author/Neapolitan-Spoken-Corpus"

    def __init__(self, local_cache_path: Optional[Path] = None):
        self.local_cache = local_cache_path

    def load(self) -> list[dict]:
        """
        Ritorna lista di dizionari con almeno:
        {clip_id, domain, human_transcript, whisper_transcript}
        
        Se il dataset non è disponibile, usa i sample dal paper Cacioli (Tabella 2 + Appendice D).
        """
        try:
            from datasets import load_dataset
            ds = load_dataset(self.DATASET_NAME, split="train")
            return [self._normalize(row) for row in ds]
        except Exception as e:
            logger.warning(f"HuggingFace load fallita ({e}), uso sample dal paper.")
            return self._paper_samples()

    def _normalize(self, row: dict) -> dict:
        """Normalizza i campi del dataset verso la struttura attesa."""
        return {
            "clip_id": row.get("clip_id", row.get("id", "unknown")),
            "domain": row.get("domain", row.get("source", "unknown")),
            "human_transcript": row.get("transcript", row.get("reference", "")),
            "whisper_transcript": row.get("whisper_output", row.get("asr_output", "")),
        }

    def _paper_samples(self) -> list[dict]:
        """
        Sample hardcoded dal paper Cacioli (Tabella 2 + Appendice D).
        Usati come fallback / test unitario.
        """
        return [
            {
                "clip_id": "002",
                "domain": "play",
                "human_transcript": "E chesto capisce tu: 'e denare!",
                "whisper_transcript": "E questo capisce tu: i denari!",
            },
            {
                "clip_id": "003",
                "domain": "play",
                "human_transcript": "E cu' 'e denare t'he accattato tutto chello ca he voluto!",
                "whisper_transcript": "E con i soldi ti sei comprato tutto quello che hai voluto!",
            },
            {
                "clip_id": "err_001",
                "domain": "play",
                "human_transcript": "Qua sta 'a cena.",
                "whisper_transcript": "Castagena.",      # Appendice D: phonetic hallucination
            },
            {
                "clip_id": "err_002",
                "domain": "play",
                "human_transcript": "Aggio juto a Rroma l'at ajere.",
                "whisper_transcript": "Oggi sono andato a Roma ieri.",   # syntactic overcorrection
            },
            {
                "clip_id": "err_003",
                "domain": "play",
                "human_transcript": "Nun 'o voglio veré.",
                "whisper_transcript": "Non lo voglio vedere.",   # automatic Italianization
            },
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

class OverconfidenceAnalyzer:
    """
    Esegue l'overconfidence analysis sul corpus Cacioli.
    
    Per ogni clip:
    1. Costruisce un prompt contestuale (es. "L'utente dice: <transcript>. Rispondi.")
    2. Esegue SemanticUncertaintyPipeline su input corretto e distorto
    3. Confronta SE_corretto vs SE_distorto
    """

    # Prompt template per simulare il contesto assistente sanitario
    # (scenario più critico per PARLA CHIARO: healthcare assistant)
    ASSISTANT_PROMPT = """Sei un assistente vocale per servizi pubblici italiani.
L'utente ha detto: "{transcript}"
Rispondi in modo appropriato e accurato."""

    def __init__(
        self,
        pipeline: SemanticUncertaintyPipeline,
        output_dir: Path = Path("results/overconfidence"),
    ):
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_clip(self, clip: dict) -> ClipAnalysis:
        """Analizza una singola clip: corretto vs distorto."""
        analysis = ClipAnalysis(
            clip_id=clip["clip_id"],
            domain=clip["domain"],
            human_transcript=clip["human_transcript"],
            whisper_transcript=clip["whisper_transcript"],
        )

        # Prompt per trascrizione corretta
        prompt_correct = self.ASSISTANT_PROMPT.format(transcript=clip["human_transcript"])
        # Prompt per trascrizione distorta (Whisper output)
        prompt_distorted = self.ASSISTANT_PROMPT.format(transcript=clip["whisper_transcript"])

        logger.info(f"[{clip['clip_id']}] Analisi trascrizione corretta...")
        result_correct = self.pipeline.evaluate_claim(
            claim=clip["human_transcript"],
            context=prompt_correct,
        )

        logger.info(f"[{clip['clip_id']}] Analisi trascrizione distorta...")
        result_distorted = self.pipeline.evaluate_claim(
            claim=clip["whisper_transcript"],
            context=prompt_distorted,
        )

        analysis.se_correct = result_correct.semantic_entropy
        analysis.se_distorted = result_distorted.semantic_entropy
        analysis.n_clusters_correct = result_correct.n_distinct_meanings
        analysis.n_clusters_distorted = result_distorted.n_distinct_meanings
        analysis.llm_response_correct = result_correct.generations[0].text if result_correct.generations else ""
        analysis.llm_response_distorted = result_distorted.generations[0].text if result_distorted.generations else ""

        return analysis

    def run_full_analysis(self, clips: list[dict]) -> OverconfidenceReport:
        """Esegue l'analisi su tutto il corpus e produce il report."""
        results: list[ClipAnalysis] = []

        for i, clip in enumerate(clips):
            logger.info(f"Clip {i+1}/{len(clips)}: {clip['clip_id']}")
            try:
                analysis = self.analyze_clip(clip)
                results.append(analysis)
                # Salva checkpoint per evitare di perdere lavoro
                self._save_checkpoint(analysis)
            except Exception as e:
                logger.error(f"Errore su clip {clip['clip_id']}: {e}")
                continue

        report = self._compute_report(results)
        self._save_results(results, report)
        return report

    def _compute_report(self, results: list[ClipAnalysis]) -> OverconfidenceReport:
        """
        Calcola statistiche aggregate e test di Wilcoxon.
        
        H0: mediana(SE_corretto - SE_distorto) = 0
        H1: mediana > 0  (overconfidence sistematica su input distorto)
        Test: Wilcoxon signed-rank (one-tailed, paired, non-parametric)
        """
        se_correct = np.array([r.se_correct for r in results])
        se_distorted = np.array([r.se_distorted for r in results])
        deltas = se_correct - se_distorted

        # Wilcoxon signed-rank test (alternativa: t-test se distribuzione normale)
        stat, pvalue = stats.wilcoxon(deltas, alternative="greater")

        # Effect size r = Z / sqrt(N)
        n = len(deltas)
        z_approx = stats.norm.ppf(1 - pvalue / 2)  # approx Z score
        effect_size_r = abs(z_approx) / math.sqrt(n)

        # Breakdown per dominio
        domain_breakdown = {}
        for domain in set(r.domain for r in results):
            domain_deltas = [r.delta_se for r in results if r.domain == domain]
            domain_breakdown[domain] = float(np.mean(domain_deltas)) if domain_deltas else 0.0

        n_overconfident = sum(1 for r in results if r.is_overconfident)

        return OverconfidenceReport(
            n_clips=n,
            n_overconfident=n_overconfident / n if n > 0 else 0.0,
            mean_se_correct=float(np.mean(se_correct)),
            mean_se_distorted=float(np.mean(se_distorted)),
            mean_delta_se=float(np.mean(deltas)),
            std_delta_se=float(np.std(deltas)),
            wilcoxon_statistic=float(stat),
            wilcoxon_pvalue=float(pvalue),
            effect_size_r=float(effect_size_r),
            domain_breakdown=domain_breakdown,
        )

    def _save_checkpoint(self, analysis: ClipAnalysis):
        path = self.output_dir / f"clip_{analysis.clip_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)

    def _save_results(self, results: list[ClipAnalysis], report: OverconfidenceReport):
        with open(self.output_dir / "full_results.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        with open(self.output_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "play": "#E63946",
    "poetry": "#457B9D",
    "blog": "#2A9D8F",
    "unknown": "#6C757D",
}


def plot_overconfidence_scatter(
    results: list[ClipAnalysis],
    report: OverconfidenceReport,
    save_path: Optional[Path] = None,
):
    """
    Scatter plot SE_corretta (y) vs SE_distorta (x).
    
    Quadranti:
    - Alto-sinistra: SE_corretta > SE_distorta → overconfidence su input distorto ← CRITICO
    - Basso-destra:  SE_corretta < SE_distorta → underconfidence su input distorto
    - Diagonale: nessun effetto
    
    Colori per dominio (play/poetry/blog) come Cacioli Tabella 1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Scatter principale ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    ax.grid(True, alpha=0.3, linestyle="--")

    for result in results:
        color = DOMAIN_COLORS.get(result.domain, DOMAIN_COLORS["unknown"])
        marker = "^" if result.is_overconfident else "o"
        ax.scatter(
            result.se_distorted,
            result.se_correct,
            c=color,
            marker=marker,
            s=80,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
        )

    # Diagonal reference
    all_se = [r.se_correct for r in results] + [r.se_distorted for r in results]
    if all_se:
        lim = (min(all_se) - 0.05, max(all_se) + 0.05)
        ax.plot(lim, lim, "k--", alpha=0.4, linewidth=1.2, label="SE pari")
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    # Overconfidence region shading
    if all_se:
        ax.fill_between(
            [lim[0], lim[1]], [lim[1], lim[1]], [lim[0], lim[1]],
            alpha=0.06, color="#E63946", label="Overconfidence su distorto"
        )

    # Annotazioni
    ax.set_xlabel("SE su trascrizione Whisper (distorta)", fontsize=11)
    ax.set_ylabel("SE su trascrizione umana (corretta)", fontsize=11)
    ax.set_title(
        f"Overconfidence Analysis — PARLA CHIARO\n"
        f"Wilcoxon p={report.wilcoxon_pvalue:.4f}, r={report.effect_size_r:.3f}",
        fontsize=12
    )

    # Legend dominio
    patches = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLORS.items() if d != "unknown"]
    patches.append(plt.Line2D([0], [0], marker="^", color="gray", linestyle="None",
                               markersize=8, label="Overconfident clip"))
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    # ── Istogramma delta SE ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    ax2.grid(True, alpha=0.3, linestyle="--")

    deltas = [r.delta_se for r in results]
    ax2.hist(deltas, bins=15, color="#457B9D", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1.2)
    ax2.axvline(report.mean_delta_se, color="#E63946", linestyle="-",
                linewidth=2, label=f"Media Δ={report.mean_delta_se:.3f}")

    ax2.set_xlabel("ΔSE = SE_corretta − SE_distorta", fontsize=11)
    ax2.set_ylabel("Frequenza", fontsize=11)
    ax2.set_title(
        f"Distribuzione ΔSE\n"
        f"Media={report.mean_delta_se:.3f} ± {report.std_delta_se:.3f}",
        fontsize=12
    )
    ax2.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot salvato: {save_path}")

    return fig


def plot_domain_breakdown(results: list[ClipAnalysis], save_path: Optional[Path] = None):
    """Bar chart ΔSE per dominio (plays / poetry / blogs)."""
    domains = sorted(set(r.domain for r in results))
    means = []
    stds = []
    colors = []

    for d in domains:
        domain_deltas = [r.delta_se for r in results if r.domain == d]
        means.append(np.mean(domain_deltas))
        stds.append(np.std(domain_deltas))
        colors.append(DOMAIN_COLORS.get(d, DOMAIN_COLORS["unknown"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#F8F9FA")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    bars = ax.bar(domains, means, yerr=stds, color=colors, edgecolor="white",
                  capsize=5, error_kw={"elinewidth": 1.5})
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)

    ax.set_title("ΔSE per dominio testuale\n(positivo = overconfidence su input distorto)", fontsize=12)
    ax.set_ylabel("ΔSE medio (SE_corretta − SE_distorta)", fontsize=11)
    ax.set_xlabel("Dominio", fontsize=11)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
