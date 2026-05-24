"""
daws/pipeline/daws.py
-----------------
DAWS Online Inference Pipeline — 1D Markov Spettrale (Laplace k=1)
Ablation study definitivo: 21 Maggio 2026, N=50, corpus PARLA CHIARO.

Tabella Pearson certificata (§4.5 DAWS_ablation_spectral_2026-05-21.md):

  Metodo                              WER      E_sem_top  E_sem_cross  GT@runtime
  ─────────────────────────────────────────────────────────────────────────────────
  1D Markov Spettrale ONLINE (k=1)  +0.5010   +0.5688    +0.5684      NO  ← PROD
  VN H_hybrid (RBF)                 +0.4590   +0.5530    +0.5770      NO  ← backup
  Inv-Entropy H_k1 (NeurIPS 2025)   +0.5226   +0.4947    +0.6664      SÌ  ← UB offline
  Inv-Entropy H_k4 (NeurIPS 2025)   +0.5150   +0.6070    +0.6680      SÌ  ← UB offline
  ─────────────────────────────────────────────────────────────────────────────────
  1D Markov Spettrale (k=4 in 1D)  −0.2160   −0.3120    −0.2170      —   ← ESCLUSO

  1D Markov ONLINE vs Inv-Entropy H_k1 su E_sem_top: +0.5688 vs +0.4947 → +11 pp.
  k=4 in 1D distrugge il segnale (correlazione NEGATIVA) — vizio geometrico §4.4.

Runtime flow (T=0 per determinismo):
  1. Audio → WhisperX → W1 (greedy) + W2/W3 (Gaussian acoustic perturbations)
  2. U_ASR = 1 − mean(WhisperX word-level confidence for W1)
  3. W1, W2, W3 → Ollama Mistral T=0 → R_W1, R_W2, R_W3
  4. SBERT encode [W1,W2,W3] → project onto w_input_drift → p_in = [p1,p2,p3]
     SBERT encode [R_W1,R_W2,R_W3] → project onto w_resp_drift → s_out = [s1,s2,s3]
  5. Assemble 6-point vectors with frozen GT anchors:
       pts_in  = anchors_in  + p_in   (GT anchors slots 0-2, live inputs slots 3-5)
       pts_out = anchors_out + s_out  (GT anchors slots 0-2, live outputs slots 3-5)
  6. 1D Markov Spettrale: Px = row_stochastic(Laplace(pts_in, sigma_in))
                           Py = row_stochastic(Laplace(pts_out, sigma_out))
                           P_comb = Py @ Px
                           H_spectral = Shannon(|eigvals(P_comb)| / sum)  [nats]
  7. H_risk = clip((H_spectral − H_MIN) / (H_MAX − H_MIN), 0, 1)
     empirical bounds N=50: H_MIN=0.4320, H_MAX=0.8452
  8. Risk level: GREEN if H_risk < 0.39 | YELLOW if H_risk < 0.52 | RED otherwise
     thresholds derived via 3-class ROC on damage=(wer+δPC1) P33/P66, N=50
     AUC_green=0.76 (sens≥0.90 criterion)  AUC_red=0.85 (Youden's J)
  9. If RED: Mistral generates clarifying question (Italian)
 10. Persist inference log to MongoDB via daws.database.DAWSRepository
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests

log = logging.getLogger(__name__)

BASE     = Path(__file__).parent.parent.parent
CFG_PATH = BASE / "config" / "geometry_calibration.json"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Empirical H_spectral bounds — N=50 corpus PARLA CHIARO (ablation 2026-05-21)
H_SPECTRAL_MIN: float = 0.4320   # nats
H_SPECTRAL_MAX: float = 0.8452   # nats
_H_RANGE: float = H_SPECTRAL_MAX - H_SPECTRAL_MIN  # 0.4132 nats

# H_risk triage thresholds — calibrated via 3-class ROC on N=50 PARLA CHIARO corpus
# Label: damage = wer + δPC1, P33/P66 split → GREEN / YELLOW / RED
# GREEN: AUC=0.76, sens≥0.90 criterion  |  RED: AUC=0.85, Youden's J
# Bootstrap 95% CI: GREEN [0.15, 0.47]  |  RED [0.47, 0.70]
# See daws/study/threshold_calibration.py for full derivation.
_THRESH_GREEN: float = 0.39
_THRESH_RED:   float = 0.52

_MEDICAL_PROMPT = (
    "Sei un assistente medico. Rispondi brevemente in italiano alla seguente "
    "dichiarazione del paziente in massimo 2 frasi.\n\n"
    "Paziente: {text}\n\nRisposta:"
)
_CLARIFY_PROMPT = (
    "Sei un assistente medico che non ha capito chiaramente la dichiarazione di un "
    "paziente. Genera UNA sola domanda di chiarimento in italiano standard, breve e "
    "cortese, per chiedere al paziente di riformulare.\n\n"
    "Dichiarazione: {text}\n\nDomanda di chiarimento:"
)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    transcript:             str
    words:                  list[dict] = field(default_factory=list)
    llm_response:           str = ""
    u_asr:                  float = 0.0
    u_llm:                  float = 0.0      # H_spectral raw (nats)
    u_pipeline:             float = 0.0      # = H_spectral (backward compat)
    risk_level:             str  = "green"   # "green" | "yellow" | "red"
    clarification_question: Optional[str] = None
    h_spectral:             float = 0.0      # 1D Markov Spettrale entropy (nats)
    h_risk:                 float = 0.0      # clip((H_spectral−H_MIN)/(H_MAX−H_MIN), 0, 1)
    s_w:                    list[float] = field(default_factory=list)  # [s1,s2,s3] output projs
    processing_time_s:      float = 0.0


# ── 1D Markov Spettrale helpers ────────────────────────────────────────────────

def _laplacian(pts: np.ndarray, sigma: float) -> np.ndarray:
    diff = pts[:, None] - pts[None, :]
    return np.exp(-np.abs(diff) / sigma)


def _row_stochastic(A: np.ndarray) -> np.ndarray:
    s = A.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0
    return A / s


def _spectral_H(pts_in: np.ndarray, pts_out: np.ndarray,
                sigma_in: float, sigma_out: float) -> float:
    """1D Markov Spettrale: |eigvals(Py@Px)| normalizzati → Shannon entropy."""
    px   = _row_stochastic(_laplacian(pts_in,  sigma_in))
    py   = _row_stochastic(_laplacian(pts_out, sigma_out))
    eigs = np.abs(np.linalg.eigvals(py @ px))
    total = eigs.sum()
    if total < 1e-12:
        return 0.0
    e = eigs / total
    e = e[e > 1e-12]
    return float(-np.sum(e * np.log(e)))


# ── Ollama helper ──────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, temperature: float = 0.0, timeout: int = 300) -> str:
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 120, "seed": 0},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip().split("\n")[0].strip()


# ── Main pipeline class ────────────────────────────────────────────────────────

class DAWSPipeline:
    """
    DAWS online inference pipeline.

    Parameters
    ----------
    use_mongo : bool
        If True, persist inference records to MongoDB (graceful no-op if offline).
    asr_cache_dir : str | None
        Optional path to ASR cache directory (speeds up repeated audio files).
    """

    def __init__(
        self,
        use_mongo: bool = True,
        asr_cache_dir: Optional[str] = None,
    ):
        self._asr   = None
        self._sbert = None
        self._repo  = None
        self._cfg   = None
        self._use_mongo    = use_mongo
        self._asr_cache_dir = Path(asr_cache_dir) if asr_cache_dir else None

    # ── resource loading ───────────────────────────────────────────────────────

    def _load_cfg(self):
        if self._cfg is not None:
            return
        if not CFG_PATH.exists():
            raise FileNotFoundError(
                f"Calibration not found: {CFG_PATH}\n"
                "Run: python scripts/benchmark_final.py"
            )
        raw = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        self._cfg = {
            "w_resp_drift":  np.array(raw["w_resp_drift"],  dtype=np.float32),
            "w_input_drift": np.array(raw["w_input_drift"], dtype=np.float32),
            "sigma_out":     float(raw["sigma_out"]),
            "sigma_in":      float(raw["sigma_in"]),
            "anchors_out":   list(raw["anchors_out"]),
            "anchors_in":    list(raw["anchors_in"]),
            "mu_gt":         [raw["mu_R_GT1"], raw["mu_R_GT2"], raw["mu_R_GT3"]],
            "threshold_green": float(raw["threshold_green"]),
            "threshold_red":   float(raw["threshold_red"]),
        }
        log.info("Geometry calibration loaded.")

    def _load_sbert(self):
        if self._sbert is not None:
            return
        from sentence_transformers import SentenceTransformer
        log.info("Loading SBERT ...")
        self._sbert = SentenceTransformer(
            "paraphrase-multilingual-mpnet-base-v2", device="mps"
        )

    def _load_asr(self):
        if self._asr is not None:
            return
        from daws.pipeline.asr import ASRPipeline
        log.info("Loading WhisperX ASR ...")
        self._asr = ASRPipeline()

    def _load_repo(self):
        if self._repo is not None or not self._use_mongo:
            return
        from daws.database.repository import DAWSRepository
        self._repo = DAWSRepository()

    # ── public API ─────────────────────────────────────────────────────────────

    def process_audio(self, audio_path: str) -> InferenceResult:
        """Full DAWS inference on a WAV file."""
        t0 = time.time()
        self._load_cfg()
        self._load_sbert()
        self._load_asr()
        self._load_repo()

        cfg = self._cfg

        # ── Step 1: ASR → W1, W2, W3 ──────────────────────────────────────
        log.info(f"ASR: {audio_path}")
        cache_path = None
        if self._asr_cache_dir:
            stem = Path(audio_path).stem
            cache_path = str(self._asr_cache_dir / f"{stem}.json")
        asr_out = self._asr.process(audio_path, cache_path=cache_path)
        w1 = asr_out.top3[0] if asr_out.top3 else asr_out.transcription
        w2 = asr_out.top3[1] if len(asr_out.top3) > 1 else w1
        w3 = asr_out.top3[2] if len(asr_out.top3) > 2 else w1
        words = [
            {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
            for w in asr_out.words
        ]

        # ── Step 2: U_ASR ──────────────────────────────────────────────────
        u_asr = float(asr_out.u_asr)

        # ── Step 3: LLM responses R_W1, R_W2, R_W3 (T=0) ─────────────────
        log.info("Generating LLM responses via Ollama ...")
        r_w = []
        for w_text in [w1, w2, w3]:
            r_w.append(_ollama_generate(_MEDICAL_PROMPT.format(text=w_text), temperature=0.0))
        primary_response = r_w[0]

        # ── Step 4: SBERT encode inputs + outputs → 1D projections ────────
        embs_in  = self._sbert.encode([w1, w2, w3], normalize_embeddings=False)
        embs_out = self._sbert.encode(r_w,          normalize_embeddings=False)
        p_in  = (embs_in  @ cfg["w_input_drift"]).tolist()   # [p1, p2, p3]
        s_out = (embs_out @ cfg["w_resp_drift"] ).tolist()   # [s1, s2, s3]

        # ── Step 5: Assembla vettori con ancoraggi congelati ───────────────
        pts_in  = np.array(cfg["anchors_in"]  + p_in )   # (6,)
        pts_out = np.array(cfg["anchors_out"] + s_out)   # (6,)

        # ── Step 6: 1D Markov Spettrale → H_spectral ──────────────────────
        h_spectral = _spectral_H(pts_in, pts_out, cfg["sigma_in"], cfg["sigma_out"])

        # ── Step 7: H_risk normalizzato (bounds empirici N=50) ────────────
        u_pipe = h_spectral
        h_risk = float(np.clip(
            (h_spectral - H_SPECTRAL_MIN) / max(_H_RANGE, 1e-12),
            0.0, 1.0,
        ))

        # ── Step 8: Risk level — soglie su H_risk [0,1] ────────────────────
        if h_risk < _THRESH_GREEN:
            risk = "green"
        elif h_risk < _THRESH_RED:
            risk = "yellow"
        else:
            risk = "red"

        # ── Step 9: Clarifying question if RED ────────────────────────────
        clarification = None
        if risk == "red":
            log.info("Risk=RED — generating clarifying question ...")
            clarification = _ollama_generate(
                _CLARIFY_PROMPT.format(text=w1), temperature=0.0
            )

        elapsed = time.time() - t0

        result = InferenceResult(
            transcript=w1,
            words=words,
            llm_response=primary_response,
            u_asr=u_asr,
            u_llm=h_spectral,
            u_pipeline=u_pipe,
            risk_level=risk,
            clarification_question=clarification,
            h_spectral=h_spectral,
            h_risk=h_risk,
            s_w=s_out,
            processing_time_s=elapsed,
        )

        # ── Step 10: Persist to MongoDB ────────────────────────────────────
        if self._repo:
            self._repo.log_inference({
                "audio_path":          audio_path,
                "transcript":          w1,
                "u_asr":               u_asr,
                "u_llm":               h_spectral,
                "u_pipeline":          u_pipe,
                "risk_level":          risk,
                "llm_response":        primary_response,
                "clarifying_question": clarification,
            })

        log.info(
            f"Done in {elapsed:.1f}s | U_ASR={u_asr:.3f} H_spectral={h_spectral:.4f} "
            f"H_risk={h_risk:.3f} → {risk.upper()}"
        )
        return result
