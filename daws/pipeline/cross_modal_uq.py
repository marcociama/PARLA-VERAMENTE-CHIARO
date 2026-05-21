"""
pipeline/cross_modal_uq.py
--------------------------
Cross-modal uncertainty propagation.

Formula:
  U_pipeline = alpha * U_ASR + beta * U_LLM + gamma * U_ASR * U_LLM

  U_ASR  = 1 - mean_WhisperX_word_confidence
  U_LLM  = 1 - SD                             (Semantic Density)
  gamma term models the non-linear amplification when ASR errors
  propagate through the LLM (compounding uncertainty).

Default weights (calibratable via study/calibration.py):
  alpha=0.3, beta=0.5, gamma=0.2   (sum=1.0 → U_pipeline ∈ [0, 1])

Risk levels:
  GREEN  — U_pipeline < threshold_green    (reliable)
  YELLOW — threshold_green ≤ U < threshold_red  (caution)
  RED    — U_pipeline ≥ threshold_red      (trigger clarification)

Weights and thresholds are loaded from config/thresholds.json at init.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA = 0.3
_DEFAULT_BETA = 0.5
_DEFAULT_GAMMA = 0.2
_DEFAULT_THRESHOLD_GREEN = 0.35
_DEFAULT_THRESHOLD_RED = 0.65

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "thresholds.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CrossModalUQResult:
    u_asr: float
    u_llm: float
    u_pipeline: float
    risk_level: str        # "GREEN" | "YELLOW" | "RED"
    alpha: float
    beta: float
    gamma: float
    threshold_green: float
    threshold_red: float

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "RED"

    @property
    def needs_clarification(self) -> bool:
        return self.risk_level in ("YELLOW", "RED")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read thresholds.json: {e} — using defaults")
    return {}


# ---------------------------------------------------------------------------
# Cross-modal UQ
# ---------------------------------------------------------------------------

class CrossModalUQ:
    """Computes U_pipeline and classifies risk level."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        threshold_green: Optional[float] = None,
        threshold_red: Optional[float] = None,
    ):
        cfg = _load_config()
        self.alpha = alpha if alpha is not None else cfg.get("alpha", _DEFAULT_ALPHA)
        self.beta = beta if beta is not None else cfg.get("beta", _DEFAULT_BETA)
        self.gamma = gamma if gamma is not None else cfg.get("gamma", _DEFAULT_GAMMA)
        self.threshold_green = threshold_green if threshold_green is not None else cfg.get("threshold_green", _DEFAULT_THRESHOLD_GREEN)
        self.threshold_red = threshold_red if threshold_red is not None else cfg.get("threshold_red", _DEFAULT_THRESHOLD_RED)

    def compute(self, u_asr: float, u_llm: float) -> CrossModalUQResult:
        u_asr = max(0.0, min(1.0, u_asr))
        u_llm = max(0.0, min(1.0, u_llm))

        u_pipeline = (
            self.alpha * u_asr
            + self.beta * u_llm
            + self.gamma * u_asr * u_llm
        )
        u_pipeline = max(0.0, min(1.0, u_pipeline))

        if u_pipeline < self.threshold_green:
            risk_level = "GREEN"
        elif u_pipeline < self.threshold_red:
            risk_level = "YELLOW"
        else:
            risk_level = "RED"

        logger.debug(
            f"U_ASR={u_asr:.3f} | U_LLM={u_llm:.3f} | "
            f"U_pipeline={u_pipeline:.3f} | {risk_level}"
        )

        return CrossModalUQResult(
            u_asr=u_asr,
            u_llm=u_llm,
            u_pipeline=u_pipeline,
            risk_level=risk_level,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            threshold_green=self.threshold_green,
            threshold_red=self.threshold_red,
        )

    def save(self, path: Optional[Path] = None):
        """Persist current weights and thresholds to config/thresholds.json."""
        path = path or _CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
                "threshold_green": self.threshold_green, "threshold_red": self.threshold_red,
            }, f, indent=2)
        logger.info(f"Thresholds saved → {path}")
