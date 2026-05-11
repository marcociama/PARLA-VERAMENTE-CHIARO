"""
daws.py
-------
Dialect-Aware Warning System (DAWS) — Risk Scoring

Integra i segnali dell'ASR (WhisperX + OOV) con l'uncertainty quantification
(semantic entropy) per produrre un risk score per claim.

Schema di fusione:
  risk(claim) = α · SE_normalizzata(claim)
              + β · overlap_dialetto(claim, asr_output)
  
  con α=0.6, β=0.4 (pesi da calibrare su corpus annotato)
  
  dove overlap_dialetto = frazione di token del claim che cadono in segmenti
  flaggati come dialettali dall'ASR.

Output per claim:
  GREEN  (risk < 0.33): risposta affidabile
  YELLOW (0.33 ≤ risk < 0.66): richiedi chiarimento
  RED    (risk ≥ 0.66): trattieni risposta, chiedi riformulazione

MongoDB schema per storage trasversale:
  {
    session_id, timestamp, audio_path,
    asr_transcript, dialect_risk, dialect_tokens[],
    llm_response, claims[{text, se, risk_score, risk_level, clusters}],
    overall_risk, alert_triggered, clarification_question
  }
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from core.semantic_entropy import ClaimUncertainty
from core.whisperx_integration import ASROutput

logger = logging.getLogger(__name__)

# Pesi fusione (calibrabili)
ALPHA = 0.6   # peso uncertainty semantica
BETA = 0.4    # peso segnale dialettale ASR

# Soglie risk level
GREEN_THRESHOLD = 0.33
RED_THRESHOLD = 0.66

# Semantic entropy max teorica con M=10: log(10) ≈ 2.303
SE_MAX = 2.303


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClaimRisk:
    claim_text: str
    semantic_entropy: float
    se_normalized: float          # SE / log(M), ∈ [0, 1]
    asr_overlap: float            # frazione token claim che sono dialettali ∈ [0, 1]
    risk_score: float             # score fusione finale ∈ [0, 1]
    risk_level: str               # "green" / "yellow" / "red"
    n_clusters: int               # numero di significati distinti (Kuhn)
    justification: str            # spiegazione human-readable

    @classmethod
    def from_uncertainty(
        cls,
        uncertainty: ClaimUncertainty,
        asr_output: Optional[ASROutput] = None,
    ) -> "ClaimRisk":
        se = uncertainty.semantic_entropy
        se_norm = min(se / SE_MAX, 1.0)

        # Calcola overlap dialettale: token del claim che sono in segmenti flaggati
        asr_overlap = _compute_asr_overlap(uncertainty.claim_text, asr_output)

        # Fusione pesata
        risk_score = ALPHA * se_norm + BETA * asr_overlap

        risk_level = _classify_risk(risk_score)
        justification = _build_justification(se, se_norm, asr_overlap, uncertainty.n_distinct_meanings, risk_level)

        return cls(
            claim_text=uncertainty.claim_text,
            semantic_entropy=se,
            se_normalized=se_norm,
            asr_overlap=asr_overlap,
            risk_score=risk_score,
            risk_level=risk_level,
            n_clusters=uncertainty.n_distinct_meanings,
            justification=justification,
        )


@dataclass
class ResponseRisk:
    claims: list[ClaimRisk]
    overall_risk_score: float
    overall_risk_level: str
    alert_triggered: bool
    clarification_question: Optional[str] = None

    @property
    def red_claims(self) -> list[ClaimRisk]:
        return [c for c in self.claims if c.risk_level == "red"]

    @property
    def yellow_claims(self) -> list[ClaimRisk]:
        return [c for c in self.claims if c.risk_level == "yellow"]


# ──────────────────────────────────────────────────────────────────────────────
# DAWS core
# ──────────────────────────────────────────────────────────────────────────────

class DAWS:
    """
    Dialect-Aware Warning System.
    
    Integra claim-level risk + ASR signals → risposta sicura o alert.
    """

    def __init__(
        self,
        clarification_llm=None,   # OllamaLLM per generare domande chiarificatrici
        mongo_collection=None,     # pymongo Collection per logging
    ):
        self.clarification_llm = clarification_llm
        self.mongo = mongo_collection

    def assess_response(
        self,
        uncertainties: list[ClaimUncertainty],
        asr_output: Optional[ASROutput] = None,
        session_id: Optional[str] = None,
    ) -> ResponseRisk:
        """
        Valuta la risposta LLM claim-by-claim e produce il risk assessment.
        """
        claim_risks = [
            ClaimRisk.from_uncertainty(u, asr_output)
            for u in uncertainties
        ]

        overall_score = _compute_overall_risk(claim_risks)
        overall_level = _classify_risk(overall_score)
        alert = overall_level in ("yellow", "red")

        clarification = None
        if alert and self.clarification_llm:
            clarification = self._generate_clarification(claim_risks, asr_output)

        response_risk = ResponseRisk(
            claims=claim_risks,
            overall_risk_score=overall_score,
            overall_risk_level=overall_level,
            alert_triggered=alert,
            clarification_question=clarification,
        )

        if self.mongo and session_id:
            self._log_to_mongo(response_risk, asr_output, session_id)

        return response_risk

    def _generate_clarification(
        self,
        claims: list[ClaimRisk],
        asr_output: Optional[ASROutput],
    ) -> str:
        """
        Genera una domanda chiarificatrice contestuale.
        Priorità ai claim RED + alle zone dialettali identificate.
        """
        high_risk_claims = [c for c in claims if c.risk_level == "red"]
        if not high_risk_claims:
            high_risk_claims = [c for c in claims if c.risk_level == "yellow"]

        dialect_context = ""
        if asr_output and asr_output.dialect_risk_tokens:
            dialect_tokens = [t.token for t in asr_output.dialect_risk_tokens[:5]]
            dialect_context = f"Possibili termini dialettali: {', '.join(dialect_tokens)}."

        prompt = f"""Sei un assistente italiano che non ha capito bene la richiesta dell'utente
a causa di possibili espressioni dialettali o regionali.

{dialect_context}

Claims con alta incertezza:
{chr(10).join(f'- {c.claim_text}' for c in high_risk_claims[:3])}

Genera UNA sola domanda di chiarimento, concisa, in italiano standard,
che aiuti l'utente a riformulare la propria richiesta in modo più comprensibile.
La domanda deve essere cortese e non accusatoria.

Domanda:"""

        try:
            gens = self.clarification_llm.generate(prompt, n_samples=1, temperature=0.3)
            return gens[0].text if gens else "Potrebbe riformulare la sua richiesta?"
        except Exception as e:
            logger.error(f"Errore generazione chiarimento: {e}")
            return "Ho difficoltà a comprendere la richiesta. Potrebbe riformularla?"

    def _log_to_mongo(
        self,
        response_risk: ResponseRisk,
        asr_output: Optional[ASROutput],
        session_id: str,
    ):
        """Salva la sessione su MongoDB per auditing e analytics."""
        doc = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asr_transcript": asr_output.transcript if asr_output else "",
            "asr_dialect_risk": asr_output.overall_dialect_risk if asr_output else 0.0,
            "dialect_tokens": [t.token for t in asr_output.dialect_risk_tokens] if asr_output else [],
            "claims": [
                {
                    "text": c.claim_text,
                    "semantic_entropy": c.semantic_entropy,
                    "asr_overlap": c.asr_overlap,
                    "risk_score": c.risk_score,
                    "risk_level": c.risk_level,
                    "n_clusters": c.n_clusters,
                    "justification": c.justification,
                }
                for c in response_risk.claims
            ],
            "overall_risk_score": response_risk.overall_risk_score,
            "overall_risk_level": response_risk.overall_risk_level,
            "alert_triggered": response_risk.alert_triggered,
            "clarification_question": response_risk.clarification_question,
        }
        try:
            self.mongo.insert_one(doc)
        except Exception as e:
            logger.error(f"MongoDB insert failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _compute_asr_overlap(claim_text: str, asr_output: Optional[ASROutput]) -> float:
    """
    Frazione di token del claim che cadono in segmenti dialettali dell'ASR.
    
    Approccio: confronto string-based semplice (lowercase, strip punctuation).
    Per una versione production usare char-level alignment.
    """
    if not asr_output or not asr_output.tokens:
        return 0.0

    dialect_token_texts = {t.token.lower() for t in asr_output.dialect_risk_tokens}
    if not dialect_token_texts:
        return 0.0

    import re
    claim_tokens = re.findall(r'\w+', claim_text.lower())
    if not claim_tokens:
        return 0.0

    overlap_count = sum(1 for t in claim_tokens if t in dialect_token_texts)
    return overlap_count / len(claim_tokens)


def _classify_risk(score: float) -> str:
    if score < GREEN_THRESHOLD:
        return "green"
    elif score < RED_THRESHOLD:
        return "yellow"
    else:
        return "red"


def _compute_overall_risk(claims: list[ClaimRisk]) -> float:
    """
    Overall risk = max ponderato.
    Il worst-case domina: se un claim è RED, la risposta è RED.
    (conservativo per applicazioni healthcare)
    """
    if not claims:
        return 0.0
    scores = [c.risk_score for c in claims]
    # Pesato verso il max (conservativo)
    return 0.7 * max(scores) + 0.3 * (sum(scores) / len(scores))


def _build_justification(
    se: float,
    se_norm: float,
    asr_overlap: float,
    n_clusters: int,
    risk_level: str,
) -> str:
    """Genera spiegazione human-readable per il dashboard."""
    parts = []

    if se_norm > 0.5:
        parts.append(f"alta incertezza semantica (SE={se:.2f}, {n_clusters} interpretazioni distinte)")
    elif se_norm > 0.2:
        parts.append(f"incertezza semantica moderata (SE={se:.2f})")

    if asr_overlap > 0.3:
        parts.append(f"il {asr_overlap:.0%} dei token è flaggato come potenzialmente dialettale")
    elif asr_overlap > 0.1:
        parts.append(f"alcuni token potrebbero essere dialettali ({asr_overlap:.0%})")

    if not parts:
        return "Bassa incertezza, nessun segnale dialettale rilevato."

    prefix = {
        "green": "Nota:",
        "yellow": "Attenzione:",
        "red": "Alert:",
    }.get(risk_level, "")

    return f"{prefix} {'; '.join(parts)}."
