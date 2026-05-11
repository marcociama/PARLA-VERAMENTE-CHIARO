"""
token_heatmap.py
----------------
Uncertainty attribution a livello token per il dashboard Streamlit.

Idea: ogni token della risposta LLM riceve un "contribution score" all'entropia
del claim a cui appartiene. Questo permette la heatmap visualizzata nel dashboard.

Due livelli di granularità:
1. Claim-level: colore del claim in base alla sua semantic entropy
2. Token-level: all'interno del claim, peso di ogni token basato sulla sua
   varianza across le generazioni (quanto quel token cambia tra generazioni diverse)

Il secondo è una proxy per "token importance" — i token ad alta varianza
inter-generazione sono quelli su cui il modello è più incerto.
"""

import re
import difflib
import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.semantic_entropy import ClaimUncertainty


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenHeatEntry:
    token: str
    token_uncertainty: float     # ∈ [0, 1], normalizzato
    claim_entropy: float         # SE del claim padre
    
    @property
    def color_rgba(self) -> tuple[int, int, int, float]:
        """RGBA per rendering Streamlit/HTML. Scala rosso-giallo-verde."""
        u = self.token_uncertainty
        if u < 0.33:
            # Verde: bassa incertezza
            r, g, b = 76, 175, 80
        elif u < 0.66:
            # Giallo: incertezza media
            r, g, b = 255, 193, 7
        else:
            # Rosso: alta incertezza
            r, g, b = 229, 57, 53
        return r, g, b, 0.15 + 0.70 * u   # alpha cresce con incertezza


@dataclass
class ClaimHeatmap:
    claim_text: str
    semantic_entropy: float
    tokens: list[TokenHeatEntry]
    risk_level: str   # "green" / "yellow" / "red"

    @property
    def html_snippet(self) -> str:
        """Genera HTML con span colorati per rendering nel dashboard."""
        spans = []
        for t in self.tokens:
            r, g, b, a = t.color_rgba
            style = f"background-color: rgba({r},{g},{b},{a:.2f}); padding: 1px 2px; border-radius: 3px;"
            title = f"Incertezza token: {t.token_uncertainty:.3f} | SE claim: {t.claim_entropy:.3f}"
            spans.append(f'<span style="{style}" title="{title}">{t.token}</span>')
        return " ".join(spans)


# ──────────────────────────────────────────────────────────────────────────────
# Token uncertainty estimation
# ──────────────────────────────────────────────────────────────────────────────

class TokenUncertaintyEstimator:
    """
    Stima l'incertezza per ogni token del claim originale.
    
    Approccio: per ogni token nella risposta "gold" (prima generazione),
    conta quante volte quel token appare nelle altre generazioni
    alla stessa posizione (±2 token di finestra).
    
    Bassa frequenza = alta incertezza sul token.
    
    Nota: questo è una proxy, non un white-box approach.
    Per un approccio white-box vero si avrebbe bisogno dei logit del modello
    token-by-token, che Ollama non espone facilmente. WhisperX li espone
    per la parte ASR — per l'LLM usiamo questa proxy.
    """

    def estimate(
        self,
        claim: ClaimUncertainty,
        reference_generation: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """
        Ritorna lista di (token, uncertainty_score) per il reference_generation.
        uncertainty_score ∈ [0, 1].
        """
        if not claim.generations:
            return []

        ref = reference_generation or claim.generations[0].text
        ref_tokens = self._tokenize(ref)

        if not ref_tokens:
            return []

        # Raccoglie tutti i testi delle altre generazioni
        other_texts = [g.text for g in claim.generations[1:]]
        if not other_texts:
            # Se c'è solo 1 generazione, incertezza uniforme = 0.5
            return [(t, 0.5) for t in ref_tokens]

        # Per ogni token del reference, calcola la sua "stabilità" cross-generazioni
        token_uncertainty = []
        for i, token in enumerate(ref_tokens):
            presence_count = sum(
                1 for text in other_texts
                if self._token_present_near_position(token, i, text)
            )
            # Presenza alta → bassa incertezza
            stability = presence_count / len(other_texts)
            uncertainty = 1.0 - stability
            token_uncertainty.append((token, uncertainty))

        return token_uncertainty

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizzazione semplice word-level con punteggiatura separata."""
        # Regex che separa la punteggiatura dai token
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return [t for t in tokens if t.strip()]

    def _token_present_near_position(
        self,
        token: str,
        position: int,
        text: str,
        window: int = 3,
    ) -> bool:
        """
        Verifica se il token appare entro ±window posizioni nel testo.
        Usa lowercase per confronto case-insensitive.
        """
        other_tokens = self._tokenize(text)
        start = max(0, position - window)
        end = min(len(other_tokens), position + window + 1)
        window_tokens = [t.lower() for t in other_tokens[start:end]]
        return token.lower() in window_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap builder
# ──────────────────────────────────────────────────────────────────────────────

# Soglie per risk level (calibrate su Kuhn: SE_max = log(M) con M=10 → ~2.30)
ENTROPY_THRESHOLDS = {
    "green": 0.5,    # SE < 0.5 → bassa incertezza, risposta affidabile
    "yellow": 1.2,   # 0.5 ≤ SE < 1.2 → rischio medio
    # SE ≥ 1.2 → red
}


def classify_risk(se: float) -> str:
    if se < ENTROPY_THRESHOLDS["green"]:
        return "green"
    elif se < ENTROPY_THRESHOLDS["yellow"]:
        return "yellow"
    else:
        return "red"


class HeatmapBuilder:
    """
    Converte una lista di ClaimUncertainty in ClaimHeatmap per il dashboard.
    """

    def __init__(self):
        self.estimator = TokenUncertaintyEstimator()

    def build(self, claim: ClaimUncertainty) -> ClaimHeatmap:
        token_uncertainties = self.estimator.estimate(claim)

        entries = []
        for token, uncertainty in token_uncertainties:
            entries.append(TokenHeatEntry(
                token=token,
                token_uncertainty=uncertainty,
                claim_entropy=claim.semantic_entropy,
            ))

        return ClaimHeatmap(
            claim_text=claim.claim_text,
            semantic_entropy=claim.semantic_entropy,
            tokens=entries,
            risk_level=classify_risk(claim.semantic_entropy),
        )

    def build_all(self, claims: list[ClaimUncertainty]) -> list[ClaimHeatmap]:
        return [self.build(c) for c in claims]

    def render_html(self, heatmaps: list[ClaimHeatmap]) -> str:
        """
        Genera HTML completo per embedding in Streamlit via st.markdown(unsafe_allow_html=True).
        """
        risk_badge = {
            "green": '<span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:12px;font-size:0.8em;">✓ Affidabile</span>',
            "yellow": '<span style="background:#FFC107;color:black;padding:2px 8px;border-radius:12px;font-size:0.8em;">⚠ Rischio medio</span>',
            "red": '<span style="background:#E53935;color:white;padding:2px 8px;border-radius:12px;font-size:0.8em;">✗ Alto rischio</span>',
        }

        sections = []
        for hm in heatmaps:
            badge = risk_badge.get(hm.risk_level, "")
            se_str = f"SE={hm.semantic_entropy:.3f}"
            sections.append(
                f'<div style="margin-bottom:12px;padding:8px;border-left:4px solid '
                f'{"#4CAF50" if hm.risk_level == "green" else "#FFC107" if hm.risk_level == "yellow" else "#E53935"}'
                f';background:#FAFAFA;border-radius:4px;">'
                f'<div style="margin-bottom:4px;">{badge} <small style="color:#666">{se_str}</small></div>'
                f'<div>{hm.html_snippet}</div>'
                f'</div>'
            )

        return "\n".join(sections)
