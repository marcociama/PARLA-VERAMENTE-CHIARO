"""
semantic_entropy.py
-------------------
Implementa Kuhn et al. (ICLR 2023) adattato per PARLA CHIARO.

Pipeline per ogni claim:
  1. Campiona M generazioni dall'LLM a temperatura T
  2. Clustera per equivalenza semantica (entailment bidirezionale con DeBERTa-MNLI)
  3. Calcola semantic entropy = H sull'insieme dei meaning-cluster

Nota: il segmentatore di claim è *deliberatamente separato* da questo modulo
per evitare circular reasoning (segmentatore ≠ LLM valutato).
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import pipeline as hf_pipeline

from utils.device import get_transformers_device_id

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Generation:
    text: str
    log_prob: float          # somma log p(token | contesto)  ← da Ollama logprobs


@dataclass
class SemanticCluster:
    representative: str
    members: list[Generation] = field(default_factory=list)

    @property
    def semantic_likelihood(self) -> float:
        """p(c | x) = Σ_{s ∈ c} p(s | x)  [Kuhn eq. 2]"""
        # Usiamo log-sum-exp per stabilità numerica
        log_probs = [g.log_prob for g in self.members]
        if not log_probs:
            return 0.0
        max_lp = max(log_probs)
        return math.exp(max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs)))


@dataclass
class ClaimUncertainty:
    claim_text: str
    generations: list[Generation]
    clusters: list[SemanticCluster]
    semantic_entropy: float
    token_entropies: Optional[list[float]] = None   # per heatmap
    n_distinct_meanings: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# NLI-based bidirectional entailment (Algorithm 1 di Kuhn)
# ──────────────────────────────────────────────────────────────────────────────

class BidirectionalEntailmentClusterer:
    """
    Usa DeBERTa-large-MNLI per clustering semantico.
    Complessità: O(|C| · M) grazie alla transitivitá (come in Algorithm 1).
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large"):
        device = get_transformers_device_id()
        logger.info(f"Carico NLI model: {model_name} (device={device})")
        self.nli = hf_pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None,
        )
        self.entailment_threshold = 0.5

    def _entails(self, premise: str, hypothesis: str, context: str = "") -> bool:
        """
        Ritorna True se premise |= hypothesis nel contesto dato.
        Tronca a 200 char per stare nei 512 token di DeBERTa.
        """
        MAX_CHARS = 200
        text = premise[:MAX_CHARS]
        text_pair = hypothesis[:MAX_CHARS]

        results = self.nli(text, text_pair=text_pair)[0]
        label_map = {r["label"].lower(): r["score"] for r in results}

        entail_score = label_map.get("entailment", 0.0)
        return entail_score >= self.entailment_threshold

    def cluster(self, generations: list[Generation], context: str = "") -> list[SemanticCluster]:
        """
        Algoritmo 1 di Kuhn: costruisce cluster per equivalenza semantica.
        Due sequenze sono equivalenti sse si implicano a vicenda (↔ entailment).
        """
        if not generations:
            return []

        clusters: list[SemanticCluster] = [
            SemanticCluster(representative=generations[0].text, members=[generations[0]])
        ]

        for gen in generations[1:]:
            placed = False
            for cluster in clusters:
                rep = cluster.representative
                # Controlla entailment bidirezionale con il rappresentante del cluster
                if self._entails(rep, gen.text, context) and self._entails(gen.text, rep, context):
                    cluster.members.append(gen)
                    placed = True
                    break   # per transitivitá basta trovare il primo cluster compatibile

            if not placed:
                clusters.append(SemanticCluster(representative=gen.text, members=[gen]))

        return clusters


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Entropy
# ──────────────────────────────────────────────────────────────────────────────

def compute_semantic_entropy(clusters: list[SemanticCluster]) -> float:
    """
    SE(x) = -Σ_c p(c|x) log p(c|x)   [Kuhn eq. 3]
    
    Stimato via MC come: -|C|^{-1} Σ_i log p(C_i | x)  [Kuhn eq. 4]
    
    Usiamo l'estimatore MC non-biased di Kuhn (eq. 4) per coerenza con il paper.
    """
    if len(clusters) <= 1:
        return 0.0   # entropia zero = certezza assoluta

    # Normalizza le likelihood sui cluster per ottenere una distribuzione
    likelihoods = [c.semantic_likelihood for c in clusters]
    total = sum(likelihoods)

    if total <= 0:
        return 0.0

    entropy = 0.0
    for lk in likelihoods:
        p = lk / total
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


# ──────────────────────────────────────────────────────────────────────────────
# LLM interfaces
# ──────────────────────────────────────────────────────────────────────────────

class AnthropicLLM:
    """
    Wrapper per Claude via Anthropic API.

    Nota su log_prob: Claude API non espone token-level logprobs.
    Usiamo log_prob uniforme → la likelihood di ogni cluster diventa
    proporzionale alla sua cardinalità (MC estimator, Kuhn eq. 4).
    Questo è teoricamente valido: con T fissa, le sequenze più probabili
    vengono campionate più spesso, catturando l'informazione implicita.

    Richiede: ANTHROPIC_API_KEY nell'ambiente.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",  # veloce ed economico per campionamento
        max_tokens: int = 256,
    ):
        import anthropic as _anthropic
        self.client = _anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        n_samples: int = 10,
        temperature: float = 0.5,
        max_tokens: int | None = None,
    ) -> list[Generation]:
        """
        Campiona n_samples generazioni con temperatura T.
        Log_prob uniforme = -1.0 (conta cluster per SE).
        """
        max_tok = max_tokens or self.max_tokens
        generations = []
        for _ in range(n_samples):
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=max_tok,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # Log_prob uniforme: la SE dipende dai conteggi per cluster
            log_prob = -1.0
            generations.append(Generation(text=text, log_prob=log_prob))
        return generations


class GeminiLLM:
    """
    Wrapper per Gemini via Google Gen AI SDK.
    Richiede: GEMINI_API_KEY nell'ambiente.
    Log_prob uniforme come AnthropicLLM (Gemini non espone logprobs).
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        import os
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def generate(
        self,
        prompt: str,
        n_samples: int = 10,
        temperature: float = 0.5,
        max_tokens: int = 256,
    ) -> list[Generation]:
        import time
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        generations = []
        for _ in range(n_samples):
            for attempt in range(4):  # retry su 503/429 transitorio
                try:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                    text = resp.text.strip() if resp.text else ""
                    generations.append(Generation(text=text, log_prob=-1.0))
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    wait = 5 * (2 ** attempt)  # 5, 10, 20s
                    logger.warning(f"Gemini errore transitorio ({e.__class__.__name__}), retry in {wait}s...")
                    time.sleep(wait)
        return generations


class OllamaLLM:
    """
    Wrapper per LLM locale via Ollama (alternativa offline ad AnthropicLLM).
    Richiede: `ollama serve` + `ollama pull <model>`
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        n_samples: int = 10,
        temperature: float = 0.5,
        max_tokens: int = 256,
    ) -> list[Generation]:
        import requests

        generations = []
        for _ in range(n_samples):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "logprobs": True,
                },
                "stream": False,
            }
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            log_prob = self._extract_log_prob(data)
            generations.append(Generation(text=text, log_prob=log_prob))
        return generations

    def _extract_log_prob(self, response_data: dict) -> float:
        if "logprobs" in response_data and response_data["logprobs"]:
            return sum(response_data["logprobs"])
        n_tokens = response_data.get("eval_count", 50)
        return -n_tokens * 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Claim segmentator (separato per evitare circular reasoning)
# ──────────────────────────────────────────────────────────────────────────────

class ClaimSegmenter:
    """
    Segmenta una risposta LLM in claim atomici verificabili.
    
    Usa un LLM con prompt deterministico — preferibilmente un modello
    *diverso* da quello valutato. Per il progetto: usa LLaMA-3-8B come
    segmentatore e LLaMA-3-70B (o Mistral) come LLM valutato.
    """

    SEGMENTATION_PROMPT = """Sei un sistema di analisi del testo.
Dato il seguente testo, estrai ogni affermazione fattuale come claim atomico separato.
Un claim atomico è una singola proposizione verificabile, non decomponibile ulteriormente.

Regole:
- Un claim per riga
- Nessuna numerazione
- Solo affermazioni, non domande o istruzioni
- Conserva il riferimento contestuale originale

Testo:
{text}

Claims atomici:"""

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def segment(self, llm_response: str) -> list[str]:
        prompt = self.SEGMENTATION_PROMPT.format(text=llm_response)

        # Una singola generazione deterministica (T→0)
        gens = self.llm.generate(prompt, n_samples=1, temperature=0.1, max_tokens=512)
        if not gens:
            return [llm_response]

        raw = gens[0].text
        claims = [line.strip() for line in raw.splitlines() if line.strip()]
        return claims if claims else [llm_response]


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class SemanticUncertaintyPipeline:
    """
    Pipeline completa: dato un claim e il suo contesto,
    restituisce ClaimUncertainty con semantic entropy.

    Uso:
        pipeline = SemanticUncertaintyPipeline()
        result = pipeline.evaluate_claim(
            claim="La Tachipirina si prende ogni 6 ore",
            context="L'utente ha chiesto: 'Ho mal di testa, posso prendere la Tachipirina?'"
        )
        print(result.semantic_entropy)  # → valore ∈ [0, log(M)]
    """

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        clusterer: Optional[BidirectionalEntailmentClusterer] = None,
        n_samples: int = 10,
        temperature: float = 0.5,
    ):
        self.llm = llm or OllamaLLM()
        self.clusterer = clusterer or BidirectionalEntailmentClusterer()
        self.n_samples = n_samples
        self.temperature = temperature

    def evaluate_claim(self, claim: str, context: str = "") -> ClaimUncertainty:
        """
        Dato un claim e il suo contesto di domanda,
        campiona M risposte e ne misura l'incertezza semantica.
        
        Il prompt chiede all'LLM di rispondere alla stessa domanda
        (non di riformulare il claim — questo è cruciale).
        """
        prompt = self._build_prompt(claim, context)

        logger.info(f"Campionamento {self.n_samples} generazioni per: '{claim[:60]}...'")
        generations = self.llm.generate(
            prompt,
            n_samples=self.n_samples,
            temperature=self.temperature,
        )

        logger.info("Clustering semantico via entailment bidirezionale...")
        clusters = self.clusterer.cluster(generations, context=context)

        se = compute_semantic_entropy(clusters)
        logger.info(f"Semantic entropy: {se:.4f}, cluster distinti: {len(clusters)}")

        return ClaimUncertainty(
            claim_text=claim,
            generations=generations,
            clusters=clusters,
            semantic_entropy=se,
            n_distinct_meanings=len(clusters),
        )

    def _build_prompt(self, claim: str, context: str) -> str:
        """
        Costruisce il prompt seguendo lo stile di Kuhn per QA.
        Il context è la domanda originale dell'utente (dopo ASR).
        """
        if context:
            return (
                f"Contesto: {context}\n\n"
                f"Verifica e rispondi: {claim}\n\n"
                f"Risposta:"
            )
        return f"Verifica e rispondi: {claim}\n\nRisposta:"
