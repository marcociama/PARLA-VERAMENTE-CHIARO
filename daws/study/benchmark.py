"""
study/benchmark.py
------------------
Multi-LLM benchmark for DAWS uncertainty quantification.

Architecture: adapter pattern over a common LLMAdapter interface.

  LocalLLMAdapter   — wraps LocalLLM (Mistral 7B, any HF causal model)
                      has_logits=True  → full Semantic Density (Eq. 1–4)
                      has_logits=True  → Inv-Entropy

  GeminiAdapter     — wraps Gemini API (black-box, no logits)
                      has_logits=False → SD NOT computed (requires logits)
                      has_logits=False → Inv-Entropy only

  ClaudeAdapter     — same as GeminiAdapter

  OllamaAdapter     — same as GeminiAdapter

Semantic Density requires token-level log-probabilities (Eq. 1 of the SD paper):
  p_norm_i = exp( (1/L_i) * sum_t log p(y_i_t | x) )
Black-box APIs do not expose these → SD is undefined for them.
Only Inv-Entropy (text-output only) is computed for black-box models.

Benchmark loop (per recording):
  1. Load GT1/GT2/GT3 from promptText, W1/W2/W3 from ASR cache
  2. For each model:
     a. generate 6 responses (GT1..GT3, W1..W3) → Inv-Entropy (all models)
     b. if has_logits: generate_single + generate_diverse on GT1 → SD (local only)
  3. Collect: model, sd (local only), h_norm, time_s
  4. Save results to daws/results/benchmark.json
  5. Print LaTeX-ready comparison table

Usage:
    python main.py --mode benchmark --limit 10
    python main.py --mode benchmark --models mistral,gemini,claude
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
BENCHMARK_FILE = RESULTS_DIR / "benchmark.json"


# ---------------------------------------------------------------------------
# LLM adapter protocol
# ---------------------------------------------------------------------------

@dataclass
class AdapterResponse:
    text: str
    length_norm_prob: float   # exp(mean logprob per token); set to 1/M for black-box


class LLMAdapter(Protocol):
    name: str
    has_logits: bool

    def generate_single(self, prompt: str) -> AdapterResponse: ...
    def generate_diverse(self, prompt: str, M: int) -> list[AdapterResponse]: ...


# ---------------------------------------------------------------------------
# LocalLLMAdapter
# ---------------------------------------------------------------------------

class LocalLLMAdapter:
    """Wraps LocalLLM (Mistral 7B or any HF causal model). Has token-level logits."""

    def __init__(self, model_id: Optional[str] = None, max_new_tokens: int = 256):
        self.name = (model_id or "mistral-7b").split("/")[-1].lower()
        self.has_logits = True
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from daws.pipeline.llm import LocalLLM
            self._llm = LocalLLM(model_id=self._model_id, max_new_tokens=self._max_new_tokens)
        return self._llm

    def generate_single(self, prompt: str) -> AdapterResponse:
        result = self._get_llm().generate_single(prompt)
        return AdapterResponse(text=result.text, length_norm_prob=result.length_norm_prob)

    def generate_diverse(self, prompt: str, M: int) -> list[AdapterResponse]:
        result = self._get_llm().generate_diverse(prompt, M=M)
        return [AdapterResponse(text=r.text, length_norm_prob=r.length_norm_prob) for r in result.responses]


# ---------------------------------------------------------------------------
# GeminiAdapter (black-box)
# ---------------------------------------------------------------------------

class GeminiAdapter:
    """
    Wraps Gemini API via google-genai.
    No logits → length_norm_prob = 1/M (uniform weight for SD approximation).
    Multiple samples obtained via temperature > 0.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite", temperature: float = 0.7):
        self.name = model.replace("/", "-")
        self.has_logits = False
        self._model = model
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os
            from google import genai
            self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        return self._client

    def _call(self, prompt: str, temperature: float) -> str:
        import time
        from google.genai import types
        for attempt in range(4):
            try:
                response = self._get_client().models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=256),
                )
                return response.text or ""
            except Exception as e:
                wait = 5 * (2 ** attempt)
                logger.warning(f"Gemini attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
        return ""

    def generate_single(self, prompt: str) -> AdapterResponse:
        text = self._call(prompt, temperature=0.0)
        return AdapterResponse(text=text, length_norm_prob=1.0)

    def generate_diverse(self, prompt: str, M: int) -> list[AdapterResponse]:
        # Sample M times with temperature; uniform p_norm = 1/M for SD
        responses = []
        for _ in range(M):
            text = self._call(prompt, temperature=self._temperature)
            responses.append(AdapterResponse(text=text, length_norm_prob=1.0 / M))
        return responses


# ---------------------------------------------------------------------------
# ClaudeAdapter (black-box)
# ---------------------------------------------------------------------------

class ClaudeAdapter:
    """
    Wraps Anthropic Claude API.
    No logits → uniform p_norm for SD approximation.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", temperature: float = 0.7):
        self.name = model
        self.has_logits = False
        self._model = model
        self._temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os, anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        return self._client

    def _call(self, prompt: str, temperature: float) -> str:
        try:
            msg = self._get_client().messages.create(
                model=self._model,
                max_tokens=256,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if msg.content else ""
        except Exception as e:
            logger.warning(f"Claude call failed: {e}")
            return ""

    def generate_single(self, prompt: str) -> AdapterResponse:
        return AdapterResponse(text=self._call(prompt, 0.0), length_norm_prob=1.0)

    def generate_diverse(self, prompt: str, M: int) -> list[AdapterResponse]:
        return [
            AdapterResponse(text=self._call(prompt, self._temperature), length_norm_prob=1.0 / M)
            for _ in range(M)
        ]


# ---------------------------------------------------------------------------
# OllamaAdapter (black-box, local server)
# ---------------------------------------------------------------------------

class OllamaAdapter:
    """Wraps a local Ollama server. Treats as black-box (no logits)."""

    def __init__(self, model: str = "llama3", temperature: float = 0.7):
        self.name = f"ollama-{model}"
        self.has_logits = False
        self._model = model
        self._temperature = temperature

    def _call(self, prompt: str, temperature: float) -> str:
        import urllib.request, json as _json
        payload = _json.dumps({"model": self._model, "prompt": prompt,
                               "stream": False, "options": {"temperature": temperature}}).encode()
        try:
            with urllib.request.urlopen("http://localhost:11434/api/generate", data=payload, timeout=120) as r:
                return _json.loads(r.read())["response"].strip()
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return ""

    def generate_single(self, prompt: str) -> AdapterResponse:
        return AdapterResponse(text=self._call(prompt, 0.0), length_norm_prob=1.0)

    def generate_diverse(self, prompt: str, M: int) -> list[AdapterResponse]:
        return [AdapterResponse(text=self._call(prompt, self._temperature), length_norm_prob=1.0 / M) for _ in range(M)]


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

def build_adapters(model_names: list[str]) -> list:
    """
    Build a list of adapters from model name strings.

    Supported names:
      "mistral"    → LocalLLMAdapter (Mistral 7B)
      "gemini"     → GeminiAdapter (gemini-2.5-flash-lite)
      "claude"     → ClaudeAdapter (claude-haiku-4-5)
      "ollama"     → OllamaAdapter (llama3)
      any HF ID    → LocalLLMAdapter with that model ID
    """
    adapters = []
    for name in model_names:
        name = name.strip().lower()
        if name == "mistral":
            adapters.append(LocalLLMAdapter())
        elif name == "gemini":
            adapters.append(GeminiAdapter())
        elif name == "claude":
            adapters.append(ClaudeAdapter())
        elif name == "ollama":
            adapters.append(OllamaAdapter())
        else:
            # Treat as HuggingFace model ID
            adapters.append(LocalLLMAdapter(model_id=name))
    return adapters


# ---------------------------------------------------------------------------
# SD computation with adapter responses
# ---------------------------------------------------------------------------

def _compute_sd_from_responses(
    query: str,
    best: AdapterResponse,
    refs: list[AdapterResponse],
    nli_scorer,
) -> tuple[float, float]:
    """
    Full Semantic Density (Algorithm 1, Qiu & Miikkulainen NeurIPS 2024).
    Called ONLY for models with logits (has_logits=True).
    p_norm_i = best.length_norm_prob from LocalLLM token logprobs.

    Returns (sd, u_llm).
    """
    from daws.pipeline.semantic_density import _NLI_MAX_CHARS

    if not refs:
        return 0.5, 0.5

    pairs = [
        (
            (query + " " + best.text)[:_NLI_MAX_CHARS],
            (query + " " + r.text)[:_NLI_MAX_CHARS],
        )
        for r in refs
    ]
    nli_results = nli_scorer.score_batch(pairs)

    distances, p_norms = [], []
    for ref, (p_c, p_n, p_e) in zip(refs, nli_results):
        distances.append(p_c + 0.5 * p_n)
        p_norms.append(ref.length_norm_prob)

    kernels = [1.0 - d for d in distances]
    denom = sum(p_norms)
    if denom < 1e-12:
        sd = sum(kernels) / len(kernels)
    else:
        sd = sum(p * k for p, k in zip(p_norms, kernels)) / denom

    return max(0.0, min(1.0, sd)), 1.0 - max(0.0, min(1.0, sd))


# ---------------------------------------------------------------------------
# Per-recording benchmark
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_name: str
    has_logits: bool
    sd: Optional[float]     # None for black-box models (logits unavailable)
    u_llm: Optional[float]  # None for black-box models
    h_norm: float           # Inv-Entropy — computed for ALL models
    time_sd_s: float        # time for SD computation (0 for black-box)
    time_ie_s: float        # time for 6 IE responses


@dataclass
class RecordingResult:
    participant_id: str
    dialect: str
    wer: float
    u_asr: float
    gt1: str
    w1: str
    models: list[ModelResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    corpus_root: str,
    asr_cache_dir: str,
    adapters: list,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    sd_m_refs: int = 5,
    bootstrap_B: int = 20,   # fewer than full study for speed
) -> list[dict]:
    """
    Run multi-LLM benchmark on PARLA CHIARO corpus.

    Args:
        corpus_root:   corpus root path
        asr_cache_dir: ASR cache directory (requires asr-cache to be run first)
        adapters:      list of LLMAdapter instances
        output_path:   JSON output path
        limit:         max recordings
        sd_m_refs:     M diverse references for SD
        bootstrap_B:   bootstrap replicates for Inv-Entropy

    Returns:
        List of serialised RecordingResult dicts.
    """
    from daws.utils.corpus_loader import PARLACHIAROLoader
    from daws.pipeline.asr import ASRResult
    from daws.pipeline.semantic_density import NLIScorer
    from daws.pipeline.inv_entropy import InvEntropyCalculator

    loader = PARLACHIAROLoader(corpus_root)
    sessions = loader.load_all()
    nli = NLIScorer()
    ie_calc = InvEntropyCalculator(B=bootstrap_B)

    asr_cache_dir = Path(asr_cache_dir)
    out_path = Path(output_path) if output_path else BENCHMARK_FILE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    processed = 0

    def _wer(ref: str, hyp: str) -> float:
        from daws.study.inv_entropy_study import _word_error_rate
        return _word_error_rate(ref, hyp)

    def _prompt(text: str) -> str:
        return f"A patient says: '{text}'\nReply briefly in standard Italian."

    for session in sessions:
        if limit and processed >= limit:
            break
        for rec in session.recordings:
            if limit and processed >= limit:
                break
            if rec.audio_path is None:
                continue

            # Load ASR cache
            cache_file = asr_cache_dir / f"{rec.audio_path.stem}.json"
            if not cache_file.exists():
                continue

            with open(cache_file, encoding="utf-8") as f:
                asr = ASRResult.from_dict(json.load(f))

            # Load GT1/GT2/GT3
            json_path = rec.audio_path.with_suffix(".json")
            if not json_path.exists():
                continue
            with open(json_path, encoding="utf-8") as f:
                raw_prompt = json.load(f).get("promptText", "")
            gt = (raw_prompt + [raw_prompt[-1]] * 3)[:3] if isinstance(raw_prompt, list) else [raw_prompt] * 3
            top3 = asr.top3

            if not all(w.strip() for w in top3):
                continue

            rec_result = RecordingResult(
                participant_id=rec.participant_id,
                dialect=session.participant.dialect_label,
                wer=_wer(gt[0], top3[0]),
                u_asr=asr.u_asr,
                gt1=gt[0],
                w1=top3[0],
            )

            for adapter in adapters:
                logger.info(f"  [{processed+1}] {adapter.name} | {rec.participant_id}")

                # Inv-Entropy: 6 responses — computed for ALL models
                inputs = gt + top3
                t0 = time.time()
                try:
                    ie_responses = [adapter.generate_single(_prompt(t)).text for t in inputs]
                    ie_result = ie_calc.compute(inputs=inputs, outputs=ie_responses)
                    h_norm = ie_result.inv_entropy_norm
                except Exception as e:
                    logger.error(f"IE failed for {adapter.name}: {e}")
                    h_norm = float("nan")
                t_ie = time.time() - t0

                # SD: only for models with logits (requires token-level log-probs)
                sd, u_llm, t_sd = None, None, 0.0
                if adapter.has_logits:
                    p_sd = _prompt(gt[0])
                    t0 = time.time()
                    try:
                        best = adapter.generate_single(p_sd)
                        refs = adapter.generate_diverse(p_sd, M=sd_m_refs)
                        sd_val, u_llm_val = _compute_sd_from_responses(
                            p_sd, best, refs, nli
                        )
                        sd, u_llm = sd_val, u_llm_val
                    except Exception as e:
                        logger.error(f"SD failed for {adapter.name}: {e}")
                    t_sd = time.time() - t0

                rec_result.models.append(ModelResult(
                    model_name=adapter.name,
                    has_logits=adapter.has_logits,
                    sd=sd,
                    u_llm=u_llm,
                    h_norm=h_norm,
                    time_sd_s=t_sd,
                    time_ie_s=t_ie,
                ))

            all_results.append(asdict(rec_result))
            processed += 1

            if processed % 3 == 0:
                _save(all_results, out_path)

    _save(all_results, out_path)
    logger.info(f"Benchmark complete: {processed} recordings → {out_path}")
    _print_table(all_results, adapters)
    return all_results


def _save(results: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# LaTeX-ready summary table
# ---------------------------------------------------------------------------

def _print_table(results: list[dict], adapters: list):
    """Print a LaTeX-ready comparison table.

    SD column is shown only for models with logits.
    Inv-Entropy (H_norm) is shown for all models.
    """
    model_names = [a.name for a in adapters]
    stats = {n: {"sd": [], "u_llm": [], "h_norm": [], "time_ie": []} for n in model_names}

    for rec in results:
        for m in rec.get("models", []):
            n = m["model_name"]
            if n not in stats:
                continue
            if m["sd"] is not None:
                stats[n]["sd"].append(m["sd"])
                stats[n]["u_llm"].append(m["u_llm"])
            if not math.isnan(m.get("h_norm", float("nan"))):
                stats[n]["h_norm"].append(m["h_norm"])
            stats[n]["time_ie"].append(m.get("time_ie_s", 0.0))

    def mv(lst): return f"{np.mean(lst):.3f}" if lst else "—"
    def sv(lst): return f"{np.std(lst):.3f}" if lst else "—"

    print("\n=== Benchmark Results ===")
    print(f"  SD computed only for models with logits (LocalLLMAdapter).")
    print(f"  Inv-Entropy computed for all models.\n")
    header = f"{'Model':<30} {'Logits':>7} {'SD':>8} {'U_LLM':>8} {'H_norm (IE)':>12} {'IE time(s)':>11}"
    print(header)
    print("-" * len(header))
    for n in model_names:
        st = stats[n]
        has_l = next((a.has_logits for a in adapters if a.name == n), False)
        print(
            f"{n:<30} {'yes' if has_l else 'no':>7} "
            f"{mv(st['sd']):>8} {mv(st['u_llm']):>8} "
            f"{mv(st['h_norm']):>12} {mv(st['time_ie']):>11}"
        )

    print("\nLaTeX rows:")
    for n in model_names:
        st = stats[n]
        has_l = next((a.has_logits for a in adapters if a.name == n), False)
        lbl = r"\checkmark" if has_l else r"$\times$"
        sd_cell = f"${mv(st['sd'])} \\pm {sv(st['sd'])}$" if st["sd"] else r"\textemdash"
        print(
            f"{n} & {lbl} & {sd_cell}"
            f" & ${mv(st['h_norm'])} \\pm {sv(st['h_norm'])}$ \\\\"
        )
