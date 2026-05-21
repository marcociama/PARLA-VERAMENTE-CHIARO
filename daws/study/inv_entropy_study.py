"""
study/inv_entropy_study.py
--------------------------
Offline Inv-Entropy study on the PARLA CHIARO corpus.

For each recording (N=6 design):
  inputs  = [GT1, GT2, GT3, W1, W2, W3]
    GT1/GT2/GT3  — three variants from the promptText list in the recording JSON
    W1/W2/W3     — top-3 beam search hypotheses from the ASR cache

  outputs = six LLM responses, one per input

Computed metrics per recording:
  wer     — word error rate proxy: edit_distance(GT1, W1) / len(GT1)
  u_asr   — 1 - mean WhisperX word confidence (from ASR cache)
  h_norm  — Inv-Entropy u_inv_entropy = H_hat / log(6) ∈ [0, 1]

Results saved to daws/results/inv_entropy_study.json.
Summary statistics printed via analyze_results().
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RESULTS_DIR    = Path(__file__).parent.parent / "results"
RESULTS_FILE   = RESULTS_DIR / "inv_entropy_study.json"
LLM_CACHE_DIR  = RESULTS_DIR / "llm_cache"


# ---------------------------------------------------------------------------
# WER (word-level edit distance / reference length)
# ---------------------------------------------------------------------------

def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 1.0 if hyp else 0.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / n


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _medical_prompt(text: str) -> str:
    return (
        f"Sei un assistente medico virtuale esperto e rigoroso. Rispondi all'affermazione o alla richiesta del paziente in modo appropriato e professionale.\n\n"
        f"RICEVI QUESTI VINCOLI TASSATIVI DI GENERAZIONE:\n\n"
        f"1. Lingua: Rispondi ESCLUSIVAMENTE in lingua italiana standard. Non includere MAI traduzioni in inglese, note sintattiche, glosse o metatesto in altre lingue.\n\n"
        f"2. Formato: Non usare alcun simbolo di formattazione Markdown, asterischi (**), hashtag (#), o trattini. Non racchiudere la risposta tra virgolette esterne.\n\n"
        f"3. Sintassi: Usa solo parole e punteggiatura standard (punti e virgole). Non generare elenchi puntati o numerati; rispondi con un testo continuo e lineare.\n\n"
        f"4. Stile: Sii diretto. Inizia immediatamente con la risposta medica o la richiesta di chiarimento, senza preamboli cerimoniosi o saluti introduttivi.\n\n"
        f"5. Lunghezza: Rispondi in massimo due frasi. Sii conciso.\n\n"
        f"Paziente: {text}"
    )


def _clean_response(text: str) -> str:
    """Strip Mistral formatting artifacts before NLI computation.

    Mistral sometimes wraps responses in quotes and appends English translations:
        '"Sono preoccupato..."\n\n(Translation: I am concerned...)'
    These confuse the NLI model even for semantically identical texts.
    """
    text = text.strip().strip('"').strip("'")
    # Remove \n\n(Translation: ...) / \n\n(I am ...) fragments
    for sep in ["\n\n(", "\n(", "(Translation", "(I am", "(Sono"]:
        if sep in text:
            text = text.split(sep)[0]
    return text.strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Main study loop
# ---------------------------------------------------------------------------

def run_study(
    corpus_root: str,
    asr_cache_dir: str,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    llm_model: Optional[str] = None,
) -> list[dict]:
    """
    Run Inv-Entropy study over the full PARLA CHIARO corpus.

    Args:
        corpus_root:   root of the PARLA CHIARO dataset (contains recordings/, participants/)
        asr_cache_dir: directory with ASR JSON cache files (from --mode asr-cache)
        output_path:   where to write results JSON (default: daws/results/inv_entropy_study.json)
        limit:         process at most this many recordings
        llm_model:     Ollama model name, or None for default (mistral)

    Returns:
        List of per-recording result dicts.
    """
    from daws.utils.corpus_loader import PARLACHIAROLoader
    from daws.pipeline.asr import ASRResult
    from daws.pipeline.llm import OllamaLLM
    from daws.pipeline.inv_entropy import InvEntropyCalculator

    loader = PARLACHIAROLoader(corpus_root)
    sessions = loader.load_all()
    llm = OllamaLLM(model=llm_model or "mistral")
    ie_calc = InvEntropyCalculator()

    asr_cache_dir = Path(asr_cache_dir)
    out_path = Path(output_path) if output_path else RESULTS_FILE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results, processed = [], 0

    for session in sessions:
        for rec in session.recordings:
            if limit and processed >= limit:
                break
            if rec.audio_path is None:
                continue

            # Load promptText list from original JSON for GT1/GT2/GT3
            json_path = rec.audio_path.with_suffix(".json")
            if not json_path.exists():
                logger.warning(f"JSON not found: {json_path}")
                continue

            with open(json_path, encoding="utf-8") as f:
                rec_data = json.load(f)

            raw_prompt = rec_data.get("promptText", "")
            if isinstance(raw_prompt, list):
                gt = (raw_prompt + [raw_prompt[-1]] * 3)[:3]
            else:
                gt = [raw_prompt] * 3

            # Load W1/W2/W3 from ASR cache
            cache_file = asr_cache_dir / f"{rec.audio_path.stem}.json"
            if not cache_file.exists():
                logger.warning(f"ASR cache missing: {cache_file} — skipping")
                continue

            with open(cache_file, encoding="utf-8") as f:
                asr_result = ASRResult.from_dict(json.load(f))

            top3 = asr_result.top3
            if not all(w.strip() for w in top3):
                logger.warning(f"Incomplete top-3 for {rec.audio_path.stem} — skipping")
                continue

            wer = _word_error_rate(gt[0], top3[0])
            logger.info(
                f"[{processed+1}] {rec.audio_path.stem} | WER={wer:.3f} | "
                f"GT1='{gt[0][:40]}' | W1='{top3[0][:40]}'"
            )

            # Generate 6 LLM responses via Ollama — cached to disk for reproducibility
            inputs = gt + top3   # [GT1, GT2, GT3, W1, W2, W3]
            llm_cache_file = LLM_CACHE_DIR / f"{rec.audio_path.stem}.json"
            if llm_cache_file.exists():
                with open(llm_cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                responses = cached["responses"]
                logger.info(f"  LLM cache hit: {rec.audio_path.stem}")
            else:
                try:
                    responses = [
                        llm.generate_text(_medical_prompt(t)) for t in inputs
                    ]
                    with open(llm_cache_file, "w", encoding="utf-8") as f:
                        json.dump({"inputs": inputs, "responses": responses}, f,
                                  ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"LLM error for {rec.audio_path.stem}: {e}")
                    continue

            # Compute Inv-Entropy N=6
            # Clean responses before NLI: strip quote wrapping and English translation artifacts
            cleaned = [_clean_response(r) for r in responses]
            try:
                ie_result = ie_calc.compute_n6(
                    gt_texts=gt,
                    whisper_texts=top3,
                    llm_responses=cleaned,
                )
            except Exception as e:
                logger.error(f"Inv-Entropy error for {rec.audio_path.stem}: {e}")
                continue

            record = {
                "participant_id": rec.participant_id,
                "filename": rec.filename,
                "dialect": session.participant.dialect_label,
                "gender": session.participant.gender,
                "age_range": session.participant.age_range,
                "gt1": gt[0], "gt2": gt[1], "gt3": gt[2],
                "w1": top3[0], "w2": top3[1], "w3": top3[2],
                "wer": wer,
                "u_asr": asr_result.u_asr,
                "h_norm": ie_result.inv_entropy_norm,
                "u_inv_entropy": ie_result.u_inv_entropy,
                "diag_probs": ie_result.diag_probs,
                "llm_responses": responses,        # raw Mistral output
                "llm_responses_clean": cleaned,    # after artifact stripping (used for NLI)
            }
            results.append(record)
            processed += 1

            if processed % 5 == 0:
                _save(results, out_path)
                logger.info(f"Checkpoint: {processed} samples processed.")

    _save(results, out_path)
    logger.info(f"Study complete: {processed} samples → {out_path}")
    return results


def _save(results: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------

def analyze_results(results_path: Optional[str] = None) -> dict:
    """Load study results and print summary statistics."""
    path = Path(results_path) if results_path else RESULTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    wers = [r["wer"] for r in results]
    h_norms = [r["h_norm"] for r in results]

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Pearson correlation between WER and H_norm
    def pearson(x, y):
        n = len(x)
        mx, my = mean(x), mean(y)
        num = sum((a - mx) * (b - my) for a, b in zip(x, y))
        den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
        return num / den if den > 1e-12 else 0.0

    stats = {
        "n_samples": len(results),
        "wer_mean": mean(wers),
        "h_norm_mean": mean(h_norms),
        "pearson_wer_h": pearson(wers, h_norms),
    }

    print("=== Inv-Entropy Study Results ===")
    print(f"  Samples:              {stats['n_samples']}")
    print(f"  WER mean (W1 vs GT1): {stats['wer_mean']:.3f}")
    print(f"  H_norm mean:          {stats['h_norm_mean']:.4f}")
    print(f"  Pearson(WER, H_norm): {stats['pearson_wer_h']:.4f}")

    return stats
