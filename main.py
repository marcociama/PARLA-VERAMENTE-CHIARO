"""
main.py
-------
Entry point end-to-end per PARLA CHIARO — DialectGuard.

Modalità di esecuzione:
  python main.py --mode smoke          # verifica env e modelli (no audio)
  python main.py --mode corpus         # stats corpus PARLA CHIARO
  python main.py --mode transcribe     # ASR su primo file audio del corpus
  python main.py --mode pipeline       # pipeline completa (Claude API)
  python main.py --mode overconfidence # overconfidence analysis (Claude API)

Flags:
  --audio PATH         file audio specifico (override corpus)
  --limit N            max clip da processare in overconfidence mode
  --llm-backend        'anthropic' (default) oppure 'ollama'
  --ollama-model       modello Ollama se backend=ollama (default: llama3)
  --claude-model       modello Claude (default: claude-haiku-4-5-20251001)

Richiede: ANTHROPIC_API_KEY nell'ambiente (per backend anthropic).
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # carica .env se presente

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

CORPUS_PATH = Path("PARLA_CHIARO_recordings")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: smoke test
# ──────────────────────────────────────────────────────────────────────────────

def mode_smoke():
    """Verifica device, import e modelli senza dati reali."""
    print("\n=== SMOKE TEST ===")

    # Device
    from utils.device import device_info
    info = device_info()
    print(f"  torch_device:            {info['torch_device']}")
    print(f"  transformers_device:     {info['transformers_device']}")
    print(f"  ctranslate2_device:      {info['ctranslate2_device']}/{info['ctranslate2_compute_type']}")
    print(f"  MPS available:           {info['mps_available']}")
    print(f"  CUDA available:          {info['cuda_available']}")

    # Corpus loader
    print("\n  [corpus loader]", end=" ")
    from utils.corpus_loader import PARLACHIAROLoader
    loader = PARLACHIAROLoader(CORPUS_PATH)
    sessions = loader.load_all()
    stats = loader.stats(sessions)
    print(f"OK — {stats['n_participants']} partecipanti, {stats['n_recordings']} registrazioni")

    # OOV detector (UmBERTo tokenizer — solo tokenizer, veloce)
    print("  [OOV detector]", end=" ", flush=True)
    from core.whisperx_integration import OOVDetector
    oov = OOVDetector()
    # 'ttruvà' (napoletano: trovare con geminazione) → 4 subword → OOV
    # 'guaglione' → solo 2 subword (UmBERTo lo conosce da testi web) → NON OOV
    # 'paziente' → 1 subword, italiano standard
    assert oov.is_oov("ttruvà"), "Atteso OOV per 'ttruvà'"
    assert not oov.is_oov("paziente"), "Atteso non-OOV per 'paziente'"
    print("OK — 'ttruvà' è OOV (4 subword), 'paziente' non è OOV")
    print("       Nota: 'guaglione'=2 subword (UmBERTo lo conosce). AND con ASR conf compensa.")

    # NLI model (DeBERTa — download pesante al primo run)
    print("  [NLI clusterer]", end=" ", flush=True)
    from core.semantic_entropy import BidirectionalEntailmentClusterer, Generation
    clusterer = BidirectionalEntailmentClusterer()
    g1 = Generation("Il paziente ha mal di testa.", -10.0)
    g2 = Generation("Il paziente soffre di cefalea.", -10.5)
    g3 = Generation("Il paziente ha la febbre.", -11.0)
    clusters = clusterer.cluster([g1, g2, g3], context="Sintomi del paziente")
    print(f"OK — {len(clusters)} cluster distinti (atteso: 2 se mal-di-testa≡cefalea)")

    print("\n  SMOKE TEST COMPLETATO")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: corpus stats
# ──────────────────────────────────────────────────────────────────────────────

def mode_corpus():
    """Mostra statistiche del corpus PARLA CHIARO."""
    from utils.corpus_loader import PARLACHIAROLoader
    loader = PARLACHIAROLoader(CORPUS_PATH)
    sessions = loader.load_all()
    stats = loader.stats(sessions)

    print("\n=== CORPUS PARLA CHIARO ===")
    print(f"  Partecipanti:        {stats['n_participants']}")
    print(f"  Registrazioni:       {stats['n_recordings']}")
    print(f"  Con audio:           {stats['n_with_audio']}")
    print(f"  Parlanti napoletani: {stats['n_neapolitan']}")
    print(f"  Dialetti:            {stats['dialects']}")
    print(f"  Fasce età:           {stats['age_ranges']}")
    print(f"  Categorie prompt:    {stats['prompt_categories']}")

    print("\n  Esempio ground truth (prime 3 registrazioni):")
    for s in sessions[:3]:
        for r in s.recordings[:1]:
            print(f"    [{s.participant.participant_id}]")
            print(f"    promptText: \"{r.prompt_text[:100]}\"")
            print(f"    audio: {r.audio_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: transcribe
# ──────────────────────────────────────────────────────────────────────────────

def mode_transcribe(audio_path: Path | None = None):
    """Trascrive un file audio con WhisperX + annotazione dialettale."""
    if audio_path is None:
        from utils.corpus_loader import PARLACHIAROLoader
        loader = PARLACHIAROLoader(CORPUS_PATH)
        sessions = loader.load_all()
        recordings_with_audio = [
            r for s in sessions for r in s.recordings if r.audio_path
        ]
        if not recordings_with_audio:
            print("Nessun file audio trovato nel corpus.")
            return
        rec = recordings_with_audio[0]
        audio_path = rec.audio_path
        ground_truth = rec.prompt_text
        print(f"\n=== TRASCRIZIONE ===")
        print(f"  File:         {audio_path}")
        print(f"  Ground truth: \"{ground_truth[:100]}\"")
    else:
        ground_truth = None
        print(f"\n=== TRASCRIZIONE ===")
        print(f"  File: {audio_path}")

    from core.whisperx_integration import DialectSignalExtractor
    extractor = DialectSignalExtractor()
    asr_output = extractor.process_audio(audio_path)

    print(f"\n  Trascrizione Whisper: \"{asr_output.transcript}\"")
    print(f"  Lingua rilevata:      {asr_output.language_detected}")
    print(f"  Token totali:         {len(asr_output.tokens)}")
    print(f"  Token dialettali:     {len(asr_output.dialect_risk_tokens)} "
          f"({asr_output.overall_dialect_risk:.1%})")

    if asr_output.dialect_risk_tokens:
        print("\n  Token flaggati come dialettali:")
        for t in asr_output.dialect_risk_tokens:
            print(f"    [{t.start_time:.2f}s] \"{t.token}\" "
                  f"conf={t.asr_confidence:.2f} OOV={t.is_oov_italian}")

    if ground_truth:
        # WER semplice (token-level)
        ref = ground_truth.lower().split()
        hyp = asr_output.transcript.lower().split()
        common = set(ref) & set(hyp)
        wer_proxy = 1.0 - len(common) / max(len(ref), 1)
        print(f"\n  WER proxy (token set):  {wer_proxy:.1%}")

    return asr_output


# ──────────────────────────────────────────────────────────────────────────────
# Mode: pipeline completa
# ──────────────────────────────────────────────────────────────────────────────

def _make_llm(backend: str, ollama_model: str, claude_model: str, gemini_model: str):
    from core.semantic_entropy import AnthropicLLM, GeminiLLM, OllamaLLM
    if backend == "ollama":
        return OllamaLLM(model=ollama_model)
    if backend == "gemini":
        return GeminiLLM(model=gemini_model)
    return AnthropicLLM(model=claude_model)


def mode_pipeline(
    audio_path: Path | None = None,
    backend: str = "gemini",
    ollama_model: str = "llama3",
    claude_model: str = "claude-haiku-4-5-20251001",
    gemini_model: str = "gemini-2.5-flash-lite",
    n_samples: int = 5,
    max_claims: int = 3,
):
    """Pipeline completa: ASR → LLM → SE → DAWS risk assessment."""
    import uuid

    # Step 1: ASR
    asr_output = mode_transcribe(audio_path)
    if asr_output is None:
        return

    transcript = asr_output.transcript
    print(f"\n=== PIPELINE LLM + SEMANTIC ENTROPY ===")

    # Step 2: LLM sampling + semantic entropy
    from core.semantic_entropy import (
        BidirectionalEntailmentClusterer,
        ClaimSegmenter, SemanticUncertaintyPipeline,
    )

    llm = _make_llm(backend, ollama_model, claude_model, gemini_model)
    segmenter_llm = _make_llm(backend, ollama_model, claude_model, gemini_model)
    clusterer = BidirectionalEntailmentClusterer()
    segmenter = ClaimSegmenter(segmenter_llm)
    pipeline = SemanticUncertaintyPipeline(
        llm=llm, clusterer=clusterer, n_samples=n_samples, temperature=0.5
    )

    # Genera risposta LLM e segmenta in claim
    print(f"  Generando risposta per: \"{transcript[:80]}...\"")
    initial_gens = llm.generate(
        f"Sei un assistente italiano. L'utente dice: \"{transcript}\". Rispondi.",
        n_samples=1, temperature=0.3,
    )
    llm_response = initial_gens[0].text if initial_gens else transcript
    print(f"\n  Risposta LLM:\n  \"{llm_response[:400]}\"")

    claims = segmenter.segment(llm_response)
    print(f"\n  Claim estratti: {len(claims)} (analizzo i primi {min(max_claims, len(claims))})")

    uncertainties = []
    for i, claim in enumerate(claims[:max_claims]):
        print(f"\n  [{i+1}/{min(max_claims, len(claims))}] Claim: \"{claim}\"")
        result = pipeline.evaluate_claim(claim=claim, context=transcript)
        uncertainties.append(result)
        print(f"    Generazioni ({n_samples}):")
        seen = set()
        for g in result.generations:
            short = g.text[:100]
            if short not in seen:
                print(f"      · {short}")
                seen.add(short)
        print(f"    Cluster semantici distinti: {result.n_distinct_meanings}")
        print(f"    Semantic Entropy: {result.semantic_entropy:.4f}")

    # Step 3: DAWS risk assessment
    print("\n=== DAWS RISK ASSESSMENT ===")
    from core.daws import DAWS
    daws = DAWS()
    response_risk = daws.assess_response(
        uncertainties, asr_output=asr_output, session_id=str(uuid.uuid4())
    )

    print(f"  Overall risk: {response_risk.overall_risk_score:.4f} "
          f"→ {response_risk.overall_risk_level.upper()}")
    print(f"  Alert triggered: {response_risk.alert_triggered}")

    for cr in response_risk.claims:
        icon = {"green": "✓", "yellow": "⚠", "red": "✗"}.get(cr.risk_level, "?")
        print(f"  {icon} [{cr.risk_level.upper()}] \"{cr.claim_text[:60]}\"")
        print(f"      {cr.justification}")

    if response_risk.clarification_question:
        print(f"\n  Domanda chiarificatrice: \"{response_risk.clarification_question}\"")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: overconfidence analysis
# ──────────────────────────────────────────────────────────────────────────────

def mode_overconfidence(
    limit: int = 5,
    backend: str = "gemini",
    ollama_model: str = "llama3",
    claude_model: str = "claude-haiku-4-5-20251001",
    gemini_model: str = "gemini-2.5-flash-lite",
):
    """
    Overconfidence analysis su paper samples (o corpus Cacioli se disponibile).
    Non richiede audio: usa le trascrizioni testo.
    """
    from core.semantic_entropy import (
        BidirectionalEntailmentClusterer, SemanticUncertaintyPipeline,
    )
    from analysis.overconfidence_analysis import (
        CacioliCorpusLoader, OverconfidenceAnalyzer,
        plot_overconfidence_scatter, plot_domain_breakdown,
    )

    print("\n=== OVERCONFIDENCE ANALYSIS ===")

    llm = _make_llm(backend, ollama_model, claude_model, gemini_model)
    clusterer = BidirectionalEntailmentClusterer()
    pipeline = SemanticUncertaintyPipeline(llm=llm, clusterer=clusterer, n_samples=10)

    loader = CacioliCorpusLoader()
    clips = loader.load()[:limit]
    print(f"  Clip da analizzare: {len(clips)}")

    analyzer = OverconfidenceAnalyzer(pipeline)
    report = analyzer.run_full_analysis(clips)

    print(f"\n  Risultati:")
    print(f"    N clip:             {report.n_clips}")
    print(f"    % overconfident:    {report.n_overconfident:.1%}")
    print(f"    SE corretta (mean): {report.mean_se_correct:.4f}")
    print(f"    SE distorta (mean): {report.mean_se_distorted:.4f}")
    print(f"    ΔSE medio:          {report.mean_delta_se:.4f} ± {report.std_delta_se:.4f}")
    print(f"    Wilcoxon p-value:   {report.wilcoxon_pvalue:.4f}")
    print(f"    Effect size r:      {report.effect_size_r:.3f}")

    # Plot
    from pathlib import Path
    results_dir = Path("results/overconfidence")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Carica i risultati per il plot
    import json
    from analysis.overconfidence_analysis import ClipAnalysis
    full_results_path = results_dir / "full_results.json"
    if full_results_path.exists():
        with open(full_results_path) as f:
            raw = json.load(f)
        results = [ClipAnalysis(**r) for r in raw]
        plot_overconfidence_scatter(results, report, save_path=results_dir / "scatter.png")
        plot_domain_breakdown(results, save_path=results_dir / "domain_breakdown.png")
        print(f"\n  Plot salvati in: {results_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PARLA CHIARO — DialectGuard pipeline")
    parser.add_argument(
        "--mode",
        choices=["smoke", "corpus", "transcribe", "pipeline", "overconfidence"],
        default="smoke",
        help="Modalità di esecuzione (default: smoke)",
    )
    parser.add_argument("--audio", type=Path, default=None, help="File audio specifico")
    parser.add_argument("--limit", type=int, default=5, help="Max clip (overconfidence mode)")
    parser.add_argument("--llm-backend", default="gemini", choices=["gemini", "anthropic", "ollama"])
    parser.add_argument("--ollama-model", default="llama3")
    parser.add_argument("--claude-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--n-samples", type=int, default=5, help="Campionamenti LLM per claim (default: 5)")
    parser.add_argument("--max-claims", type=int, default=3, help="Max claim da analizzare (default: 3)")
    args = parser.parse_args()

    if args.mode == "smoke":
        mode_smoke()
    elif args.mode == "corpus":
        mode_corpus()
    elif args.mode == "transcribe":
        mode_transcribe(args.audio)
    elif args.mode == "pipeline":
        mode_pipeline(args.audio, args.llm_backend, args.ollama_model, args.claude_model, args.gemini_model, args.n_samples, args.max_claims)
    elif args.mode == "overconfidence":
        mode_overconfidence(args.limit, args.llm_backend, args.ollama_model, args.claude_model, args.gemini_model)


if __name__ == "__main__":
    main()
