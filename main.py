"""
main.py
-------
PARLA CHIARO — DialectGuard entry point.

Modes:
  corpus      — corpus statistics
  asr-cache   — pre-generate ASR JSON cache (top-3 beam search) for all recordings
  daws        — end-to-end DAWS on a WAV file (Von Neumann H_k6 pipeline)
  ie-study    — offline Inv-Entropy study over the full corpus
  calibrate   — legacy SD-based threshold calibration (use scripts/calibrate_geometry.py
                for the production Von Neumann calibration)
  characterize — NeurIPS-style LLM uncertainty characterization figure
  benchmark   — multi-LLM benchmark across local and API models

Flags:
  --audio PATH        WAV file for daws mode
  --limit N           process at most N recordings
  --asr-cache-dir     ASR cache directory (default: daws/results/asr_cache)
  --bootstrap-b N     bootstrap replicates for Inv-Entropy (default: 50)
"""

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

CORPUS_PATH = Path("PARLA_CHIARO_recordings_enriched")


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def mode_corpus():
    """Print PARLA CHIARO corpus statistics."""
    from daws.utils.corpus_loader import PARLACHIAROLoader
    loader = PARLACHIAROLoader(CORPUS_PATH)
    sessions = loader.load_all()
    stats = loader.stats(sessions)

    print("\n=== CORPUS PARLA CHIARO ===")
    print(f"  Participants:     {stats['n_participants']}")
    print(f"  Recordings:       {stats['n_recordings']}")
    print(f"  With audio:       {stats['n_with_audio']}")
    print(f"  Neapolitan:       {stats['n_neapolitan']}")
    print(f"  Dialects:         {stats['dialects']}")
    print(f"  Age ranges:       {stats['age_ranges']}")
    print(f"  Prompt categories:{stats['prompt_categories']}")

    print("\n  Ground truth samples (first 3 recordings):")
    for s in sessions[:3]:
        for r in s.recordings[:1]:
            print(f"    [{s.participant.participant_id}]")
            print(f"    promptText: \"{r.prompt_text[:100]}\"")
            print(f"    audio:      {r.audio_path}")


# ---------------------------------------------------------------------------
# asr-cache
# ---------------------------------------------------------------------------

def mode_asr_cache(asr_cache_dir: str, limit: int | None):
    """Pre-generate ASR JSON cache (WhisperX + top-3 beam search) for all recordings."""
    from daws.utils.corpus_loader import PARLACHIAROLoader
    from daws.pipeline.asr import ASRPipeline

    loader = PARLACHIAROLoader(CORPUS_PATH)
    sessions = loader.load_all()
    recordings = [r for s in sessions for r in s.recordings if r.audio_path]
    if limit:
        recordings = recordings[:limit]

    cache_dir = Path(asr_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ASRPipeline(language="it")

    print(f"\n=== ASR CACHE ({len(recordings)} files) ===")
    for i, rec in enumerate(recordings):
        cache_path = str(cache_dir / f"{rec.audio_path.stem}.json")
        print(f"  [{i+1}/{len(recordings)}] {rec.filename}")
        try:
            result = pipeline.process(str(rec.audio_path), cache_path=cache_path)
            print(f"    transcript: '{result.transcription[:60]}'")
            print(f"    top3[0]:    '{result.top3[0][:60]}'")
        except Exception as e:
            logger.error(f"Error on {rec.filename}: {e}")


# ---------------------------------------------------------------------------
# daws  (Von Neumann H_k6 pipeline)
# ---------------------------------------------------------------------------

def mode_daws(audio_path: Path | None, asr_cache_dir: str):
    """End-to-end DAWS: WAV → ASR → Mistral → H_k6 → U_pipeline → risk alert."""
    if audio_path is None:
        from daws.utils.corpus_loader import PARLACHIAROLoader
        sessions = PARLACHIAROLoader(CORPUS_PATH).load_all()
        recordings = [r for s in sessions for r in s.recordings if r.audio_path]
        if not recordings:
            print("No audio files found in corpus.")
            return
        audio_path = recordings[0].audio_path
        print(f"Using first corpus file: {audio_path}")

    from daws.pipeline.daws import DAWSPipeline
    pipeline = DAWSPipeline(use_mongo=False, asr_cache_dir=asr_cache_dir)
    result = pipeline.process_audio(str(audio_path))

    print(f"\n=== DAWS RESULT (Von Neumann H_k6) ===")
    print(f"  Transcript:    '{result.transcript[:100]}'")
    print(f"  LLM response:  '{result.llm_response[:100]}'")
    print(f"  U_ASR:         {result.u_asr:.4f}")
    print(f"  H_k6 (U_LLM): {result.h_k6_value:.4f}  (Von Neumann entropy, nats)")
    print(f"  s_W:           {[f'{s:.4f}' for s in result.s_w]}")
    print(f"  U_pipeline:    {result.u_pipeline:.4f}")
    print(f"  Risk level:    {result.risk_level}")
    if result.clarification_question:
        print(f"  Clarification: '{result.clarification_question}'")
    print(f"  Time:          {result.processing_time_s:.1f}s")
    return result


# ---------------------------------------------------------------------------
# ie-study
# ---------------------------------------------------------------------------

def mode_ie_study(asr_cache_dir: str, limit: int | None):
    """Offline Inv-Entropy study (N=6: GT1/GT2/GT3 + W1/W2/W3) over full corpus."""
    from daws.study.inv_entropy_study import run_study, analyze_results
    run_study(
        corpus_root=str(CORPUS_PATH),
        asr_cache_dir=asr_cache_dir,
        limit=limit,
    )
    analyze_results()


# ---------------------------------------------------------------------------
# calibrate  (legacy SD-based; for geometry use scripts/calibrate_geometry.py)
# ---------------------------------------------------------------------------

def mode_calibrate(results_path: str | None = None):
    """Legacy calibration: alpha/beta/gamma weights via SLSQP on SD+H_k4 proxy."""
    from daws.study.calibration import calibrate
    calibrate(results_path=results_path)


# ---------------------------------------------------------------------------
# characterize
# ---------------------------------------------------------------------------

def mode_characterize(reuse: bool):
    """NeurIPS-style LLM uncertainty characterization figure."""
    from daws.study.llm_characterization import run_characterization
    run_characterization()


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def mode_benchmark(
    asr_cache_dir: str,
    limit: int | None,
    models: str,
    bootstrap_b: int,
):
    """Multi-LLM benchmark: SD + Inv-Entropy across local and API models."""
    from daws.study.benchmark import build_adapters, run_benchmark

    model_list = [m.strip() for m in models.split(",")]
    adapters = build_adapters(model_list)
    print(f"Benchmarking {len(adapters)} models: {[a.name for a in adapters]}")

    run_benchmark(
        corpus_root=str(CORPUS_PATH),
        asr_cache_dir=asr_cache_dir,
        adapters=adapters,
        limit=limit,
        bootstrap_B=bootstrap_b,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PARLA CHIARO — DialectGuard")
    parser.add_argument(
        "--mode",
        choices=["corpus", "asr-cache", "daws",
                 "ie-study", "calibrate", "characterize", "benchmark"],
        default="corpus",
    )
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--asr-cache-dir", default="daws/results/asr_cache")
    parser.add_argument("--bootstrap-b", type=int, default=50)
    parser.add_argument("--models", default="mistral",
                        help="Comma-separated list: mistral,gemini,claude,ollama (benchmark mode)")
    parser.add_argument("--reuse-results", action="store_true",
                        help="Reuse cached results if available (characterize mode)")
    args = parser.parse_args()

    if args.mode == "corpus":
        mode_corpus()
    elif args.mode == "asr-cache":
        mode_asr_cache(args.asr_cache_dir, args.limit)
    elif args.mode == "daws":
        mode_daws(args.audio, args.asr_cache_dir)
    elif args.mode == "ie-study":
        mode_ie_study(args.asr_cache_dir, args.limit)
    elif args.mode == "calibrate":
        mode_calibrate()
    elif args.mode == "characterize":
        mode_characterize(args.reuse_results)
    elif args.mode == "benchmark":
        mode_benchmark(args.asr_cache_dir, args.limit, args.models, args.bootstrap_b)


if __name__ == "__main__":
    main()
