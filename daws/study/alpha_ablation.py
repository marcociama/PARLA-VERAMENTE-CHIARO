"""
daws/study/alpha_ablation.py
-----------------------------
Ablation study to choose alpha for proportional test-time audio augmentation.

Noise model:
    noise = np.random.randn(len(audio)) * audio.std() * alpha

Constant SNR across speakers: each sample gets perturbation proportional
to its own RMS amplitude, regardless of microphone gain or distance.

Protocol:
  - First 5 corpus samples (already processed)
  - alpha ∈ {0.005, 0.01, 0.02, 0.05}
  - Seeds {1, 2, 3} per alpha per sample
  - For each (sample, alpha, seed): compare W_augmented vs W_original

Metrics (per alpha, averaged over 5 samples × 3 seeds = 15 pairs):
  diversity_rate     — fraction of augmented W that differ from W_original
  edit_distance_mean — mean word-level Levenshtein(W_original, W_augmented)

Criterion: choose minimum alpha with diversity_rate > 0.30.

Usage:
    python daws/study/alpha_ablation.py
"""

import sys
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORPUS_ROOT = Path("PARLA_CHIARO_recordings_enriched")
ALPHAS = [0.005, 0.01, 0.02, 0.05]
SEEDS = [1, 2]   # seeds for W2, W3 (W1 = original, no noise)
N_SAMPLES = 5


# ---------------------------------------------------------------------------
# Word-level Levenshtein distance
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    a_words = a.lower().split()
    b_words = b.lower().split()
    n, m = len(a_words), len(b_words)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev if a_words[i-1] == b_words[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


# ---------------------------------------------------------------------------
# Audio loading + augmentation
# ---------------------------------------------------------------------------

def _load_audio(path: str) -> np.ndarray:
    """Load WAV as float32 mono at 16 kHz via whisperx."""
    import whisperx
    return whisperx.load_audio(path)


def _augment(audio: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    """Proportional Gaussian noise: SNR constant across speakers."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(audio)).astype(np.float32) * audio.std() * alpha
    return audio + noise   # no clipping — preserve dialectal prosody


# ---------------------------------------------------------------------------
# WhisperX loader (singleton, loaded once)
# ---------------------------------------------------------------------------

_WX_MODEL = None


def _get_wx_model():
    global _WX_MODEL
    if _WX_MODEL is None:
        import torch, whisperx
        if torch.cuda.is_available():
            device, compute = "cuda", "float16"
        else:
            device, compute = "cpu", "int8"
        logger.info(f"Loading WhisperX large-v3 ({device}/{compute}) ...")
        _WX_MODEL = whisperx.load_model(
            "large-v3", device=device, compute_type=compute, language="it"
        )
        logger.info("WhisperX loaded.")
    return _WX_MODEL


def _transcribe(audio: np.ndarray) -> str:
    model = _get_wx_model()
    result = model.transcribe(audio, batch_size=1)
    return " ".join(seg["text"] for seg in result["segments"]).strip()


# ---------------------------------------------------------------------------
# Collect first N audio paths from corpus
# ---------------------------------------------------------------------------

def _get_audio_paths(n: int) -> list[Path]:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from daws.utils.corpus_loader import PARLACHIAROLoader

    loader = PARLACHIAROLoader(CORPUS_ROOT)
    sessions = loader.load_all()
    paths = [r.audio_path for s in sessions for r in s.recordings if r.audio_path]
    return paths[:n]


# ---------------------------------------------------------------------------
# Ablation loop
# ---------------------------------------------------------------------------

def run_ablation():
    audio_paths = _get_audio_paths(N_SAMPLES)
    logger.info(f"Running ablation on {len(audio_paths)} samples, "
                f"alpha={ALPHAS}, seeds={SEEDS}")

    # Pre-compute W_original for each sample
    originals = []
    for i, path in enumerate(audio_paths):
        logger.info(f"  [{i+1}/{N_SAMPLES}] Original: {path.name}")
        audio = _load_audio(str(path))
        w_orig = _transcribe(audio)
        originals.append((path, audio, w_orig))
        logger.info(f"    W_orig: '{w_orig[:70]}'")

    print("\n" + "="*65)
    print(f"{'alpha':>8}  {'diversity_rate':>16}  {'edit_dist_mean':>16}")
    print("="*65)

    results = {}
    chosen_alpha = None

    for alpha in ALPHAS:
        diversities = []
        edit_distances = []

        for path, audio, w_orig in originals:
            for seed in SEEDS:
                audio_aug = _augment(audio, alpha, seed)
                w_aug = _transcribe(audio_aug)
                is_different = int(w_aug.strip() != w_orig.strip())
                ed = _levenshtein(w_orig, w_aug)
                diversities.append(is_different)
                edit_distances.append(ed)
                logger.debug(f"  alpha={alpha} seed={seed}: diff={is_different} ed={ed} | '{w_aug[:50]}'")

        diversity_rate = float(np.mean(diversities))
        edit_dist_mean = float(np.mean(edit_distances))
        results[alpha] = (diversity_rate, edit_dist_mean)

        marker = " ← candidate" if diversity_rate > 0.30 and chosen_alpha is None else ""
        if diversity_rate > 0.30 and chosen_alpha is None:
            chosen_alpha = alpha

        print(f"{alpha:>8.3f}  {diversity_rate:>15.1%}  {edit_dist_mean:>16.2f}{marker}")

    print("="*65)

    if chosen_alpha is not None:
        print(f"\nChosen alpha: {chosen_alpha}  "
              f"(minimum alpha with diversity_rate > 30%)")
        print(f"  diversity_rate = {results[chosen_alpha][0]:.1%}")
        print(f"  edit_dist_mean = {results[chosen_alpha][1]:.2f} words")
    else:
        print("\nNo alpha achieved diversity_rate > 30%. Consider increasing range.")
        print("Defaulting to alpha = 0.05")
        chosen_alpha = 0.05

    print(f"\nNext step: update _AUGMENT_ALPHA = {chosen_alpha} in daws/pipeline/asr.py")
    print("Then run: python main.py --mode asr-cache  (cache already cleared)")
    return chosen_alpha, results


if __name__ == "__main__":
    run_ablation()
