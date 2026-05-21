"""
pipeline/asr.py
---------------
ASR pipeline using WhisperX (ctranslate2 CPU int8) for both
the main transcript and the three diversity hypotheses W1/W2/W3.

Architecture:
  1) WhisperX .transcribe(batch_size=1) + .align()
     → main transcript + word-level confidence scores + timestamps

  2) Test-time audio augmentation → W1, W2, W3
       W1 = WhisperX(original audio)
       W2 = WhisperX(audio + Gaussian noise, σ=0.002, seed=42)
       W3 = WhisperX(audio + Gaussian noise, σ=0.002, seed=43)

     Rationale: we apply the TTA-for-UQ principle of Ayhan & Berens (2018)
     — originally proposed for medical image uncertainty via geometric/colour
     augmentations — adapted here to the audio domain via Gaussian waveform
     noise. Adding independent noise across 3 seeds samples the ASR model's
     aleatoric input uncertainty without modifying model weights.
     If W1≈W2≈W3 even under perturbation, this is a stronger overconfidence
     finding than beam-search collapse: WhisperX is robust to realistic
     acoustic variability yet still produces a systematically wrong (Italianised)
     transcription of Neapolitan dialect.

  Single model in memory — no HuggingFace Whisper, no DataLoader,
  no MPS/CUDA deadlock.

Output (ASRResult):
  transcription   — best WhisperX hypothesis (force-aligned)
  words           — [{word, start, end, confidence}]
  top3            — [W1, W2, W3] from audio augmentation
  mean_confidence — mean word-alignment confidence ∈ [0, 1]
  u_asr           — 1 - mean_confidence
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Proportional noise: noise = randn * audio.std() * alpha
# Constant SNR across speakers regardless of microphone gain / distance.
# Set empirically via daws/study/alpha_ablation.py (criterion: diversity_rate > 30%).
_AUGMENT_ALPHA = 0.005          # chosen by ablation: min alpha with diversity_rate > 30%
                                # (40.0% diversity, edit_dist_mean=1.33 words on 5-sample pilot)
# None = original audio (W1, top beam); 1, 2 = augmented seeds (W2, W3)
_AUGMENT_SEEDS = [None, 1, 2]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WordInfo:
    word: str
    start: float
    end: float
    confidence: float   # WhisperX alignment score ∈ [0, 1]


@dataclass
class ASRResult:
    transcription: str
    words: list[WordInfo]
    top3: list[str]          # [W1, W2, W3] — audio-augmented hypotheses
    language: str = "it"
    mean_confidence: float = 0.0
    u_asr: float = 0.0       # = 1 - mean_confidence

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ASRResult":
        words = [WordInfo(**w) for w in d.get("words", [])]
        return cls(
            transcription=d["transcription"],
            words=words,
            top3=d.get("top3", []),
            language=d.get("language", "it"),
            mean_confidence=d.get("mean_confidence", 0.0),
            u_asr=d.get("u_asr", 0.0),
        )


# ---------------------------------------------------------------------------
# WhisperX transcriber — single model for transcript + top-3
# ---------------------------------------------------------------------------

class WhisperXTranscriber:
    """
    WhisperX pipeline wrapping faster-whisper (ctranslate2 CPU int8).

    Main transcript: WhisperX .transcribe(batch_size=1) + .align()
    Top-3 diversity: same model, 3× with audio augmentation
    """

    def __init__(self, model_size: str = "large-v3", language: str = "it"):
        self.model_size = model_size
        self.language = language
        self._wx_model = None
        self._align_model = None
        self._align_meta = None

        import torch
        if torch.cuda.is_available():
            self._align_device = "cuda"
            self._asr_device = "cuda"
            self._asr_compute = "float16"   # ctranslate2 CUDA: float16 is fastest
        elif torch.backends.mps.is_available():
            self._align_device = "mps"
            self._asr_device = "cpu"        # ctranslate2 does not support MPS
            self._asr_compute = "int8"
        else:
            self._align_device = "cpu"
            self._asr_device = "cpu"
            self._asr_compute = "int8"

    def _load(self):
        if self._wx_model is not None:
            return
        import whisperx

        logger.info(
            f"Loading WhisperX {self.model_size} "
            f"(ASR: {self._asr_device}/{self._asr_compute}, align: {self._align_device}) ..."
        )
        self._wx_model = whisperx.load_model(
            self.model_size,
            device=self._asr_device,
            compute_type=self._asr_compute,
            language=self.language,
        )
        self._align_model, self._align_meta = whisperx.load_align_model(
            language_code=self.language, device=self._align_device
        )
        logger.info("WhisperX loaded.")

    def _transcribe_array(self, audio: np.ndarray) -> str:
        """Run WhisperX transcription (no alignment) on a numpy audio array."""
        result = self._wx_model.transcribe(audio, batch_size=1)
        return " ".join(seg["text"] for seg in result["segments"]).strip()

    def transcribe(self, audio_path: str) -> tuple[str, list[WordInfo], str]:
        """
        Main transcription with word-level alignment.
        Returns (transcript, word_list, language_code).
        """
        self._load()
        import whisperx

        audio = whisperx.load_audio(audio_path)
        result = self._wx_model.transcribe(audio, batch_size=1)
        transcript = " ".join(seg["text"] for seg in result["segments"]).strip()
        language = result.get("language", self.language)

        result_aligned = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_meta,
            audio,
            self._align_device,
            return_char_alignments=False,
        )

        words = []
        for seg in result_aligned.get("segments", []):
            for w in seg.get("words", []):
                word = w.get("word", "").strip()
                if not word:
                    continue
                words.append(WordInfo(
                    word=word,
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    confidence=float(w.get("score", 0.5)),
                ))

        return transcript, words, language

    def transcribe_top3(self, audio_path: str) -> list[str]:
        """
        Generate W1, W2, W3 via test-time audio augmentation.

          W1 = WhisperX(original audio)
          W2 = WhisperX(audio + Gaussian noise, σ=0.002, seed=42)
          W3 = WhisperX(audio + Gaussian noise, σ=0.002, seed=43)

        Following Ayhan & Berens (2018) — TTA-for-UQ adapted to audio domain.
        If W1≈W2≈W3 under perturbation: publishable overconfidence finding.
        """
        self._load()
        import whisperx

        audio = whisperx.load_audio(audio_path)   # float32 numpy array @ 16 kHz

        top3 = []
        for seed in _AUGMENT_SEEDS:
            if seed is None:
                audio_input = audio   # W1 = original top beam, no perturbation
            else:
                rng = np.random.default_rng(seed)
                noise = rng.standard_normal(len(audio)).astype(np.float32) * audio.std() * _AUGMENT_ALPHA
                audio_input = audio + noise   # no clipping — preserve dialectal prosody
            try:
                text = self._transcribe_array(audio_input)
                top3.append(text)
            except Exception as e:
                logger.warning(f"Augmented transcription failed (seed={seed}): {e}")
                top3.append(top3[0] if top3 else "")

        # Pad to 3 in the unlikely case of repeated failures
        fallback = top3[0] if top3 else ""
        while len(top3) < 3:
            top3.append(fallback)

        return top3[:3]


# ---------------------------------------------------------------------------
# Combined ASR pipeline
# ---------------------------------------------------------------------------

class ASRPipeline:
    """
    ASR pipeline — single WhisperX model.
      WhisperX + alignment → main transcript + word confidence
      WhisperX × 3 (audio augmentation) → W1, W2, W3
    """

    def __init__(self, language: str = "it"):
        self.language = language
        self._transcriber = WhisperXTranscriber(language=language)

    def process(self, audio_path: str, cache_path: Optional[str] = None) -> ASRResult:
        """
        Process a WAV file and return a complete ASRResult.

        Args:
            audio_path: path to WAV file
            cache_path: if set, load from JSON cache if it exists, else save after processing
        """
        if cache_path and Path(cache_path).exists():
            logger.info(f"ASR cache hit: {cache_path}")
            with open(cache_path, encoding="utf-8") as f:
                return ASRResult.from_dict(json.load(f))

        logger.info(f"ASR processing: {audio_path}")

        transcript, words, language = self._transcriber.transcribe(audio_path)
        logger.info(f"  Transcript: '{transcript[:80]}'")

        logger.info("  Top-3 audio augmentation ...")
        top3 = self._transcriber.transcribe_top3(audio_path)
        logger.info(f"  W1: '{top3[0][:60]}'")
        logger.info(f"  W2: '{top3[1][:60]}'")
        logger.info(f"  W3: '{top3[2][:60]}'")

        confidences = [w.confidence for w in words]
        mean_conf = float(np.mean(confidences)) if confidences else 0.5

        result = ASRResult(
            transcription=transcript,
            words=words,
            top3=top3,
            language=language,
            mean_confidence=mean_conf,
            u_asr=1.0 - mean_conf,
        )

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"  Cached: {cache_path}")

        return result
