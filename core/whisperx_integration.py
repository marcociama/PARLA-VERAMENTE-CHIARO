"""
whisperx_integration.py
------------------------
Interfaccia con WhisperX per estrarre confidence logit-level per token.

WhisperX aggiunge word-level timestamps e confidence scores su Whisper Large V3.
Usiamo queste confidence per il modulo di detection dialettale.

Detection = AND logico (come da design PARLA CHIARO):
  segnale_dialettale = (confidence_whisperX < soglia_ASR) AND (token_OOV_UmBERTo)

I due segnali sono complementari:
- WhisperX confidence: bassa quando il modello ASR è incerto sul token
- OOV UmBERTo: alto quando il token non è italiano standard
  
Il loro AND riduce i falsi positivi (parole rare italiane che WhisperX trascrive
male ≠ dialetto, a meno che non siano anche OOV per il vocabolario italiano).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.device import get_ctranslate2_device, get_torch_device

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenSignal:
    token: str
    start_time: float
    end_time: float
    asr_confidence: float        # WhisperX word-level confidence ∈ [0, 1]
    is_oov_italian: bool         # OOV rispetto al vocabolario UmBERTo/mDeBERTa
    is_dialect_signal: bool      # AND logico: bassa conf + OOV
    char_start: int = 0          # posizione nel testo trascritto
    char_end: int = 0


@dataclass
class ASROutput:
    transcript: str
    language_detected: str
    tokens: list[TokenSignal] = field(default_factory=list)

    @property
    def dialect_risk_tokens(self) -> list[TokenSignal]:
        return [t for t in self.tokens if t.is_dialect_signal]

    @property
    def overall_dialect_risk(self) -> float:
        """
        Frazione di token flaggati come dialettali.
        Soglia per alert: > 0.2 (20% dei token flaggati → input probabilmente dialettale).
        """
        if not self.tokens:
            return 0.0
        return len(self.dialect_risk_tokens) / len(self.tokens)


# ──────────────────────────────────────────────────────────────────────────────
# WhisperX wrapper
# ──────────────────────────────────────────────────────────────────────────────

class WhisperXTranscriber:
    """
    Wrapper per WhisperX con estrazione di confidence per token.
    
    WhisperX: https://github.com/m-bain/whisperX
    Installa: pip install whisperx
    
    Aggiunge forced alignment (wav2vec2) per word-level timestamps
    e confidence rispetto a Whisper base.
    """

    ASR_CONFIDENCE_THRESHOLD = 0.6   # soglia bassa confidence per segnale dialettale

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str | None = None,
        compute_type: str | None = None,
        language: str = "it",
    ):
        # ctranslate2 (faster-whisper) non supporta MPS → CPU con int8 su Apple Silicon
        ct2_device, ct2_compute = get_ctranslate2_device()
        self.model_size = model_size
        self.device = device or ct2_device
        self.compute_type = compute_type or ct2_compute
        self.language = language
        self._align_device = get_torch_device()   # allineamento wav2vec2 usa MPS
        self._model = None
        self._align_model = None

    def _load_models(self):
        """Carica WhisperX on-demand (modelli grandi, carica una volta sola)."""
        if self._model is not None:
            return

        try:
            import whisperx
            logger.info(
                f"Carico WhisperX {self.model_size} "
                f"(ASR: {self.device}/{self.compute_type}, align: {self._align_device})..."
            )
            self._model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=self.language,
            )
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self._align_device,
            )
            logger.info("WhisperX caricato.")
        except ImportError:
            raise ImportError(
                "WhisperX non installato. Esegui: pip install whisperx\n"
                "Richiede anche: ffmpeg e un modello wav2vec2 per l'italiano."
            )

    def transcribe(self, audio_path: str | Path) -> ASROutput:
        """
        Trascrivi un file audio e restituisce ASROutput con confidence per token.
        """
        import whisperx

        self._load_models()
        audio = whisperx.load_audio(str(audio_path))

        # Step 1: trascrizione Whisper Large V3
        result = self._model.transcribe(audio, batch_size=16)
        transcript = " ".join(seg["text"] for seg in result["segments"]).strip()
        language = result.get("language", "it")

        # Step 2: forced alignment per word-level timestamps + confidence
        result_aligned = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self._align_device,
            return_char_alignments=False,
        )

        tokens = self._extract_tokens(result_aligned)
        return ASROutput(transcript=transcript, language_detected=language, tokens=tokens)

    def _extract_tokens(self, aligned_result: dict) -> list[TokenSignal]:
        """Estrae token con confidence dalla struttura allineata di WhisperX."""
        tokens = []
        char_pos = 0

        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                word = word_info.get("word", "").strip()
                if not word:
                    continue

                # WhisperX fornisce 'score' come confidence del forced alignment
                confidence = float(word_info.get("score", 0.5))
                start = float(word_info.get("start", 0.0))
                end = float(word_info.get("end", 0.0))

                token_sig = TokenSignal(
                    token=word,
                    start_time=start,
                    end_time=end,
                    asr_confidence=confidence,
                    is_oov_italian=False,   # verrà settato dall'OOV detector
                    is_dialect_signal=False,
                    char_start=char_pos,
                    char_end=char_pos + len(word),
                )
                tokens.append(token_sig)
                char_pos += len(word) + 1   # +1 per spazio

        return tokens


# ──────────────────────────────────────────────────────────────────────────────
# OOV Detector (UmBERTo / mDeBERTa)
# ──────────────────────────────────────────────────────────────────────────────

class OOVDetector:
    """
    Rileva token OOV rispetto al vocabolario di un modello italiano.
    
    UmBERTo è addestrato su italiano standard → token napoletani
    (apocope, lessemi dialettali) appaiono come [UNK] o sub-word OOV.
    
    Usiamo la tokenizzazione come proxy per OOV:
    - Se il token viene tokenizzato in ≥ 3 subword → probabilmente OOV
    - Se il token è [UNK] direttamente → certamente OOV
    
    Alternativa più robusta: usare mDeBERTa con il vocabolario condiviso
    per avere copertura multilingue.
    """

    # Soglia: token con ≥ N subword sono considerati OOV
    SUBWORD_OOV_THRESHOLD = 3

    def __init__(self, model_name: str = "microsoft/mdeberta-v3-base"):
        from transformers import AutoTokenizer
        logger.info(f"Carico tokenizer OOV: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.unk_id = self.tokenizer.unk_token_id

    def is_oov(self, token: str) -> bool:
        """
        Ritorna True se il token è OOV per il vocabolario italiano standard.
        """
        token_clean = token.strip().lower()
        if not token_clean or not token_clean.isalpha():
            return False   # punteggiatura e numeri non sono OOV nel senso dialettale

        token_ids = self.tokenizer.encode(token_clean, add_special_tokens=False)

        # Check 1: token diventa [UNK]
        if self.unk_id in token_ids:
            return True

        # Check 2: token viene segmentato in molte subword (proxy per OOV)
        if len(token_ids) >= self.SUBWORD_OOV_THRESHOLD:
            return True

        return False

    def annotate_tokens(self, asr_output: ASROutput) -> ASROutput:
        """
        Aggiorna in-place i token dell'ASROutput con i flag OOV e dialect_signal.
        """
        for token_sig in asr_output.tokens:
            token_sig.is_oov_italian = self.is_oov(token_sig.token)
            # AND logico: bassa confidence ASR + token OOV
            token_sig.is_dialect_signal = (
                token_sig.asr_confidence < WhisperXTranscriber.ASR_CONFIDENCE_THRESHOLD
                and token_sig.is_oov_italian
            )

        return asr_output


# ──────────────────────────────────────────────────────────────────────────────
# High-level integration
# ──────────────────────────────────────────────────────────────────────────────

class DialectSignalExtractor:
    """
    Combina WhisperX + OOV detector per il primo modulo della pipeline PARLA CHIARO.
    Output: ASROutput con token annotati + flag di rischio dialettale.
    """

    def __init__(
        self,
        whisperx: Optional[WhisperXTranscriber] = None,
        oov_detector: Optional[OOVDetector] = None,
    ):
        self.transcriber = whisperx or WhisperXTranscriber()
        self.oov_detector = oov_detector or OOVDetector()

    def process_audio(self, audio_path: str | Path) -> ASROutput:
        """
        Pipeline completa: audio → trascrizione → annotazione dialettale.
        """
        logger.info(f"Processando: {audio_path}")
        asr_output = self.transcriber.transcribe(audio_path)
        asr_output = self.oov_detector.annotate_tokens(asr_output)

        n_dialect = len(asr_output.dialect_risk_tokens)
        logger.info(
            f"Trascrizione: '{asr_output.transcript[:60]}...' | "
            f"Token dialettali: {n_dialect}/{len(asr_output.tokens)} "
            f"({asr_output.overall_dialect_risk:.1%})"
        )

        return asr_output
