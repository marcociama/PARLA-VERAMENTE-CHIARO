"""
corpus_loader.py
----------------
Loader per il corpus PARLA CHIARO (subset 50 registrazioni).

Schema reale da recordings.json / participants.json:
- participantId: "{data}_{ora}_{dialetto}_{sesso}_{fascia_età}" → anche nome sottocartella
- promptText: ground truth già disponibile! Non serve Whisper per la reference
- promptCategory: categoria tematica (utile per breakdown dell'analisi)

Struttura attesa su disco:
  data_root/
    recordings/
      {participantId}/
        {filename}.wav
        {filename}.json   ← metadati singola registrazione
    participants/
      {participantId}.json
    recordings.json       ← schema (non dati)
    participants.json     ← schema (non dati)

Insight chiave: promptText è la trascrizione corretta → ground truth gratuito.
L'overconfidence analysis diventa:
  input_A = promptText     (testo corretto, quello che il parlante ha letto)
  input_B = whisper_output (quello che Whisper ha trascritto)
  confronta SE(A) vs SE(B)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Participant:
    participant_id: str
    gender: str
    age_range: str
    dialect: str
    dialect_other: str
    education: str
    living_context: str
    dialect_frequency: str
    created_at: str

    @property
    def is_neapolitan(self) -> bool:
        d = self.dialect.lower()
        return "napol" in d or "campan" in d

    @property
    def dialect_label(self) -> str:
        """Label normalizzato per aggregazione."""
        if self.is_neapolitan:
            return "napoletano"
        if self.dialect_other:
            return self.dialect_other.lower()
        return self.dialect.lower()


@dataclass
class Recording:
    participant_id: str
    prompt_id: str
    prompt_text: str          # ← GROUND TRUTH già disponibile
    prompt_category: str
    recording_index: int
    filename: str
    size: int
    mimetype: str
    uploaded_at: str
    audio_path: Optional[Path] = None

    # Aggiunto dopo transcription con WhisperX
    whisper_transcript: Optional[str] = None
    whisper_tokens: Optional[list[dict]] = None   # [{token, confidence, start, end}]

    @property
    def duration_estimate_s(self) -> float:
        """Stima durata da size (WAV 16kHz mono 16bit ≈ 32000 byte/s)."""
        return self.size / 32000 if self.size else 0.0


@dataclass
class CorpusSession:
    participant: Participant
    recordings: list[Recording] = field(default_factory=list)

    @property
    def n_recordings(self) -> int:
        return len(self.recordings)


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

class PARLACHIAROLoader:
    """
    Carica il corpus PARLA CHIARO dalla struttura su disco.
    """

    def __init__(self, data_root: str | Path):
        self.root = Path(data_root)
        self.recordings_dir = self.root / "recordings"
        self.participants_dir = self.root / "participants"

        if not self.root.exists():
            raise FileNotFoundError(f"Data root non trovato: {self.root}")

    def load_all(self) -> list[CorpusSession]:
        """Carica tutti i partecipanti e le loro registrazioni."""
        sessions = []
        participant_files = sorted(self.participants_dir.glob("*.json"))

        if not participant_files:
            raise FileNotFoundError(f"Nessun file partecipante in {self.participants_dir}")

        for p_file in participant_files:
            try:
                participant = self._load_participant(p_file)
                recordings = self._load_recordings(participant.participant_id)
                sessions.append(CorpusSession(participant=participant, recordings=recordings))
            except Exception as e:
                logger.warning(f"Errore caricamento {p_file.stem}: {e}")
                continue

        logger.info(
            f"Caricati {len(sessions)} partecipanti, "
            f"{sum(s.n_recordings for s in sessions)} registrazioni totali."
        )
        return sessions

    def _load_participant(self, path: Path) -> Participant:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return Participant(
            participant_id=d["participantId"],
            gender=d.get("gender", ""),
            age_range=d.get("ageRange", ""),
            dialect=d.get("dialect", ""),
            dialect_other=d.get("dialectOther", ""),
            education=d.get("education", ""),
            living_context=d.get("livingContext", ""),
            dialect_frequency=d.get("dialectFrequency", ""),
            created_at=d.get("createdAt", ""),
        )

    def _load_recordings(self, participant_id: str) -> list[Recording]:
        session_dir = self.recordings_dir / participant_id
        if not session_dir.exists():
            logger.warning(f"Cartella registrazioni non trovata: {session_dir}")
            return []

        recordings = []
        for json_file in sorted(session_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    d = json.load(f)

                # Trova il WAV corrispondente
                wav_name = d.get("filename", json_file.stem + ".wav")
                audio_path = session_dir / wav_name
                if not audio_path.exists():
                    # Prova con stesso nome del JSON
                    audio_path = json_file.with_suffix(".wav")

                # promptText può essere stringa o lista di varianti → usa la prima
                raw_prompt = d.get("promptText", "")
                if isinstance(raw_prompt, list):
                    prompt_text_val = raw_prompt[0] if raw_prompt else ""
                else:
                    prompt_text_val = raw_prompt

                rec = Recording(
                    participant_id=d["participantId"],
                    prompt_id=str(d.get("promptId", "")),
                    prompt_text=prompt_text_val,
                    prompt_category=d.get("promptCategory", ""),
                    recording_index=int(d.get("recordingIndex", 0)),
                    filename=d.get("filename", ""),
                    size=int(d.get("size", 0)),
                    mimetype=d.get("mimetype", "audio/wav"),
                    uploaded_at=d.get("uploadedAt", ""),
                    audio_path=audio_path if audio_path.exists() else None,
                )
                recordings.append(rec)
            except Exception as e:
                logger.warning(f"Errore parsing {json_file}: {e}")
                continue

        return sorted(recordings, key=lambda r: r.recording_index)

    # ── Utilità di analisi rapida ──────────────────────────────────────────

    def stats(self, sessions: list[CorpusSession]) -> dict:
        """Statistiche rapide sul corpus caricato."""
        all_recordings = [r for s in sessions for r in s.recordings]
        dialects = {}
        for s in sessions:
            dl = s.participant.dialect_label
            dialects[dl] = dialects.get(dl, 0) + 1

        age_ranges = {}
        for s in sessions:
            ar = s.participant.age_range
            age_ranges[ar] = age_ranges.get(ar, 0) + 1

        categories = {}
        for r in all_recordings:
            cat = r.prompt_category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "n_participants": len(sessions),
            "n_recordings": len(all_recordings),
            "n_with_audio": sum(1 for r in all_recordings if r.audio_path),
            "dialects": dict(sorted(dialects.items(), key=lambda x: -x[1])),
            "age_ranges": age_ranges,
            "prompt_categories": categories,
            "n_neapolitan": sum(1 for s in sessions if s.participant.is_neapolitan),
        }
