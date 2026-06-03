"""
scripts/cacioli_study_v2.py
----------------------------
Identico a cacioli_study.py ma con prompt Mistral più neutro:
  "Riassumi in una frase il significato di questa affermazione."

Cache separata: scripts/cacioli_llm_cache_v2/
Risultati:      scripts/cacioli_results_v2.json

I file di cacioli_study.py (v1) NON vengono toccati.
ASR cache e wav riutilizzati da v1 (stessi audio, stesso WhisperX).
"""

import sys
from pathlib import Path

# ── Override constants prima di importare cacioli_study ───────────────────────
# Patch al volo: importiamo il modulo v1 e sovrascriviamo le costanti
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import scripts.cacioli_study as _cs

# Nuova cache LLM — separata da v1
_cs.CACHE_DIR   = BASE / "scripts" / "cacioli_llm_cache_v2"
_cs.RESULTS_OUT = BASE / "scripts" / "cacioli_results_v2.json"

# Riutilizza ASR cache e wav di v1 (stessi audio, non serve riprocessare)
# _cs.ASR_CACHE e _cs.WAV_DIR rimangono invariati

# Nuovo prompt: neutro, content-based, non interpretativo
_cs._CULTURAL_PROMPT = (
    "Riassumi in una frase il significato di questa affermazione.\n\n"
    "Affermazione: {text}\n\nRiassunto:"
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cacioli study v2 — prompt neutro")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None)
    args = parser.parse_args()

    if args.phase == 1:
        _cs.phase1()
    elif args.phase == 2:
        _cs.phase2()
    else:
        done  = len(list(_cs.CACHE_DIR.glob("*.json"))) if _cs.CACHE_DIR.exists() else 0
        total = len(list(_cs.AUDIO_DIR.glob("*.m4a")))
        if done < total:
            print(f"Auto-detect: {done}/{total} → Phase 1")
            _cs.phase1()
        else:
            print(f"Auto-detect: {done}/{total} → Phase 2")
            _cs.phase2()
