"""
pipeline/llm.py
---------------
LLM wrapper for DAWS offline studies — Ollama HTTP API only.
Used by daws/study/inv_entropy_study.py for generating 6 responses per recording.

Production inference uses _ollama_generate() in daws/pipeline/daws.py directly.
"""

import logging

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "mistral"
OLLAMA_URL   = "http://localhost:11434/api/generate"


class OllamaLLM:
    """
    Lightweight wrapper for Ollama HTTP API (localhost:11434).
    Uses Metal-optimized GGUF — proper Apple Silicon inference, no MPS deadlocks.
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        import requests as _req
        self._requests = _req

    def generate_text(self, prompt: str, max_new_tokens: int = 120) -> str:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0.0,
                "seed": 0,
                "top_p": 1.0,
            },
        }
        resp = self._requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
