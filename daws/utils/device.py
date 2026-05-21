"""
device.py
---------
Centralized device detection per PARLA CHIARO.

Gerarchia: MPS (Apple Silicon) > CUDA > CPU
Nota: ctranslate2 (WhisperX) non supporta MPS → usa sempre CPU con int8.
      I modelli transformers (DeBERTa, UmBERTo) supportano MPS pienamente.
"""

import torch


def get_torch_device() -> str:
    """
    Device per modelli transformers (NLI, OOV).
    Ritorna 'mps', 'cuda' o 'cpu'.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_transformers_device_id() -> int | str:
    """
    Device nel formato atteso da transformers.pipeline():
    - 'mps' per Apple Silicon
    - 0 per prima GPU CUDA
    - -1 per CPU
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return -1


def get_ctranslate2_device() -> tuple[str, str]:
    """
    Device e compute_type per ctranslate2 / faster-whisper.
    ctranslate2 non supporta MPS → CPU con int8 (ottimale su Apple Silicon ARM).
    Ritorna (device, compute_type).
    """
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def device_info() -> dict:
    d, ct = get_ctranslate2_device()
    return {
        "torch_device": get_torch_device(),
        "transformers_device": get_transformers_device_id(),
        "ctranslate2_device": d,
        "ctranslate2_compute_type": ct,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
