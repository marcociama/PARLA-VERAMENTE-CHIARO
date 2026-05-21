"""
pipeline/llm.py
---------------
Local causal LM wrapper for DAWS — provides token-level logits required by
Semantic Density (Qiu & Miikkulainen, NeurIPS 2024).

Default model: mistralai/Mistral-7B-Instruct-v0.3  (~14 GB)
  Download once with: huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3

Device strategy:
  MPS  (Apple Silicon) — float16
  CUDA               — float16
  CPU                — float32
  bitsandbytes 4-bit quantization is NOT supported on MPS → not used.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _resolve_model_id(model_id: Optional[str]) -> str:
    return model_id if model_id is not None else MODEL_ID


# ---------------------------------------------------------------------------
# Ollama LLM — fast local inference via Metal-optimized GGUF
# ---------------------------------------------------------------------------

class OllamaLLM:
    """
    Lightweight wrapper for Ollama HTTP API (localhost:11434).
    Uses Metal-optimized GGUF — proper Apple Silicon inference, no MPS deadlocks.
    Only implements generate_text() for ie-study use.
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        import requests as _req  # lazy import
        self._requests = _req

    def generate_text(self, prompt: str, max_new_tokens: int = 60) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        }
        resp = self._requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenInfo:
    token: str
    logprob: float   # log P(token | context)
    prob: float      # P(token | context)
    entropy: float   # H at this decoding step over the full vocabulary


@dataclass
class LLMSingleResult:
    text: str
    tokens: list[TokenInfo]
    length_norm_logprob: float   # (1/L) * sum(logprob_i)  — Eq. 1 in SD paper
    length_norm_prob: float      # exp(length_norm_logprob)


@dataclass
class LLMDiverseResult:
    """M diverse responses for Semantic Density reference sampling."""
    responses: list[LLMSingleResult]


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

class LocalLLM:
    """
    Causal LM with logit access for Semantic Density.

    Supports any HuggingFace AutoModelForCausalLM.
    Diverse beam search (Vijayakumar et al.) provides M reference responses
    with controlled diversity for SD Algorithm 1 Step 1.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,  # None → Mistral 7B
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ):
        self.model_id = _resolve_model_id(model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._device = _get_device()

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading {self.model_id} on {self._device} ...")
        dtype = torch.float16 if self._device.type in ("mps", "cuda") else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self._device)
        self._model.eval()
        logger.info(f"Model loaded — {self.model_id} on {self._device} ({dtype}).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, user_message: str) -> str:
        """Format prompt using the model's chat template if available."""
        self._load()
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            messages = [{"role": "user", "content": user_message}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: Mistral-style
        return f"[INST] {user_message} [/INST]"

    def _decode_scores(
        self,
        sequences: torch.Tensor,      # (B, total_seq_len)
        scores: tuple[torch.Tensor],  # tuple of (B, vocab_size) per decoding step
        input_len: int,
    ) -> list[LLMSingleResult]:
        """Convert sequences + logit scores into LLMSingleResult objects."""
        results = []
        eos_id = self._tokenizer.eos_token_id

        for b in range(sequences.shape[0]):
            generated_ids = sequences[b, input_len:].tolist()

            # Trim at first EOS
            if eos_id in generated_ids:
                generated_ids = generated_ids[:generated_ids.index(eos_id)]

            L = len(generated_ids)
            if L == 0:
                continue

            token_infos: list[TokenInfo] = []
            sum_logprob = 0.0

            for step_idx in range(min(L, len(scores))):
                logits_step = scores[step_idx][b]  # (vocab_size,)

                if self.temperature != 1.0:
                    logits_step = logits_step / self.temperature

                log_probs = F.log_softmax(logits_step, dim=-1)
                token_id = generated_ids[step_idx]
                token_logprob = log_probs[token_id].item()
                token_prob = math.exp(token_logprob)

                # Step entropy over top-1000 for efficiency
                top_lp, _ = torch.topk(log_probs, k=min(1000, log_probs.shape[-1]))
                step_entropy = -(top_lp.exp() * top_lp).sum().item()

                token_infos.append(TokenInfo(
                    token=self._tokenizer.decode([token_id], skip_special_tokens=False),
                    logprob=token_logprob,
                    prob=token_prob,
                    entropy=step_entropy,
                ))
                sum_logprob += token_logprob

            length_norm_logprob = sum_logprob / L
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            results.append(LLMSingleResult(
                text=text,
                tokens=token_infos,
                length_norm_logprob=length_norm_logprob,
                length_norm_prob=math.exp(length_norm_logprob),
            ))

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_single(self, prompt: str) -> LLMSingleResult:
        """
        Greedy decode — returns best response with per-token logprobs.
        Used by SD to obtain y* and its token-level probabilities.
        """
        self._load()
        formatted = self._apply_chat_template(prompt)
        inputs = self._tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        results = self._decode_scores(outputs.sequences, outputs.scores, input_len)
        if not results:
            return LLMSingleResult("", [], -float("inf"), 0.0)
        return results[0]

    def generate_diverse(self, prompt: str, M: int = 5) -> LLMDiverseResult:
        """
        Diverse beam search (Vijayakumar et al.) — returns M reference responses
        with length-normalized probabilities for SD Algorithm 1 Step 1.

        num_beams = M*2, num_beam_groups = M, diversity_penalty = 1.0
        """
        self._load()
        formatted = self._apply_chat_template(prompt)
        inputs = self._tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=M * 2,
                num_beam_groups=M,
                num_return_sequences=M,
                diversity_penalty=1.0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        results = self._decode_scores(outputs.sequences, outputs.scores, input_len)

        # Deduplicate while preserving order
        seen, unique = set(), []
        for r in results:
            if r.text not in seen:
                seen.add(r.text)
                unique.append(r)

        # Pad to M if needed
        while len(unique) < M:
            unique.append(unique[-1] if unique else LLMSingleResult("", [], -float("inf"), 0.0))

        return LLMDiverseResult(responses=unique[:M])

    def generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Fast greedy decode — returns text only, no logprobs.
        Use this in ie-study instead of generate_single to skip the
        output_scores=True overhead (256×32K logit tensors per call).
        """
        self._load()
        formatted = self._apply_chat_template(prompt)
        inputs = self._tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=False,
                return_dict_in_generate=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0, input_len:].tolist()
        eos_id = self._tokenizer.eos_token_id
        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def token_entropy(self, prompt: str) -> list[float]:
        """Per-token decoding entropy — used for token-level uncertainty heatmap."""
        result = self.generate_single(prompt)
        return [t.entropy for t in result.tokens]

    def unload(self):
        """Release GPU/MPS memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")
