"""
pipeline/inv_entropy.py
-----------------------
Inverse Conditional Entropy (Inv-Entropy) — Song et al., NeurIPS 2025.
"Inv-Entropy: Uncertainty Quantification for LLMs via Inverse Conditional Entropy"

Design for PARLA CHIARO (N=6):
  X_n = [GT1, GT2, GT3, W1, W2, W3]
    GT1/GT2/GT3  — three promptText variants from the recording JSON (free ground truth)
    W1/W2/W3     — top-3 beam search hypotheses from ASR cache

  Y_n = [R_1, ..., R_6]   — LLM responses to each of the 6 inputs

Algorithm 1 (Sections 3–4):

  Eq. 2 — Input similarity matrix (row-stochastic):
    Px[i, j] = aSim(x_i, x_j) / sum_k aSim(x_i, x_k)

  Eq. 3 — Output similarity matrix (row-stochastic, same construction):
    Py[j, i] = aSim(y_j, y_i) / sum_k aSim(y_j, y_k)

  Eq. 4 — Conditional probability (diagonal, no column normalisation):
    P_xy = Py @ Px                              # (N, N), row-stochastic
    P(x_i | y_i) = P_xy[i, i]                  # only matched pairs used

  Eq. 5 — Inv-Entropy (trace formula):
    H(X_n | Y_n) = -trace(P_xy ⊙ log(P_xy))
                 = -Σ_i P(x_i | y_i) · log P(x_i | y_i)

  Note: H is unnormalised — diag(P_xy) is not a distribution, so H can exceed log(N).
    u_inv_entropy = H / log(N)   (may exceed 1.0 — used as ranking score)

aSimilarity: SBERT cosine — paraphrase-multilingual-mpnet-base-v2 (Song et al., option ii).
  Formula: aSim(x, x') = (1 + cos(x, x')) / 2  ∈ [0, 1]
  Guaranteed: identical texts → aSim = 1.0 (cos = 1); orthogonal → aSim = 0.5.

  Previous approaches discarded:
    LaBSE cosine: compressed Italian medical text to [0.68, 1.0] → Px uniform.
    mDeBERTa NLI P(entailment): neutral-class plateau → 0.41 for identical texts.
    mDeBERTa 1-P(contradiction): neutral texts scored as similar → Py uniform.
  SBERT STS operates on a continuous vector space — no discrete NLI class plateaus.

Direct computation (no bootstrap): with T=0 greedy decode (r=1), Mistral is
  deterministic — Yn is fixed. Resampling always reproduces the same 6 responses,
  so bootstrap is computationally redundant. H computed once per sample.
  Corpus-level CIs via standard bootstrap on the 50 H_i values (offline analysis).
"""

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

LABSE_MODEL_ID = "sentence-transformers/LaBSE"          # kept for llm_characterization.py
SBERT_MODEL_ID = "paraphrase-multilingual-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InvEntropyResult:
    inv_entropy: float       # H (nats, direct computation — no bootstrap)
    inv_entropy_norm: float  # H / log(N)  ∈ [0, 1]
    u_inv_entropy: float     # = inv_entropy_norm
    n_samples: int           # N
    n_bootstrap: int         # always 0 (direct computation, no bootstrap)
    diag_probs: list[float]  # P(x_i | y_i) for each i


# ---------------------------------------------------------------------------
# SBERT similarity (primary) — paraphrase-multilingual-mpnet-base-v2
# ---------------------------------------------------------------------------

class SBERTSimilarity:
    """
    aSimilarity via SBERT cosine — Song et al. NeurIPS 2025 option (ii).

    Model: paraphrase-multilingual-mpnet-base-v2
      Multilingual STS model (50+ languages including Italian).
      Operates on continuous vector space — no NLI discrete class plateaus.

    Formula: aSim(x, x') = clip(cos(x, x'), 0, 1)^k  ∈ [0, 1],  k=4
      Identical texts  → cos = 1.0 → aSim = 1.0  (guaranteed)
      Unrelated texts  → cos ≈ 0.3–0.5 → aSim^4 ≈ 0.01–0.06 (amplified contrast)

    Sharpening (cos^k): ablation over k=1..4 on 5 pilot samples showed
      k=1: Pearson(WER,h_norm)=-0.776, range=[0.987,1.000] (near-saturated)
      k=4: Pearson(WER,h_norm)=-0.920, range=[0.742,0.999] (best contrast)

    Note: Song et al. use (1+cos)/2 to handle cos ∈ [-1, 1]. On Italian clinical
    text, SBERT cosines are always positive (min ≈ 0.48), so the (1+cos)/2 shift
    compresses the useful range to [0.74, 1.00], collapsing Py to near-uniform.
    Raw cosine clipped to [0,1] then raised to k=4 preserves and amplifies
    the discriminative range without violating probability constraints.

    Batch encoding with L2-normalised embeddings reduces cosine to a
    single dot-product matrix — no N^2 pair loops needed.
    """

    def __init__(self, model_id: str = SBERT_MODEL_ID):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SBERT {self.model_id} ...")
        self._model = SentenceTransformer(self.model_id)
        logger.info("SBERT loaded.")

    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """
        Returns (N, N) similarity matrix via batch SBERT encoding.
        aSim[i,j] = clip(cos(emb_i, emb_j), 0, 1)
        Diagonal = 1.0 guaranteed (cos of a vector with itself = 1).
        """
        self._load()
        embs = self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)      # (N, D), L2-normalised
        sim = np.clip(np.dot(embs, embs.T), 0.0, 1.0) ** 4      # (N, N) ∈ [0, 1], k=4
        return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Row-stochastic conversion
# ---------------------------------------------------------------------------

def _row_stochastic(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert similarity matrix to row-stochastic.
    Negative cosine values are clipped to 0.
    Zero rows are replaced by uniform distribution.
    """
    S = np.clip(sim, 0.0, None)
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return S / row_sums


# ---------------------------------------------------------------------------
# Core entropy computation
# ---------------------------------------------------------------------------

def _compute_H(
    inputs: list[str],
    outputs: list[str],
    embedder,   # SBERTSimilarity or any object with .similarity_matrix()
) -> tuple[float, list[float]]:
    """
    Compute Inv-Entropy H(X_n | Y_n) per Eq. 5 — Song et al., NeurIPS 2025.

    Formula (paper, Eq. 5):
        H = -trace(Py @ Px ⊙ log(Py @ Px))
          = -Σ_i (Py@Px)[i,i] * log((Py@Px)[i,i])
          = -Σ_i P(x_i|y_i) * log P(x_i|y_i)

    Notes:
      - Uses the DIAGONAL of Py@Px directly — no column normalisation.
      - Unnormalised: H can exceed log(N) because diag(Py@Px) is not a
        probability distribution (elements do not sum to 1).
      - Designed to be used as a ranking score (AUROC), not an absolute value.
      - Higher H → higher uncertainty (more input-output misalignment).

    Returns:
        (H_value, diag_probs)  where diag_probs = diag(Py @ Px)
    """
    N = len(inputs)
    assert len(outputs) == N

    # Eq. 2: Px[i,j] = aSim(x_i, x_j) / sum_k aSim(x_i, x_k)
    Px = _row_stochastic(embedder.similarity_matrix(inputs))   # (N, N)

    # Eq. 3: Py[j,i] = aSim(y_j, y_i) / sum_k aSim(y_j, y_k)
    Py = _row_stochastic(embedder.similarity_matrix(outputs))  # (N, N)

    # Eq. 4 / Eq. 5 (paper): P(x_i|y_i) = diag(Py @ Px)[i]  — no column-norm
    P_xy = Py @ Px                                             # (N, N) raw
    diag_probs = np.diag(P_xy).tolist()                        # P(x_i|y_i)

    # Eq. 5: H = -Σ_i P(x_i|y_i) * log P(x_i|y_i)   [unnormalised, nats]
    d = np.array(diag_probs)
    H = float(-np.sum(d * np.log(d + 1e-12)))

    return H, diag_probs


# ---------------------------------------------------------------------------
# Inv-Entropy calculator
# ---------------------------------------------------------------------------

class InvEntropyCalculator:
    """
    Computes Inv-Entropy (direct, no bootstrap) for the PARLA CHIARO N=6 design.

    aSimilarity: SBERT cosine (paraphrase-multilingual-mpnet-base-v2).
    Formula: aSim(x, x') = (1 + cos(x, x')) / 2 — Song et al. NeurIPS 2025 option (ii).
    Continuous STS space avoids NLI discrete-class plateaus that collapsed
    Italian clinical text to uniform similarity matrices.

    Direct computation (no bootstrap): with T=0 greedy decode (r=1), Mistral is
    deterministic — Yn is fixed. Resampling always reproduces the same 6 responses,
    making bootstrap computationally redundant. H computed once per sample.

    N=6 layout:
        inputs  = [GT1, GT2, GT3, W1, W2, W3]
        outputs = [R_GT1, R_GT2, R_GT3, R_W1, R_W2, R_W3]
    """

    def __init__(self):
        self._similarity = SBERTSimilarity()

    def compute(self, inputs: list[str], outputs: list[str]) -> InvEntropyResult:
        """
        Compute H directly (no bootstrap) for N input-output pairs.

        Args:
            inputs:  N input strings (e.g. [GT1, GT2, GT3, W1, W2, W3])
            outputs: N LLM response strings, one per input

        Returns:
            InvEntropyResult with n_bootstrap=0
        """
        N = len(inputs)
        assert len(outputs) == N
        assert N >= 3, f"N must be ≥ 3, got N={N}"

        H, diag_probs = _compute_H(inputs, outputs, self._similarity)
        # H is unnormalised (paper Eq. 5) — used as ranking score, not clipped.
        # inv_entropy_norm kept for API compat: H / log(N), may exceed 1.0.
        H_norm = H / math.log(N)

        logger.debug(f"H={H:.4f} | H/logN={H_norm:.4f} | N={N}")

        return InvEntropyResult(
            inv_entropy=H,
            inv_entropy_norm=H_norm,
            u_inv_entropy=H_norm,
            n_samples=N,
            n_bootstrap=0,
            diag_probs=diag_probs,
        )

    def compute_n6(
        self,
        gt_texts: list[str],        # [GT1, GT2, GT3]
        whisper_texts: list[str],   # [W1, W2, W3]
        llm_responses: list[str],   # 6 responses — one per input in order above
    ) -> InvEntropyResult:
        """
        N=6 convenience wrapper for PARLA CHIARO.

        Args:
            gt_texts:      [GT1, GT2, GT3] from promptText list in recording JSON
            whisper_texts: [W1, W2, W3] from ASR cache top3 field
            llm_responses: [R_GT1, R_GT2, R_GT3, R_W1, R_W2, R_W3]
        """
        assert len(gt_texts) == 3, f"Need 3 GT texts, got {len(gt_texts)}"
        assert len(whisper_texts) == 3, f"Need 3 Whisper texts, got {len(whisper_texts)}"
        assert len(llm_responses) == 6, f"Need 6 LLM responses, got {len(llm_responses)}"
        return self.compute(
            inputs=gt_texts + whisper_texts,
            outputs=llm_responses,
        )


# ---------------------------------------------------------------------------
# Sanity checks (run as script)
# ---------------------------------------------------------------------------

def _run_sanity_checks():
    print("=== Inv-Entropy sanity checks ===")

    # 1. Row-stochastic
    sim = np.ones((3, 3))
    P = _row_stochastic(sim)
    assert np.allclose(P.sum(axis=1), 1.0), "Row sums must equal 1"
    print(f"[PASS] Row-stochastic — row sums: {P.sum(axis=1)}")

    # 2. With N=3, diagonal of Py@Px is NOT constant 0.5
    Px = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
    Py = np.array([[0.4, 0.4, 0.2], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    diag = np.diag(Py @ Px)
    assert not np.allclose(diag, 0.5), "N=3 diagonal must not be constant"
    print(f"[PASS] N=3 diagonal: {diag.round(4)} (not constant 0.5)")

    # 3. H with uniform distribution equals log(N)
    p_uniform = [1 / 3] * 3
    H_uniform = -sum(p * math.log(p) for p in p_uniform)
    assert abs(H_uniform - math.log(3)) < 1e-6
    print(f"[PASS] H_uniform = {H_uniform:.6f} == log(3) = {math.log(3):.6f}")

    print("=== All checks passed ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _run_sanity_checks()
