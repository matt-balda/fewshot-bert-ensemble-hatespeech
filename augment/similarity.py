"""
augment/similarity.py — Etapa 7: Filtro por Similaridade Semântica

Usa sentence-transformers para calcular cosine similarity entre cada
texto gerado e os exemplos reais da mesma categoria semântica.

Critério de aceitação: max_cosine_similarity(gerado, reais) ≥ threshold.
Exemplos abaixo do limiar são descartados por drift semântico.

Threshold padrão: 0.70 (conservador para hate speech, que tem vocabulário
repetitivo). Avaliação de ablação recomendada em [0.60, 0.70, 0.80, 0.90].
"""

from typing import List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_ST = True
except ImportError:
    HAS_ST = False

import logging
logger = logging.getLogger(__name__)

EMBEDDER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.70   # relaxed from 0.90 (spec) to avoid over-filtering hate speech


class SemanticFilter:
    """
    Semantic similarity filter backed by a sentence-transformer model.

    Usage
    -----
    sf = SemanticFilter(threshold=0.70)
    sf.fit(reference_texts)         # encode reference (real) examples
    kept = sf.filter(generated_texts)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = EMBEDDER_MODEL,
    ) -> None:
        if not HAS_ST:
            raise ImportError(
                "sentence-transformers is required: uv add sentence-transformers"
            )
        self.threshold  = threshold
        self.model_name = model_name
        self._embedder  = SentenceTransformer(model_name)
        self._ref_embs: Optional[np.ndarray] = None

    def fit(self, reference_texts: List[str]) -> "SemanticFilter":
        """Encode the reference (real) examples once."""
        self._ref_embs = self._embedder.encode(
            reference_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def score(self, generated_texts: List[str]) -> np.ndarray:
        """
        Return (N,) array of max cosine similarities between each generated
        text and any reference text.
        """
        if self._ref_embs is None:
            raise RuntimeError("Call .fit(reference_texts) before .score().")

        gen_embs = self._encode(generated_texts)
        # (N_gen, N_ref) dot product (both normalized → cosine similarity)
        sim_matrix = gen_embs @ self._ref_embs.T
        return sim_matrix.max(axis=1)   # (N_gen,)

    def filter(
        self,
        generated_texts: List[str],
        return_scores: bool = False,
    ) -> List[str] | Tuple[List[str], np.ndarray]:
        """
        Keep only generated texts with max cosine similarity ≥ threshold.

        Parameters
        ----------
        return_scores : If True, also return the similarity score array.

        Returns
        -------
        List of kept texts (and optionally their scores).
        """
        if not generated_texts:
            if return_scores:
                return [], np.array([])
            return []

        scores = self.score(generated_texts)
        mask   = scores >= self.threshold
        kept   = [t for t, ok in zip(generated_texts, mask) if ok]

        n_total  = len(generated_texts)
        n_kept   = len(kept)
        n_drop   = n_total - n_kept
        logger.info(
            f"  [SemanticFilter] threshold={self.threshold:.2f}  "
            f"kept={n_kept}/{n_total}  dropped={n_drop} "
            f"({n_drop/n_total*100:.1f}%)"
        )

        if return_scores:
            return kept, scores[mask]
        return kept
