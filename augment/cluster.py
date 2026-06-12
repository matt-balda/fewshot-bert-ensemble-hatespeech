"""
augment/cluster.py — Etapas 1 & 3: Estratificação Semântica + Amostragem Estratificada

Usa sentence-transformers para embutir os exemplos de hate speech e
K-Means para agrupá-los em categorias semânticas (racial attacks,
xenofobia, homofobia, sexismo, intolerância religiosa, outros).

A escolha de K é validada via silhouette score quando K não é fixado.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_ST = True
except ImportError:
    HAS_ST = False


SEMANTIC_LABELS = {
    0: "racial_attacks",
    1: "xenophobia",
    2: "homophobia",
    3: "sexism",
    4: "religious_intolerance",
    5: "other",
}

DEFAULT_K = len(SEMANTIC_LABELS)
EMBEDDER_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, multilingual-capable


def embed_texts(
    texts: List[str],
    model_name: str = EMBEDDER_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Return (N, D) sentence embeddings for the input texts."""
    if not HAS_ST:
        raise ImportError(
            "sentence-transformers is required: uv add sentence-transformers"
        )
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    return embeddings


def choose_k(
    embeddings: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    seed: int = 42,
) -> int:
    """Select optimal K via silhouette score."""
    best_k, best_score = k_range[0], -1.0
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_hate_speech(
    hate_df: pd.DataFrame,
    k: Optional[int] = None,
    auto_k: bool = False,
    seed: int = 42,
    embedder_model: str = EMBEDDER_MODEL,
) -> Tuple[pd.DataFrame, np.ndarray, KMeans]:
    """
    Cluster hate speech texts into semantic groups.

    Parameters
    ----------
    hate_df   : DataFrame with at least a 'text' column (hate speech only).
    k         : Number of clusters. If None and auto_k=True, auto-selected.
    auto_k    : If True, choose K via silhouette score.

    Returns
    -------
    hate_df   : Copy of input DataFrame with 'cluster' and 'cluster_name' columns.
    embeddings: (N, D) numpy array of sentence embeddings.
    kmeans    : Fitted KMeans object.
    """
    texts = hate_df["text"].tolist()
    embeddings = embed_texts(texts, model_name=embedder_model)

    if k is None:
        k = choose_k(embeddings) if auto_k else DEFAULT_K

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    hate_df = hate_df.copy()
    hate_df["cluster"]      = cluster_ids
    hate_df["cluster_name"] = [SEMANTIC_LABELS.get(c, f"cluster_{c}") for c in cluster_ids]
    return hate_df, embeddings, kmeans


def stratified_sample(
    clustered_df: pd.DataFrame,
    target_per_cluster: int,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Return a dict {cluster_name: sampled_dataframe} with ≤ target_per_cluster
    examples per cluster used as few-shot seeds.

    Sampling is proportional to cluster size but guarantees at least
    min(5, cluster_size) examples per cluster.
    """
    rng = np.random.default_rng(seed)
    cluster_samples: Dict[str, pd.DataFrame] = {}

    for name, group in clustered_df.groupby("cluster_name"):
        n = min(target_per_cluster, len(group))
        n = max(n, min(5, len(group)))
        idx = rng.choice(len(group), size=n, replace=False)
        cluster_samples[str(name)] = group.iloc[idx].reset_index(drop=True)

    return cluster_samples
