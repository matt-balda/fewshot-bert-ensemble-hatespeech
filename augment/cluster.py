"""
augment/cluster.py — Etapas 1 & 3: Estratificação Semântica + Amostragem Estratificada

Usa sentence-transformers para embutir os exemplos de hate speech e
K-Means para agrupá-los em categorias semânticas (racial attacks,
xenofobia, homofobia, sexismo, intolerância religiosa, outros).

A escolha de K é validada via silhouette score quando K não é fixado.

Fix (Bug 1): os IDs de cluster do K-Means são arbitrários — assign_semantic_labels
rotula cada centroide por similaridade cosine com âncoras de categoria,
garantindo que cluster_name corresponda à categoria semântica real.
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


# Canonical category names used in templates.py
SEMANTIC_LABELS = [
    "racial_attacks",
    "xenophobia",
    "homophobia",
    "sexism",
    "religious_intolerance",
    "other",
]

DEFAULT_K = len(SEMANTIC_LABELS)

# Representative anchor phrases used to semantically label each centroid
_CATEGORY_ANCHORS: Dict[str, str] = {
    "racial_attacks":        "racism racial slurs ethnic inferiority white supremacy hate",
    "xenophobia":            "anti-immigrant foreigners go back to your country nationalism",
    "homophobia":            "anti-gay lesbian LGBTQ queer homophobia faggot",
    "sexism":                "misogyny women hate gender sexist bitch female attacks",
    "religious_intolerance": "anti-religion Islam Muslim Christian Jewish religious hate",
    "other":                 "general hate dehumanizing slurs offensive attack disgusting",
}
EMBEDDER_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, multilingual-capable

# Anchor phrases for the 'neither' (neutral/non-offensive) class
NEITHER_SEMANTIC_LABELS = [
    "sarcasm_humor",
    "political_neutral",
    "everyday_talk",
    "news_media",
    "sports_entertainment",
    "other_neutral",
]

_NEITHER_CATEGORY_ANCHORS: Dict[str, str] = {
    "sarcasm_humor":        "sarcasm irony jokes humor wit funny comedy satire",
    "political_neutral":    "politics government policy opinion debate election democracy",
    "everyday_talk":        "daily life feelings observations casual conversation personal mood",
    "news_media":           "news current events journalism media reporting facts",
    "sports_entertainment": "sports football basketball music movies celebrities entertainment",
    "other_neutral":        "general neutral normal everyday non-offensive content remarks",
}


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


def assign_semantic_labels(
    kmeans: KMeans,
    embedder: "SentenceTransformer",
    category_anchors: Optional[Dict[str, str]] = None,
) -> Dict[int, str]:
    """
    Assign human-readable semantic labels to K-Means cluster IDs.

    For each centroid, finds the most similar semantic category using
    cosine similarity against the provided anchor phrases.
    Uses a greedy one-to-one assignment so each category is used at
    most once (when n_clusters <= n_categories).

    Parameters
    ----------
    kmeans          : Fitted KMeans object.
    embedder        : Already-loaded SentenceTransformer instance (avoids reloading).
    category_anchors: Dict mapping category name → anchor phrase string.
                      Defaults to _CATEGORY_ANCHORS (hate speech) if None.

    Returns
    -------
    Dict mapping cluster_id (int) → category name (str).
    """
    anchors  = category_anchors if category_anchors is not None else _CATEGORY_ANCHORS
    cat_names = list(anchors.keys())
    cat_texts = list(anchors.values())

    cat_embs = embedder.encode(
        cat_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )  # (n_cat, D)

    # Normalize centroids for cosine similarity
    centroids = kmeans.cluster_centers_.copy()  # (n_clusters, D)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / np.maximum(norms, 1e-8)

    # Similarity matrix (n_clusters, n_cat)
    sim = centroids_norm @ cat_embs.T

    n_clusters   = centroids.shape[0]
    assigned: Dict[int, str] = {}
    used_cats: set = set()

    # Greedy one-to-one assignment by descending similarity score
    pairs = [
        (float(sim[ci, ki]), ci, ki)
        for ci in range(n_clusters)
        for ki in range(len(cat_names))
    ]
    pairs.sort(reverse=True)

    for _, cluster_id, cat_idx in pairs:
        if cluster_id in assigned:
            continue
        cat = cat_names[cat_idx]
        # Allow reuse only when we have more clusters than categories
        if cat in used_cats and n_clusters <= len(cat_names):
            continue
        assigned[cluster_id] = cat
        used_cats.add(cat)

    # Fallback: handle extra clusters when n_clusters > n_categories
    for ci in range(n_clusters):
        if ci not in assigned:
            assigned[ci] = f"cluster_{ci}"

    return assigned


def cluster_hate_speech(
    hate_df: pd.DataFrame,
    k: Optional[int] = None,
    auto_k: bool = False,
    seed: int = 42,
    embedder_model: str = EMBEDDER_MODEL,
) -> Tuple[pd.DataFrame, np.ndarray, "SentenceTransformer", KMeans]:
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
    embedder  : Loaded SentenceTransformer instance (reused downstream).
    kmeans    : Fitted KMeans object.
    """
    if not HAS_ST:
        raise ImportError(
            "sentence-transformers is required: uv add sentence-transformers"
        )

    texts    = hate_df["text"].tolist()
    embedder = SentenceTransformer(embedder_model)

    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if k is None:
        k = choose_k(embeddings, seed=seed) if auto_k else DEFAULT_K

    kmeans      = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    # Bug-1 fix: assign labels semantically, not by arbitrary cluster ID position
    semantic_labels = assign_semantic_labels(kmeans, embedder)

    hate_df = hate_df.copy()
    hate_df["cluster"]      = cluster_ids
    hate_df["cluster_name"] = [semantic_labels.get(c, f"cluster_{c}") for c in cluster_ids]
    return hate_df, embeddings, embedder, kmeans


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


def cluster_texts(
    df: pd.DataFrame,
    category_anchors: Dict[str, str],
    k: Optional[int] = None,
    auto_k: bool = False,
    seed: int = 42,
    embedder_model: str = EMBEDDER_MODEL,
) -> Tuple[pd.DataFrame, np.ndarray, "SentenceTransformer", KMeans]:
    """
    Generic text clustering with configurable semantic anchors.

    Identical to cluster_hate_speech but accepts any anchor dict, making it
    usable for any class (neither, offensive_language, etc.).

    Parameters
    ----------
    df               : DataFrame with at least a 'text' column.
    category_anchors : Dict {category_name: anchor_phrase} used for labelling.
    k                : Number of clusters. Auto-selected via silhouette if None and auto_k=True.

    Returns
    -------
    df        : Copy with 'cluster' and 'cluster_name' columns added.
    embeddings: (N, D) numpy array.
    embedder  : Loaded SentenceTransformer instance.
    kmeans    : Fitted KMeans object.
    """
    if not HAS_ST:
        raise ImportError(
            "sentence-transformers is required: uv add sentence-transformers"
        )

    texts    = df["text"].tolist()
    embedder = SentenceTransformer(embedder_model)

    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    default_k = len(category_anchors)
    if k is None:
        k = choose_k(embeddings, seed=seed) if auto_k else default_k

    kmeans      = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    semantic_labels = assign_semantic_labels(kmeans, embedder, category_anchors)

    df = df.copy()
    df["cluster"]      = cluster_ids
    df["cluster_name"] = [semantic_labels.get(c, f"cluster_{c}") for c in cluster_ids]
    return df, embeddings, embedder, kmeans


def cluster_neither(
    neither_df: pd.DataFrame,
    k: Optional[int] = None,
    auto_k: bool = False,
    seed: int = 42,
    embedder_model: str = EMBEDDER_MODEL,
) -> Tuple[pd.DataFrame, np.ndarray, "SentenceTransformer", KMeans]:
    """
    Cluster 'neither' (neutral/non-offensive) examples into semantic groups.
    Uses _NEITHER_CATEGORY_ANCHORS for labelling.
    """
    return cluster_texts(
        neither_df,
        category_anchors=_NEITHER_CATEGORY_ANCHORS,
        k=k if k is not None else len(NEITHER_SEMANTIC_LABELS),
        auto_k=auto_k,
        seed=seed,
        embedder_model=embedder_model,
    )

