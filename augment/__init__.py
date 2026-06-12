"""
augment — Advanced Few-Shot Data Augmentation Package

Pipeline for generating high-quality synthetic hate speech examples with
7 quality-control stages as described in the experimental protocol.

Public API
----------
from augment.generator  import advanced_augment
from augment.cluster    import cluster_hate_speech, stratified_sample, embed_texts
from augment.templates  import build_prompt, style_combinations, CATEGORY_INTENTS
from augment.filter     import filter_generated, deduplicate, clean_line
from augment.similarity import SemanticFilter
"""

from augment.generator  import advanced_augment, compute_target  # noqa: F401
from augment.cluster    import (                          # noqa: F401
    cluster_hate_speech,
    stratified_sample,
    embed_texts,
    SEMANTIC_LABELS,
)
from augment.templates  import (                          # noqa: F401
    build_prompt,
    style_combinations,
    CATEGORY_INTENTS,
)
from augment.filter     import (                          # noqa: F401
    filter_generated,
    deduplicate,
    clean_line,
    is_valid,
)
from augment.similarity import SemanticFilter             # noqa: F401

__all__ = [
    "advanced_augment",
    "cluster_hate_speech",
    "stratified_sample",
    "embed_texts",
    "SEMANTIC_LABELS",
    "build_prompt",
    "style_combinations",
    "CATEGORY_INTENTS",
    "filter_generated",
    "deduplicate",
    "clean_line",
    "is_valid",
    "SemanticFilter",
]
