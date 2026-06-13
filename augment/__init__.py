"""
augment — Advanced Few-Shot Data Augmentation Package

Pipeline for generating high-quality synthetic examples for minority classes
with 7 quality-control stages as described in the experimental protocol.

Supported minority classes:
  • hate_speech (label=0) — clustered with hate-speech-specific anchors
  • neither     (label=2) — clustered with neutral-content anchors

Public API
----------
from augment.generator  import advanced_augment, compute_targets
from augment.cluster    import cluster_hate_speech, cluster_neither, cluster_texts, stratified_sample, embed_texts
from augment.templates  import build_prompt, build_neither_prompt, style_combinations, CATEGORY_INTENTS, NEITHER_CATEGORY_INTENTS
from augment.filter     import filter_generated, deduplicate, clean_line
from augment.similarity import SemanticFilter
"""

from augment.generator  import advanced_augment, compute_targets, compute_target  # noqa: F401
from augment.cluster    import (                          # noqa: F401
    cluster_hate_speech,
    cluster_neither,
    cluster_texts,
    stratified_sample,
    embed_texts,
    SEMANTIC_LABELS,
    NEITHER_SEMANTIC_LABELS,
)
from augment.templates  import (                          # noqa: F401
    build_prompt,
    build_neither_prompt,
    style_combinations,
    CATEGORY_INTENTS,
    NEITHER_CATEGORY_INTENTS,
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
    "compute_targets",
    "compute_target",
    "cluster_hate_speech",
    "cluster_neither",
    "cluster_texts",
    "stratified_sample",
    "embed_texts",
    "SEMANTIC_LABELS",
    "NEITHER_SEMANTIC_LABELS",
    "build_prompt",
    "build_neither_prompt",
    "style_combinations",
    "CATEGORY_INTENTS",
    "NEITHER_CATEGORY_INTENTS",
    "filter_generated",
    "deduplicate",
    "clean_line",
    "is_valid",
    "SemanticFilter",
]
