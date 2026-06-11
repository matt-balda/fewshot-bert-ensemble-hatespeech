"""
data_loader.py — Dataset loading, preprocessing, and tokenization.

Dataset: Hate Speech and Offensive Language Dataset (Davidson et al., 2017)
Source:  https://huggingface.co/datasets/hate_speech_offensive
         (mirrors the original GitHub release by T. Davidson et al.)

Justification for dataset choice
──────────────────────────────────
Davidson et al. (2017) is the most widely cited dataset for hate speech
detection in English, with 24,783 tweet samples manually annotated into three
classes: hate speech, offensive language, and neither. It is:
  • Fully public (CC-BY-4.0 license via HuggingFace hub).
  • Extensively benchmarked: referenced in >500 peer-reviewed papers.
  • Available with pre-defined train/val/test splits via sklearn or manual split.
  • Non-trivially imbalanced (offensive >> hate speech >> neither), making it a
    realistic benchmark that exercises macro F1 alongside accuracy.

OLID was considered but uses a hierarchical label scheme (task A/B/C) that
complicates direct multi-class comparison. HateXplain includes rationale spans
which require additional modeling effort beyond scope. Davidson's flat 3-class
structure is the cleanest for a transformer benchmark study.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerBase

from utils import get_logger, set_seed, SEED, plot_class_distribution

logger = get_logger(__name__, "results/data_loading.log")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LABEL_MAP: Dict[int, str] = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}
NUM_LABELS = len(LABEL_MAP)
MAX_LENGTH = 128   # 128 tokens covers >97 % of tweets without excessive padding


# ──────────────────────────────────────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_URL_RE    = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")  # keep the word, drop the hash
_MULTI_SPACE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """Lightweight normalization for Twitter text.

    Steps:
      1. Replace URLs with the token [URL] — preserves positional cues without
         leaking domain-specific information.
      2. Replace @mentions with [USER] — anonymizes entities.
      3. Expand hashtags (#BlackLivesMatter → BlackLivesMatter) — retains
         semantic content that would otherwise be discarded by sub-word models.
      4. Strip leading/trailing whitespace and collapse runs of spaces.

    Note: We deliberately do NOT lowercase, as BERT/RoBERTa use cased
    tokenizers that extract information from capitalization patterns common
    in hate speech (e.g., "THIS IS OFFENSIVE").

    Args:
        text: Raw tweet string.

    Returns:
        Cleaned tweet string.
    """
    text = _URL_RE.sub("[URL]", text)
    text = _MENTION_RE.sub("[USER]", text)
    text = _HASHTAG_RE.sub(r"\1", text)        # unwrap hashtags
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str = "data") -> pd.DataFrame:
    """Download or load from cache the Davidson et al. dataset.

    The dataset is fetched from the HuggingFace Hub
    (``hate_speech_offensive``) and cached locally under ``data_dir``.
    If a cached CSV already exists, it is loaded directly to avoid repeated
    network calls — important for reproducible offline experiments.

    Args:
        data_dir: Local directory for caching the raw CSV.

    Returns:
        DataFrame with columns: ``text``, ``label``, ``label_name``.
    """
    cache_path = Path(data_dir) / "davidson_raw.csv"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        return df

    logger.info("Downloading tdavidson/hate_speech_offensive from HuggingFace Hub …")
    # Dataset was migrated to namespace; 'trust_remote_code' removed in datasets>=5
    dataset = load_dataset("tdavidson/hate_speech_offensive")

    # The HF dataset only provides a 'train' split; we perform our own split
    records = []
    for split_name, split_data in dataset.items():
        for item in split_data:
            # Column names may vary by dataset version; handle both variants
            tweet = item.get("tweet", item.get("text", ""))
            label = item.get("class", item.get("label", 0))
            records.append({
                "tweet": tweet,
                "label": label,
            })

    df = pd.DataFrame(records)
    df["text"]       = df["tweet"].apply(preprocess_text)
    df["label_name"] = df["label"].map(LABEL_MAP)
    df = df[["text", "label", "label_name"]].reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    logger.info(f"Cached {len(df):,} samples to {cache_path}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Split strategy
# ──────────────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    # test_frac is implicitly 1 − train − val = 0.15
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split.

    Stratification on the label column ensures that each partition reflects
    the original class distribution — crucial for avoiding evaluation bias on
    imbalanced datasets.

    Args:
        df:          Full DataFrame.
        train_frac:  Fraction of data for training.
        val_frac:    Fraction of data for validation.
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert train_frac + val_frac < 1.0, "train + val fracs must be < 1.0"
    test_frac = 1.0 - train_frac - val_frac

    train_df, tmp_df = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=test_frac / (val_frac + test_frac), stratify=tmp_df["label"], random_state=seed
    )
    logger.info(
        f"Split sizes — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset statistics & EDA
# ──────────────────────────────────────────────────────────────────────────────

def describe_dataset(df: pd.DataFrame, split_name: str = "full") -> None:
    """Log and plot class distribution statistics.

    Args:
        df:         DataFrame with ``label`` and ``label_name`` columns.
        split_name: Identifier for logging (e.g., 'train', 'val', 'test').
    """
    counts = df["label_name"].value_counts().sort_index()
    total  = len(df)

    logger.info(f"\n{'='*55}")
    logger.info(f"  Dataset split: {split_name.upper()}  (N={total:,})")
    logger.info(f"{'='*55}")
    for cls, cnt in counts.items():
        logger.info(f"  {cls:<25} {cnt:>6,}  ({cnt/total*100:5.1f}%)")

    imbalance_ratio = counts.max() / counts.min()
    logger.info(f"  Imbalance ratio (max/min): {imbalance_ratio:.1f}x")
    logger.info(f"{'='*55}\n")

    plot_class_distribution(
        {row: cnt for row, cnt in counts.items()},
        title=f"Class Distribution — {split_name}",
        save_path=f"results/class_dist_{split_name}.png",
    )


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HateSpeechDataset(Dataset):
    """PyTorch Dataset wrapping the Davidson hate speech data.

    Tokenization is performed at construction time so that GPU training is not
    bottlenecked by on-the-fly tokenization in the DataLoader workers.

    Args:
        dataframe:   DataFrame with ``text`` and ``label`` columns.
        tokenizer:   HuggingFace PreTrainedTokenizer instance.
        max_length:  Maximum token sequence length.  Sequences are right-padded
                     and truncated.  128 tokens is sufficient for tweets and
                     keeps the per-sample memory footprint small.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.labels = dataframe["label"].values.astype(np.int64)
        self.encodings = tokenizer(
            dataframe["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ──────────────────────────────────────────────────────────────────────────────
# Class-weight computation (for weighted cross-entropy)
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    train_df: pd.DataFrame, num_labels: int = NUM_LABELS
) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted cross-entropy loss.

    Inverse-frequency weighting penalizes misclassification of minority classes
    proportionally to their underrepresentation, mitigating the bias toward
    dominant classes in imbalanced datasets.

    Formula: weight_c = N / (K * n_c)
      where N = total samples, K = number of classes, n_c = samples in class c.

    Args:
        train_df:   Training DataFrame with a ``label`` column.
        num_labels: Number of distinct classes.

    Returns:
        Tensor of shape (num_labels,) with per-class weights.
    """
    counts = np.bincount(train_df["label"].values, minlength=num_labels).astype(float)
    total  = counts.sum()
    weights = total / (num_labels * counts)
    weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights: {weights.tolist()}")
    return weights
