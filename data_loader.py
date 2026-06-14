"""
data_loader.py — Dataset loading, preprocessing, and tokenization.

Dataset: Hate Speech and Offensive Language Dataset (Davidson et al., 2017)
Source:  https://huggingface.co/datasets/hate_speech_offensive

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

# Constants
LABEL_MAP: Dict[int, str] = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}
NUM_LABELS = len(LABEL_MAP)
MAX_LENGTH = 128   # 128 tokens covers >97 % of tweets without excessive padding


# Text preprocessing
_URL_RE    = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")  # keep the word, drop the hash
_MULTI_SPACE = re.compile(r"\s+")

def preprocess_text(text: str) -> str:
    
    text = _URL_RE.sub("[URL]", text)
    text = _MENTION_RE.sub("[USER]", text)
    text = _HASHTAG_RE.sub(r"\1", text)        # unwrap hashtags
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# HuggingFace Dataset loader

def load_raw_data(data_dir: str = "data") -> pd.DataFrame:
    
    cache_path = Path(data_dir) / "davidson_raw.csv"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        return df

    logger.info("Downloading tdavidson/hate_speech_offensive from HuggingFace Hub")

    dataset = load_dataset("tdavidson/hate_speech_offensive")

    # Only train in huggingface
    records = []
    for split_name, split_data in dataset.items():
        for item in split_data:
            
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

# Split strategy

def split_data(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    # test_frac is implicitly 1 − train − val = 0.15
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
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


# Dataset statistics & EDA

def describe_dataset(df: pd.DataFrame, split_name: str = "full") -> None:
    
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


# PyTorch Dataset

class HateSpeechDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.labels = dataframe["label"].values.astype(np.int64)
        
        # Check if is_synthetic column is present, otherwise default to all False (0.0)
        if "is_synthetic" in dataframe.columns:
            self.is_synthetic = dataframe["is_synthetic"].values.astype(np.float32)
        else:
            self.is_synthetic = np.zeros(len(dataframe), dtype=np.float32)

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
        item["is_synthetic"] = torch.tensor(self.is_synthetic[idx], dtype=torch.float32)
        return item

# Class-weight computation (for weighted cross-entropy)

def compute_class_weights(
    train_df: pd.DataFrame, num_labels: int = NUM_LABELS
) -> torch.Tensor:
    
    counts = np.bincount(train_df["label"].values, minlength=num_labels).astype(float)
    total  = counts.sum()
    weights = total / (num_labels * counts)
    weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights: {weights.tolist()}")
    return weights
