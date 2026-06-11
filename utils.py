"""
utils.py — Utility functions for the Hate Speech Detection Benchmark.

Provides reproducibility helpers, metric computation, visualization routines,
and logging utilities shared across all pipeline stages.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Return a configured logger that writes to stdout (and optionally a file)."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for full reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch (CPU + GPU), and
    disables CUDA non-deterministic algorithms where possible.

    Args:
        seed: Integer seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enforce deterministic CUDA ops (may reduce throughput slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Device helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["hate_speech", "offensive_language", "neither"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute the full evaluation suite for multi-class classification.

    Args:
        y_true:      Ground-truth integer labels, shape (N,).
        y_pred:      Predicted integer labels, shape (N,).
        y_prob:      Softmax probabilities, shape (N, C).  Required for AUC scores.
        class_names: Human-readable class names for the report.

    Returns:
        Dictionary mapping metric names to scalar float values.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    metrics: Dict[str, float] = {}

    # Core classification metrics
    metrics["accuracy"]          = accuracy_score(y_true, y_pred)
    metrics["precision_macro"]   = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    metrics["recall_macro"]      = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
    metrics["f1_macro"]          = f1_score(       y_true, y_pred, average="macro",    zero_division=0)
    metrics["f1_weighted"]       = f1_score(       y_true, y_pred, average="weighted", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for cls_name, f1_val in zip(class_names, per_class_f1):
        metrics[f"f1_{cls_name}"] = float(f1_val)

    # AUC scores (require probability estimates)
    if y_prob is not None:
        n_classes = y_prob.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        try:
            metrics["roc_auc_macro"] = roc_auc_score(
                y_bin, y_prob, multi_class="ovr", average="macro"
            )
            metrics["roc_auc_weighted"] = roc_auc_score(
                y_bin, y_prob, multi_class="ovr", average="weighted"
            )
        except ValueError:
            # Can happen if a class is absent from y_true
            metrics["roc_auc_macro"]    = float("nan")
            metrics["roc_auc_weighted"] = float("nan")

        # PR-AUC (macro average across classes)
        pr_aucs = []
        for cls_idx in range(n_classes):
            try:
                pr_aucs.append(
                    average_precision_score(y_bin[:, cls_idx], y_prob[:, cls_idx])
                )
            except ValueError:
                pr_aucs.append(float("nan"))
        metrics["pr_auc_macro"] = float(np.nanmean(pr_aucs))

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Return sklearn's classification report as a formatted string."""
    if class_names is None:
        class_names = CLASS_NAMES
    return classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str,
) -> None:
    """Plot and save a normalized confusion matrix heatmap.

    Normalization over rows (true labels) enables per-class recall visualization
    independent of class frequency — critical for imbalanced datasets.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: List of human-readable class names.
        title:       Figure title.
        save_path:   Absolute path where the PNG will be written.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Normalized (row)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_ylabel("True label", fontsize=11)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_title(f"{title} — {subtitle}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_f1s: List[float],
    model_name: str,
    save_path: str,
) -> None:
    """Plot loss and F1 curves over training epochs.

    Args:
        train_losses: Per-epoch training loss.
        val_losses:   Per-epoch validation loss.
        val_f1s:      Per-epoch macro F1 on validation set.
        model_name:   Model identifier for the plot title.
        save_path:    Absolute path where the PNG will be written.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, val_losses,   "s--", label="Val Loss",   color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{model_name} — Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_f1s, "D-", label="Val F1 Macro", color="#2ca02c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Macro")
    ax2.set_title(f"{model_name} — Validation F1 Macro")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    label_counts: Dict[str, int],
    title: str,
    save_path: str,
) -> None:
    """Bar chart of class distribution.

    Args:
        label_counts: Mapping of class name → sample count.
        title:        Figure title.
        save_path:    Output path for the PNG.
    """
    classes = list(label_counts.keys())
    counts  = list(label_counts.values())
    total   = sum(counts)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, counts, color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="white")

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{cnt:,}\n({cnt/total*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Class")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    metric: str,
    save_path: str,
) -> None:
    """Grouped bar chart comparing a single metric across models.

    Args:
        results:   {model_name: {metric_name: value}}.
        metric:    Metric key to plot.
        save_path: Output PNG path.
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0.0) for m in models]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=colors[:len(models)], edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
