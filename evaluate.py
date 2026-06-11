"""
evaluate.py — Evaluation pipeline for the Hate Speech Detection Benchmark.

Loads each saved model checkpoint, runs inference on the held-out test set,
computes all required metrics, generates confusion matrices, and produces a
final comparative results table and ranking.

Usage:
    python evaluate.py                          # evaluate all three models
    python evaluate.py --model bert-base-uncased
    python evaluate.py --model GroNLP/hateBERT
    python evaluate.py --model roberta-base
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from utils import (
    CLASS_NAMES,
    SEED,
    compute_metrics,
    get_device,
    get_logger,
    plot_comparison_bar,
    plot_confusion_matrix,
    print_classification_report,
    set_seed,
)
from data_loader import (
    NUM_LABELS,
    HateSpeechDataset,
    compute_class_weights,
    load_raw_data,
    split_data,
)
from train import MODEL_REGISTRY, HYPERPARAMS

logger = get_logger(__name__, "results/evaluation.log")


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run forward pass on entire DataLoader; return (y_true, y_pred, y_prob).

    Args:
        model:  Fine-tuned classification model.
        loader: DataLoader wrapping the test dataset.
        device: Target compute device.

    Returns:
        Tuple of (y_true, y_pred, y_prob) as NumPy arrays.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="  Inference", leave=False, ncols=90):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

        outputs = model(**kwargs)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_key: str,
    test_df: pd.DataFrame,
    device: torch.device,
    results_dir: str = "results",
    models_dir:  str = "models",
    batch_size:  int = 32,
) -> Dict:
    """Load checkpoint and compute all metrics on the test set.

    Args:
        model_key:   Key in MODEL_REGISTRY.
        test_df:     Test DataFrame.
        device:      Compute device.
        results_dir: Directory for outputs.
        models_dir:  Directory containing model checkpoints.
        batch_size:  Inference batch size.

    Returns:
        Dictionary with all computed metrics.
    """
    cfg        = MODEL_REGISTRY[model_key]
    hf_name    = cfg["hf_name"]
    short_name = cfg["short_name"]
    safe_name  = hf_name.replace("/", "_")
    ckpt_path  = Path(models_dir) / safe_name / "best_model"

    logger.info(f"\n{'━'*60}")
    logger.info(f"  Evaluating: {short_name}")
    logger.info(f"  Checkpoint: {ckpt_path}")
    logger.info(f"{'━'*60}")

    # ── Load tokenizer & model ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), num_labels=NUM_LABELS
    ).to(device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    test_dataset = HateSpeechDataset(test_df, tokenizer, HYPERPARAMS["max_length"])
    test_loader  = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    metrics["model_key"]   = model_key
    metrics["short_name"]  = short_name

    # ── Classification report ─────────────────────────────────────────────────
    report = print_classification_report(y_true, y_pred, CLASS_NAMES)
    report_path = Path(results_dir) / f"{safe_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {short_name}\n")
        f.write("="*55 + "\n")
        f.write(report)
    logger.info(f"\n{report}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=CLASS_NAMES,
        title=short_name,
        save_path=f"{results_dir}/{safe_name}_confusion_matrix.png",
    )

    # ── Save raw predictions ──────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "text":        test_df["text"].values,
        "true_label":  y_true,
        "pred_label":  y_pred,
        **{f"prob_{CLASS_NAMES[i]}": y_prob[:, i] for i in range(len(CLASS_NAMES))},
    })
    pred_df.to_csv(f"{results_dir}/{safe_name}_predictions.csv", index=False)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    saveable = {k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                for k, v in metrics.items()}
    with open(f"{results_dir}/{safe_name}_metrics.json", "w") as f:
        json.dump(saveable, f, indent=2)

    logger.info(
        f"\n  Accuracy:        {metrics['accuracy']:.4f}"
        f"\n  Precision Macro: {metrics['precision_macro']:.4f}"
        f"\n  Recall Macro:    {metrics['recall_macro']:.4f}"
        f"\n  F1 Macro:        {metrics['f1_macro']:.4f}"
        f"\n  F1 Weighted:     {metrics['f1_weighted']:.4f}"
        f"\n  ROC-AUC Macro:   {metrics.get('roc_auc_macro', float('nan')):.4f}"
        f"\n  PR-AUC Macro:    {metrics.get('pr_auc_macro', float('nan')):.4f}"
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Comparative summary
# ──────────────────────────────────────────────────────────────────────────────

COMPARISON_METRICS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
    "roc_auc_macro", "pr_auc_macro",
]


def build_comparison_table(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Build and save a comparative results table.

    Args:
        all_metrics: {model_key: metrics_dict}.

    Returns:
        DataFrame with one row per model.
    """
    rows = []
    for mk, m in all_metrics.items():
        row = {"Model": m["short_name"]}
        for metric in COMPARISON_METRICS:
            val = m.get(metric, float("nan"))
            row[metric.replace("_", " ").title()] = (
                f"{val:.4f}" if isinstance(val, float) and not np.isnan(val) else "N/A"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def rank_models(all_metrics: Dict[str, Dict]) -> List[str]:
    """Rank models by F1 Macro (primary) then F1 Weighted (secondary).

    Args:
        all_metrics: {model_key: metrics_dict}.

    Returns:
        Ordered list of model keys, best first.
    """
    return sorted(
        all_metrics.keys(),
        key=lambda k: (
            all_metrics[k].get("f1_macro", 0.0),
            all_metrics[k].get("f1_weighted", 0.0),
        ),
        reverse=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hate Speech Benchmark — Evaluation")
    p.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + list(MODEL_REGISTRY.keys()),
        help="Model to evaluate. 'all' evaluates all three models.",
    )
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--results_dir",type=str, default="results")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed",       type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # ── Load test set ─────────────────────────────────────────────────────────
    test_csv = Path(args.data_dir) / "test.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        logger.info(f"Loaded test set from {test_csv} ({len(test_df):,} samples)")
    else:
        # Fallback: re-create split deterministically from raw data
        logger.warning("test.csv not found; re-splitting raw data with same seed.")
        df = load_raw_data(args.data_dir)
        _, _, test_df = split_data(df, seed=args.seed)

    # ── Select models ─────────────────────────────────────────────────────────
    if args.model == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = [args.model]

    # ── Evaluate each model ───────────────────────────────────────────────────
    all_metrics: Dict[str, Dict] = {}
    for mk in model_keys:
        m = evaluate_model(
            mk, test_df, device,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
            batch_size=args.batch_size,
        )
        all_metrics[mk] = m

    if len(all_metrics) < 2:
        logger.info("Single-model evaluation complete.")
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    table = build_comparison_table(all_metrics)
    table_path = Path(args.results_dir) / "comparison_table.csv"
    table.to_csv(table_path, index=False)

    logger.info("\n" + "="*80)
    logger.info("COMPARATIVE RESULTS TABLE")
    logger.info("="*80)
    logger.info("\n" + table.to_string(index=False))

    # ── Ranking ───────────────────────────────────────────────────────────────
    ranked = rank_models(all_metrics)
    logger.info("\n" + "="*80)
    logger.info("FINAL RANKING (by F1 Macro)")
    logger.info("="*80)
    for rank, mk in enumerate(ranked, 1):
        m = all_metrics[mk]
        logger.info(
            f"  #{rank}  {m['short_name']:<20}  "
            f"F1 Macro={m['f1_macro']:.4f}  "
            f"Accuracy={m['accuracy']:.4f}"
        )

    # ── Comparison bar charts ─────────────────────────────────────────────────
    for metric in ["f1_macro", "f1_weighted", "accuracy", "roc_auc_macro"]:
        plot_comparison_bar(
            {mk: all_metrics[mk] for mk in all_metrics},
            metric=metric,
            save_path=f"{args.results_dir}/compare_{metric}.png",
        )

    # ── Save full results ─────────────────────────────────────────────────────
    with open(f"{args.results_dir}/all_metrics.json", "w") as f:
        json.dump(
            {mk: {k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                  for k, v in m.items()}
             for mk, m in all_metrics.items()},
            f, indent=2,
        )

    logger.info(f"\nAll results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
