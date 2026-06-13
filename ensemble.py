"""
ensemble.py — Cenário B: Ensemble de Hard Voting e Soft Voting

Carrega as predições (y_pred, y_prob) dos três modelos individuais já
treinados (BERT, RoBERTa, HateBERT) e combina-as via:

  B1 — Hard Voting : classe mais votada pelos três modelos (scipy.stats.mode).
  B2 — Soft Voting : argmax da média das probabilidades softmax.

Saídas (em results_dir/):
  ensemble_hard_metrics.json / ensemble_soft_metrics.json
  ensemble_hard_predictions.csv / ensemble_soft_predictions.csv
  ensemble_hard_confusion_matrix.png / ensemble_soft_confusion_matrix.png
  ensemble_hard_classification_report.txt / ...
  ensemble_compare_<metric>.png

Usage:
    python ensemble.py                      # avalia hard + soft voting
    python ensemble.py --strategy hard
    python ensemble.py --strategy soft
    python ensemble.py --models_dir models/scenario_A --results_dir results/scenario_B
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader import HateSpeechDataset, NUM_LABELS, load_raw_data, split_data
from train import HYPERPARAMS, MODEL_REGISTRY
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

logger = get_logger(__name__, "results/ensemble.log")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, y_prob) for a loaded checkpoint."""
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


def load_model_predictions(
    model_key: str,
    test_df: pd.DataFrame,
    device: torch.device,
    models_dir: str = "models",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved checkpoint and return (y_true, y_pred, y_prob) on test_df."""
    cfg       = MODEL_REGISTRY[model_key]
    hf_name   = cfg["hf_name"]
    safe_name = hf_name.replace("/", "_")
    ckpt_path = Path(models_dir) / safe_name / "best_model"

    logger.info(f"  Loading checkpoint: {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), num_labels=NUM_LABELS
    ).to(device)

    dataset = HateSpeechDataset(test_df, tokenizer, HYPERPARAMS["max_length"])
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    y_true, y_pred, y_prob = _run_inference(model, loader, device)

    # Release GPU memory before loading the next model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Voting strategies
# ---------------------------------------------------------------------------

def hard_voting(all_preds: Dict[str, np.ndarray]) -> np.ndarray:
    """
    B1 — Hard Voting.
    Stack per-model predictions (shape: M × N) and take the mode along axis 0.
    In case of a tie, scipy returns the smallest class index.
    """
    stacked = np.stack(list(all_preds.values()), axis=0)   # (M, N)
    voted, _ = stats.mode(stacked, axis=0, keepdims=False)
    return voted.flatten().astype(int)


def soft_voting(
    all_probs: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    B2 — Soft Voting.
    Average softmax probability matrices (M × N × C) and take argmax.
    Returns (y_pred_ensemble, avg_probs).
    """
    stacked   = np.stack(list(all_probs.values()), axis=0)  # (M, N, C)
    avg_probs = stacked.mean(axis=0)                         # (N, C)
    y_pred    = np.argmax(avg_probs, axis=1)
    return y_pred, avg_probs


# ---------------------------------------------------------------------------
# Evaluation + artefact saving
# ---------------------------------------------------------------------------

def evaluate_and_save(
    strategy_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    results_dir: str,
    test_df: pd.DataFrame,
) -> Dict:
    """Compute all metrics, save plots and JSON for one ensemble strategy."""
    metrics = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    metrics["strategy"] = strategy_name

    # ── Classification report ───────────────────────────────────────────────
    report = print_classification_report(y_true, y_pred, CLASS_NAMES)
    rpt_path = Path(results_dir) / f"ensemble_{strategy_name}_classification_report.txt"
    rpt_path.write_text(
        f"Ensemble — {strategy_name.upper()} VOTING\n{'='*55}\n{report}"
    )
    logger.info(f"\n{report}")

    # ── Confusion matrix ────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=CLASS_NAMES,
        title=f"Ensemble {strategy_name.title()} Voting",
        save_path=str(Path(results_dir) / f"ensemble_{strategy_name}_confusion_matrix.png"),
    )

    # ── Raw predictions CSV ─────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "text":       test_df["text"].values,
        "true_label": y_true,
        "pred_label": y_pred,
        **{f"prob_{CLASS_NAMES[i]}": y_prob[:, i] for i in range(len(CLASS_NAMES))},
    })
    pred_df.to_csv(
        Path(results_dir) / f"ensemble_{strategy_name}_predictions.csv", index=False
    )

    # ── Metrics JSON ────────────────────────────────────────────────────────
    saveable = {
        k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
        for k, v in metrics.items()
    }
    (Path(results_dir) / f"ensemble_{strategy_name}_metrics.json").write_text(
        json.dumps(saveable, indent=2)
    )

    # ── Structured log ──────────────────────────────────────────────────────
    logger.info(
        f"\n  {'─'*50}\n"
        f"  Ensemble {strategy_name.upper()} VOTING\n"
        f"  {'─'*50}\n"
        f"  Accuracy           : {metrics['accuracy']:.4f}\n"
        f"\n  ── Macro ───────────────────────────────────\n"
        f"  Precision Macro    : {metrics['precision_macro']:.4f}\n"
        f"  Recall    Macro    : {metrics['recall_macro']:.4f}\n"
        f"  F1        Macro    : {metrics['f1_macro']:.4f}\n"
        f"\n  ── Weighted ────────────────────────────────\n"
        f"  Precision Weighted : {metrics['precision_weighted']:.4f}\n"
        f"  Recall    Weighted : {metrics['recall_weighted']:.4f}\n"
        f"  F1        Weighted : {metrics['f1_weighted']:.4f}\n"
        f"\n  ── AUC ─────────────────────────────────────\n"
        f"  ROC-AUC Macro      : {metrics.get('roc_auc_macro', float('nan')):.4f}\n"
        f"  PR-AUC  Macro      : {metrics.get('pr_auc_macro',  float('nan')):.4f}\n"
        f"\n  ── Hate Speech class (minority) ─────────────\n"
        f"  Precision (HS)     : {metrics.get('precision_hate_speech', float('nan')):.4f}\n"
        f"  Recall    (HS)     : {metrics.get('recall_hate_speech',    float('nan')):.4f}\n"
        f"  F1        (HS)     : {metrics.get('f1_hate_speech',        float('nan')):.4f}\n"
    )

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hate Speech Benchmark: Ensemble Evaluation")
    p.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["hard", "soft", "both"],
        help="Ensemble strategy to evaluate.",
    )
    p.add_argument("--data_dir",    type=str, default="data")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--models_dir",  type=str, default="models")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--seed",        type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    logger.info("  ENSEMBLE EVALUATION")
    logger.info("=" * 60)

    # ── Load test set ───────────────────────────────────────────────────────
    test_csv = Path(args.data_dir) / "test.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        logger.info(f"Loaded test set from {test_csv} ({len(test_df):,} samples)")
    else:
        logger.warning("test.csv not found; re-splitting raw data with same seed.")
        df = load_raw_data(args.data_dir)
        _, _, test_df = split_data(df, seed=args.seed)

    # ── Collect per-model predictions ───────────────────────────────────────
    model_keys = list(MODEL_REGISTRY.keys())
    all_preds: Dict[str, np.ndarray] = {}
    all_probs: Dict[str, np.ndarray] = {}
    y_true_ref: np.ndarray | None = None

    for mk in model_keys:
        ckpt = Path(args.models_dir) / mk.replace("/", "_") / "best_model"
        if not ckpt.exists():
            logger.error(
                f"Checkpoint not found: {ckpt}. "
                f"Run train.py first. Skipping {mk}."
            )
            continue

        logger.info(f"\nLoading: {MODEL_REGISTRY[mk]['short_name']}")
        y_true, y_pred, y_prob = load_model_predictions(
            mk, test_df, device, args.models_dir, args.batch_size
        )
        all_preds[mk] = y_pred
        all_probs[mk] = y_prob
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.array_equal(y_true, y_true_ref), (
                f"Ground-truth labels mismatch for model '{mk}'. "
                "Ensure all models use the same test set and DataLoader (shuffle=False)."
            )

    if len(all_preds) < 2:
        logger.error("Need ≥ 2 model checkpoints to run ensemble. Exiting.")
        return
    if y_true_ref is None:
        logger.error("Could not obtain ground-truth labels. Exiting.")
        return

    all_ensemble_metrics: Dict[str, Dict] = {}

    # ── B1 — Hard Voting ────────────────────────────────────────────────────
    if args.strategy in ("hard", "both"):
        logger.info("\n" + "-" * 60)
        logger.info("  B1 — Hard Voting")
        logger.info("-" * 60)
        y_hard = hard_voting(all_preds)
        # Bug-2 fix: use one-hot probabilities for Hard Voting instead of
        # averaged softmax (which is identical to Soft Voting and incorrect).
        # One-hot encodes the hard decision so AUC metrics reflect the actual
        # voting outcome rather than the average probability distribution.
        n_classes  = NUM_LABELS
        hard_probs = np.eye(n_classes)[y_hard]   # shape (N, n_classes)
        m = evaluate_and_save(
            "hard", y_true_ref, y_hard, hard_probs, args.results_dir, test_df
        )
        all_ensemble_metrics["hard_voting"] = m

    # ── B2 — Soft Voting ────────────────────────────────────────────────────
    if args.strategy in ("soft", "both"):
        logger.info("\n" + "-" * 60)
        logger.info("  B2 — Soft Voting")
        logger.info("-" * 60)
        y_soft, avg_probs_soft = soft_voting(all_probs)
        m = evaluate_and_save(
            "soft", y_true_ref, y_soft, avg_probs_soft, args.results_dir, test_df
        )
        all_ensemble_metrics["soft_voting"] = m

    # ── Comparison bar charts ────────────────────────────────────────────────
    if len(all_ensemble_metrics) > 1:
        for metric in ["f1_macro", "f1_weighted", "accuracy", "f1_hate_speech"]:
            plot_comparison_bar(
                all_ensemble_metrics,
                metric=metric,
                save_path=str(
                    Path(args.results_dir) / f"ensemble_compare_{metric}.png"
                ),
                title=f"Ensemble Strategies — {metric.replace('_', ' ').title()}",
            )

    # ── Combined ensemble results JSON ───────────────────────────────────────
    combined = {
        k: {
            kk: (vv if not (isinstance(vv, float) and np.isnan(vv)) else None)
            for kk, vv in v.items()
        }
        for k, v in all_ensemble_metrics.items()
    }
    (Path(args.results_dir) / "ensemble_metrics.json").write_text(
        json.dumps(combined, indent=2)
    )

    logger.info(f"\nEnsemble evaluation complete. Results in {args.results_dir}/")


if __name__ == "__main__":
    main()
