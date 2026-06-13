"""
train.py

Trains three Transformer models under an identical experimental protocol:
  1. bert-base-uncased   (BERT Base)
  2. GroNLP/hateBERT     (HateBERT)
  3. roberta-base        (RoBERTa Base)

Each model is fine-tuned with:
  • AdamW optimizer (weight decay = 0.01)
  • Linear warmup + cosine decay scheduler
  • Weighted cross-entropy (inverse-frequency class weights)
  • Early stopping on validation macro F1 (patience = 3 epochs)
  • Checkpointing of the best model state

Usage:
    python train.py                         # train all three models
    python train.py --model bert-base-uncased
    python train.py --model GroNLP/hateBERT
    python train.py --model roberta-base
    python train.py --epochs 5 --batch_size 32
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from utils import (
    SEED,
    CLASS_NAMES,
    compute_metrics,
    get_device,
    get_logger,
    plot_training_curves,
    set_seed,
)

from data_loader import (
    NUM_LABELS,
    HateSpeechDataset,
    compute_class_weights,
    describe_dataset,
    load_raw_data,
    split_data,
)

logger = get_logger(__name__, "results/training.log")

# Experimental hyperparameters (identical for all models)

HYPERPARAMS: Dict = {
    "seed":            SEED,          # 42 — fixed for full reproducibility
    "batch_size":      16,            # 16 balances GPU memory with stable gradients
    "learning_rate":   1e-5,          # canonical fine-tuning LR for BERT-family models
    "num_epochs":      200,            # upper bound; early stopping kicks in earlier
    "weight_decay":    0.01,          # L2 regularization (AdamW default)
    "warmup_ratio":    0.1,           # 10 % of total steps for linear warmup
    "max_grad_norm":   1.0,           # gradient clipping for training stability
    "early_stop_patience": 12,         # stop if val F1 doesn't improve for 12 epochs
    "max_length":      128,           # maximum token length
    "num_workers":     0,             # set > 0 only if OS supports fork-safe multiprocessing
}


# Model registry
MODEL_REGISTRY: Dict[str, Dict] = {
    "bert-base-uncased": {
        "hf_name":      "bert-base-uncased",
        "short_name":   "BERT-Base",
        "architecture": "BERT (Encoder-only Transformer, 12 layers, 768 hidden, 12 heads)",
        "params":       "~110 M",
        "pretraining":  "MLM + NSP on BooksCorpus + English Wikipedia",
        "advantages":   [
            "Strong general-purpose language understanding baseline",
            "Well-validated on hate speech tasks (Mozafari et al., 2020)",
        ],
        "limitations":  [
            "Pre-trained on clean text; may not capture hateful vernacular",
            "Fixed vocabulary may struggle with slang/neologisms",
        ],
    },
    "GroNLP/hateBERT": {
        "hf_name":      "GroNLP/hateBERT",
        "short_name":   "HateBERT",
        "architecture": "BERT (identical architecture to bert-base-uncased)",
        "params":       "~110 M",
        "pretraining":  (
            "Continued pre-training of BERT on 1.5M Reddit posts from "
            "r/offensivespeech and banned communities (Caselli et al., 2021)"
        ),
        "advantages":   [
            "Domain-adapted: exposed to slang, profanity, and hate speech during MLM",
            "Superior embedding of offensive vocabulary",
            "Directly relevant pre-training distribution for this task",
        ],
        "limitations":  [
            "May over-trigger on surface profanity regardless of context",
            "Reddit distribution differs from Twitter (Davidson dataset)",
        ],
    },
    "roberta-base": {
        "hf_name":      "roberta-base",
        "short_name":   "RoBERTa-Base",
        "architecture": "RoBERTa (BERT without NSP, dynamic masking, 12 layers, 768 hidden)",
        "params":       "~125 M",
        "pretraining":  (
            "MLM only (no NSP) on 160 GB of text: BooksCorpus, CC-News, "
            "OpenWebText, Stories; trained 10× longer than BERT"
        ),
        "advantages":   [
            "Improved training recipe yields stronger representations",
            "Better performance on GLUE/SuperGLUE than BERT-Base",
            "Dynamic masking increases pre-training diversity",
        ],
        "limitations":  [
            "Not specialized for hate speech; relies on transfer capacity",
            "Slightly more parameters; marginally longer training",
        ],
    },
}


# Training utilities

class EarlyStopping:

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            logger.info(
                f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                f"(best={self.best_score:.4f})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False, ncols=90):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # token_type_ids is absent for RoBERTa; pass only when present
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(**kwargs)
        logits  = outputs.logits                      # (B, num_labels)

        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Eval", leave=False, ncols=90):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

        outputs = model(**kwargs)
        logits  = outputs.logits
        loss    = loss_fn(logits, labels)
        probs   = torch.softmax(logits, dim=-1)

        total_loss  += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.vstack(all_probs)
    mean_loss = total_loss / len(loader)
    return mean_loss, y_true, y_pred, y_prob


# Main training function

def train_model(
    model_key: str,
    train_df,
    val_df,
    hyperparams: Dict,
    device: torch.device,
    results_dir: str = "results",
    models_dir: str  = "models",
) -> Dict:
    
    set_seed(hyperparams["seed"])
    cfg        = MODEL_REGISTRY[model_key]
    hf_name    = cfg["hf_name"]
    short_name = cfg["short_name"]
    safe_name  = hf_name.replace("/", "_")

    logger.info(f"\n{'-'*60}")
    logger.info(f"  Training: {short_name}  ({hf_name})")
    logger.info(f"  Architecture: {cfg['architecture']}")
    logger.info(f"  Parameters:   {cfg['params']}")
    logger.info(f"{'-'*60}\n")

    # Tokenizer and datasets
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    train_dataset = HateSpeechDataset(train_df, tokenizer, hyperparams["max_length"])
    val_dataset   = HateSpeechDataset(val_df,   tokenizer, hyperparams["max_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=hyperparams["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams["batch_size"] * 2,
        shuffle=False,
        num_workers=hyperparams["num_workers"],
        pin_memory=device.type == "cuda",
    )

    # Model
    logger.info("Loading model ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,   # allows replacing pre-trained heads
    )

    # Layer Freezing (Regularization)
    if hasattr(model, "roberta"):
        encoder = model.roberta.encoder.layer
        embeddings = model.roberta.embeddings
    else:
        encoder = model.bert.encoder.layer
        embeddings = model.bert.embeddings

    for param in embeddings.parameters():
        param.requires_grad = False

    for i in range(3):
        for param in encoder[i].parameters():
            param.requires_grad = False

    logger.info("Froze embeddings and first 3 encoder layers for regularization.")

    model = model.to(device)

    # Loss weighted cross-entropy for class imbalance
    class_weights = compute_class_weights(train_df).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": hyperparams["weight_decay"],
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=hyperparams["learning_rate"])

    total_steps   = len(train_loader) * hyperparams["num_epochs"]
    warmup_steps  = int(total_steps * hyperparams["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    early_stopper = EarlyStopping(patience=hyperparams["early_stop_patience"])
    best_f1   = -1.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_f1_macro": []}

    checkpoint_path = Path(models_dir) / safe_name / "best_model"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, hyperparams["num_epochs"] + 1):
        logger.info(f"Epoch {epoch}/{hyperparams['num_epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            hyperparams["max_grad_norm"],
        )
        val_loss, y_true, y_pred, y_prob = evaluate_loader(model, val_loader, loss_fn, device)
        val_metrics = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
        val_f1 = val_metrics["f1_macro"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_macro"].append(val_f1)

        logger.info(
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1_macro={val_f1:.4f}  val_acc={val_metrics['accuracy']:.4f}"
        )

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_epoch = epoch
            model.save_pretrained(str(checkpoint_path))
            tokenizer.save_pretrained(str(checkpoint_path))
            logger.info(f" V New best model saved (F1={best_f1:.4f})")

        if early_stopper.step(val_f1):
            logger.info(f"  Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    logger.info(
        f"\n{short_name} training complete. "
        f"Best val F1 Macro: {best_f1:.4f} at epoch {best_epoch}. "
        f"Total time: {elapsed/60:.1f} min\n"
    )

    # Plot training curves
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        history["val_f1_macro"],
        model_name=short_name,
        save_path=f"{results_dir}/{safe_name}_training_curves.png",
    )

    # Save hyperparams + history
    meta = {
        "model_key":    model_key,
        "short_name":   short_name,
        "hyperparams":  hyperparams,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_f1,
        "history":      history,
        "training_time_min": round(elapsed / 60, 2),
    }
    with open(f"{results_dir}/{safe_name}_training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# CLI entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hate Speech Benchmark: Training")
    p.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + list(MODEL_REGISTRY.keys()),
        help="Model to train. 'all' trains all three models sequentially.",
    )
    p.add_argument("--epochs",     type=int,   default=HYPERPARAMS["num_epochs"])
    p.add_argument("--batch_size", type=int,   default=HYPERPARAMS["batch_size"])
    p.add_argument("--lr",         type=float, default=HYPERPARAMS["learning_rate"])
    p.add_argument("--seed",       type=int,   default=HYPERPARAMS["seed"])
    p.add_argument("--use_augmented", action="store_true", help="Use augmented dataset if available.")
    p.add_argument("--data_dir",   type=str,   default="data")
    p.add_argument("--results_dir",type=str,   default="results")
    p.add_argument("--models_dir", type=str,   default="models")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")

    hp = dict(HYPERPARAMS)
    hp["num_epochs"]    = args.epochs
    hp["batch_size"]    = args.batch_size
    hp["learning_rate"] = args.lr
    hp["seed"]          = args.seed

    # Load and split data
    df = load_raw_data(args.data_dir)
    logger.info(f"Total samples: {len(df):,}")
    describe_dataset(df, "full")

    train_df, val_df, test_df = split_data(df, seed=args.seed)
    
    if args.use_augmented:

        aug_path = Path(args.data_dir) / "train_augmented.csv"
        if aug_path.exists():
            train_df = pd.read_csv(aug_path)
            logger.info(f"Loaded augmented training data from {aug_path} (N={len(train_df):,})")
        else:
            logger.warning("Augmented data requested but not found. Falling back to raw data.")

    describe_dataset(train_df, "train")
    describe_dataset(val_df,   "val")
    describe_dataset(test_df,  "test")

    # Save test set for evaluate.py
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    test_df.to_csv(f"{args.data_dir}/test.csv", index=False)

    # Select models to train
    if args.model == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = [args.model]

    # Train each model
    all_meta = {}
    for mk in model_keys:
        meta = train_model(mk, train_df, val_df, hp, device, args.results_dir, args.models_dir)
        all_meta[mk] = meta

    # Save combined training summary
    with open(f"{args.results_dir}/all_training_meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    logger.info("All models trained. Run evaluate.py to generate final results.")

if __name__ == "__main__":
    main()
