"""
augment/generator.py — Etapas 4 & 5: Few-Shot Prompting + Diversidade Linguística

Pipeline completo de augmentação avançada com as 7 etapas:

  1. Clustering semântico  → cluster.py
  2. Templates semânticos  → templates.py
  3. Amostragem estratificada → cluster.py::stratified_sample
  4. Few-shot prompting    → este arquivo
  5. Diversidade linguística → este arquivo (20 variações de estilo)
  6. Filtragem automática  → filter.py
  7. Similaridade semântica → similarity.py

Target de geração
-----------------
Por padrão, o pipeline calcula automaticamente quantos exemplos sintéticos
de hate_speech são necessários para **equalizar** as classes no conjunto de
treino:

  • 'majority' (padrão) — hate_speech → tamanho da classe mais frequente
  • 'mean'              — hate_speech → média do tamanho das 3 classes
  • int                 — número fixo explícito (modo legado)

Exemplo de distribuição típica do Davidson et al. (2017) após split 70%:
  offensive_language : ~13.400 amostras  ← majoritária
  neither            :  ~3.100 amostras
  hate_speech        :    ~870 amostras  ← minoritária
  → Target 'majority': gera ~12.530 sintéticos para nivelar hate_speech

Usage:
    python -m augment.generator                          # equalize majority
    python -m augment.generator --balance majority
    python -m augment.generator --balance mean
    python -m augment.generator --balance 5000           # número fixo
    python -m augment.generator --llm Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_logger, set_seed, SEED
from data_loader import load_raw_data, split_data, LABEL_MAP

from augment.cluster import cluster_hate_speech, cluster_neither, stratified_sample
from augment.templates import build_prompt, build_neither_prompt, style_combinations
from augment.filter import filter_generated, deduplicate
from augment.similarity import SemanticFilter

logger = get_logger(__name__, "results/augmentation_advanced.log")

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_ID  = "Qwen/Qwen2.5-1.5B-Instruct"
N_VARIANTS      = 10        # generation batch per LLM call
FEW_SHOT_N      = 5         # few-shot seeds per prompt
SIM_THRESHOLD   = 0.70      # cosine similarity threshold (Etapa 7)
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.85

HATE_LABEL      = 0         # label int for hate_speech in Davidson dataset
NEITHER_LABEL   = 2         # label int for neither in Davidson dataset
MINORITY_LABELS = [HATE_LABEL, NEITHER_LABEL]   # both need augmentation


# ---------------------------------------------------------------------------
# Target computation — equalize classes
# ---------------------------------------------------------------------------

def compute_targets(
    train_df: pd.DataFrame,
    balance: Union[str, int] = "majority",
    target_labels: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Compute how many synthetic examples are needed per minority class.

    Parameters
    ----------
    balance       : 'majority' | 'mean' | int
        'majority' — bring each minority class up to the largest class count.
        'mean'     — bring each minority class up to the mean class count.
        int        — fixed number for each minority class (legacy mode).
    target_labels : List of label ints to augment. Defaults to MINORITY_LABELS.

    Returns
    -------
    Dict {label_int: n_to_generate}  (only labels with n > 0 are included).
    """
    if target_labels is None:
        target_labels = MINORITY_LABELS

    counts = train_df["label"].value_counts()

    logger.info("=" * 55)
    logger.info("  CLASS DISTRIBUTION — training set")
    logger.info("=" * 55)
    for label_int, label_name in LABEL_MAP.items():
        n = counts.get(label_int, 0)
        logger.info(f"  {label_name:<25} {n:>6,}")
    logger.info(f"  {'─'*40}")
    logger.info(f"  Imbalance ratio (max/min) : {counts.max() / max(counts.min(), 1):.1f}×")
    logger.info(f"  Balance strategy          : '{balance}'")

    targets: Dict[int, int] = {}
    for lbl in target_labels:
        lbl_n = counts.get(lbl, 0)
        if isinstance(balance, int):
            t = balance
        elif balance == "majority":
            t = int(counts.max()) - lbl_n
        elif balance == "mean":
            t = int(counts.mean()) - lbl_n
        else:
            raise ValueError(f"balance must be 'majority', 'mean', or an int. Got: {balance!r}")
        t = max(0, t)
        if t > 0:
            targets[lbl] = t
            logger.info(f"  Synthetic target ({LABEL_MAP[lbl]:<18}): {t:,}")
        else:
            logger.info(f"  {LABEL_MAP[lbl]:<25} already at target — skipping")

    logger.info("=" * 55)
    return targets


# Keep old name as alias for backwards compatibility
def compute_target(
    train_df: pd.DataFrame,
    balance: Union[str, int] = "majority",
) -> int:
    """Legacy single-class version — returns target for hate_speech only."""
    targets = compute_targets(train_df, balance=balance, target_labels=[HATE_LABEL])
    return targets.get(HATE_LABEL, 0)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def load_llm(model_id: str, device: str) -> tuple:
    """Load a causal LLM and its tokenizer onto the target device."""
    logger.info(f"Loading LLM: {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_batch(
    prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> List[str]:
    """Run one LLM inference call and return raw output lines."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant for academic research."},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = [
        out[len(inp):]
        for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
    return [line.strip() for line in response.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# Main augmentation pipeline
# ---------------------------------------------------------------------------

def _augment_one_class(
    label_int: int,
    class_df: pd.DataFrame,
    target_count: int,
    model,
    tokenizer,
    device: str,
    k: int,
    sim_threshold: float,
    seed: int,
    n_variants: int,
    few_shot_n: int,
    seen: Set[str],
    styles: List[dict],
) -> List[str]:
    """
    Run the full generation + top-up loop for a single minority class.

    Returns a list of accepted synthetic texts (len ≤ target_count).
    Updates `seen` in-place so subsequent classes don't duplicate these texts.
    """
    label_name = LABEL_MAP[label_int]
    is_hate    = (label_int == HATE_LABEL)

    # ── Etapa 1 — Clustering ────────────────────────────────────────────────
    logger.info(f"[Etapa 1] Semantic clustering of '{label_name}' examples …")
    logger.info(f"  Real examples available: {len(class_df):,}")

    if is_hate:
        clustered_df, _, _, _ = cluster_hate_speech(class_df, k=k, seed=seed)
        _build_prompt = build_prompt
    else:
        clustered_df, _, _, _ = cluster_neither(class_df, k=k, seed=seed)
        _build_prompt = build_neither_prompt

    for name, grp in clustered_df.groupby("cluster_name"):
        logger.info(f"    {name}: {len(grp):,} examples")

    # ── Etapa 3 — Stratified sampling ───────────────────────────────────────
    seeds_per_cluster = stratified_sample(
        clustered_df, target_per_cluster=few_shot_n * 4, seed=seed
    )

    # ── Etapas 4+5 — Generation loop ────────────────────────────────────────
    all_generated: List[str] = []
    base_per_cluster = target_count // k
    remainder        = target_count % k
    cluster_sem_filters: Dict[str, SemanticFilter] = {}

    pbar = tqdm(total=target_count, desc=f"Synthetic {label_name}", unit="ex")

    for cluster_idx, (cluster_name, seed_df) in enumerate(seeds_per_cluster.items()):
        cluster_target    = base_per_cluster + (1 if cluster_idx < remainder else 0)
        cluster_generated: List[str] = []
        style_idx = 0

        sem_filter = SemanticFilter(threshold=sim_threshold)
        sem_filter.fit(seed_df["text"].tolist())
        cluster_sem_filters[cluster_name] = sem_filter

        while len(cluster_generated) < cluster_target:
            style = styles[style_idx % len(styles)]
            style_idx += 1

            few_shot_examples = seed_df["text"].sample(
                n=min(few_shot_n, len(seed_df)),
                random_state=seed + style_idx,
            ).tolist()

            prompt = _build_prompt(
                category=cluster_name,
                few_shot_examples=few_shot_examples,
                n_variants=n_variants,
                tone=style["tone"],
                length=style["length"],
                formality=style["formality"],
            )

            raw_lines = generate_batch(prompt, model, tokenizer, device)
            clean     = filter_generated(raw_lines, seen=seen)
            if clean:
                clean = sem_filter.filter(clean)

            cluster_generated.extend(clean)
            pbar.update(min(len(clean), cluster_target - len(cluster_generated) + len(clean)))

            if style_idx > 300:
                logger.warning(
                    f"  Safety limit (300 calls) reached for cluster '{cluster_name}'. "
                    f"Collected {len(cluster_generated):,}/{cluster_target:,}."
                )
                break

        all_generated.extend(cluster_generated[:cluster_target])

    pbar.close()

    # ── Top-up pass ─────────────────────────────────────────────────────────
    deficit = target_count - len(all_generated)
    if deficit > 0:
        logger.warning(
            f"[Top-up/{label_name}] {deficit:,} short "
            f"({len(all_generated):,}/{target_count:,}). Running top-up …"
        )
        cluster_list    = list(seeds_per_cluster.items())
        n_clusters      = len(cluster_list)
        topup_style_idx = 500
        topup_attempts  = 0
        MAX_TOPUP       = 1000

        pbar_topup = tqdm(total=deficit, desc=f"Top-up {label_name}", unit="ex")

        while len(all_generated) < target_count and topup_attempts < MAX_TOPUP:
            cluster_name, seed_df = cluster_list[topup_attempts % n_clusters]
            sf = cluster_sem_filters[cluster_name]

            style = styles[topup_style_idx % len(styles)]
            topup_style_idx += 1
            topup_attempts  += 1

            few_shot_examples = seed_df["text"].sample(
                n=min(few_shot_n, len(seed_df)),
                random_state=seed + topup_style_idx + 9999,
            ).tolist()

            prompt = _build_prompt(
                category=cluster_name,
                few_shot_examples=few_shot_examples,
                n_variants=n_variants,
                tone=style["tone"],
                length=style["length"],
                formality=style["formality"],
            )

            raw_lines = generate_batch(prompt, model, tokenizer, device)
            clean     = filter_generated(raw_lines, seen=seen)
            if clean:
                clean = sf.filter(clean)

            needed   = target_count - len(all_generated)
            accepted = clean[:needed]
            all_generated.extend(accepted)
            pbar_topup.update(len(accepted))

        pbar_topup.close()

        if len(all_generated) < target_count:
            logger.warning(
                f"[Top-up/{label_name}] Exhausted {MAX_TOPUP} attempts. "
                f"Final: {len(all_generated):,}/{target_count:,}. "
                "Consider lowering --threshold or raising --k."
            )
        else:
            logger.info(f"[Top-up/{label_name}] Target reached: {len(all_generated):,}.")

    # Update global seen set so next class won't duplicate these texts
    seen.update(t.lower() for t in all_generated)

    # Final deduplication
    return deduplicate(all_generated)[:target_count]


def advanced_augment(
    train_df: pd.DataFrame,
    balance: Union[str, int] = "majority",
    llm_id: str = DEFAULT_LLM_ID,
    k: int = 6,
    sim_threshold: float = SIM_THRESHOLD,
    seed: int = SEED,
    n_variants: int = N_VARIANTS,
    few_shot_n: int = FEW_SHOT_N,
) -> pd.DataFrame:
    """
    Full 7-step few-shot augmentation pipeline for ALL minority classes.

    Augments both hate_speech (label=0) and neither (label=2) until each
    reaches the target count defined by `balance`. The LLM is loaded once
    and shared across both augmentation passes.

    Parameters
    ----------
    train_df      : Original (unaugmented) training DataFrame.
    balance       : 'majority', 'mean', or int (fixed count per class).
    llm_id        : HuggingFace model ID for the causal LLM generator.
    k             : Number of semantic clusters per class (Etapa 1).
    sim_threshold : Cosine similarity threshold for Etapa 7 (default 0.70).
    seed          : Random seed for full reproducibility.
    n_variants    : Number of examples to request per LLM call (Etapa 4).
    few_shot_n    : Number of real examples to include in each prompt (Etapa 3).

    Returns
    -------
    Augmented training DataFrame (real + synthetic rows).
    Synthetic rows have `is_synthetic = True` and `label_name` set correctly.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Compute targets for each minority class ──────────────────────────────
    targets = compute_targets(train_df, balance=balance)

    if not targets:
        logger.info("All classes already balanced — no augmentation needed.")
        result = train_df.copy()
        result["is_synthetic"] = False
        return result

    # ── Load LLM once (shared across all classes) ────────────────────────────
    model, tokenizer = load_llm(llm_id, device)
    styles = style_combinations(n=20)

    # Global seen set: start with ALL real texts to avoid any duplicates
    seen: Set[str] = set(train_df["text"].str.lower().tolist())

    # ── Augment each minority class ──────────────────────────────────────────
    all_synthetic_rows: List[pd.DataFrame] = []

    for label_int, target_count in targets.items():
        label_name = LABEL_MAP[label_int]
        logger.info(f"\n{'='*55}")
        logger.info(f"  AUGMENTING CLASS: {label_name.upper()}  (target: {target_count:,})")
        logger.info(f"{'='*55}")

        class_df = train_df[train_df["label"] == label_int].copy()

        generated_texts = _augment_one_class(
            label_int=label_int,
            class_df=class_df,
            target_count=target_count,
            model=model,
            tokenizer=tokenizer,
            device=device,
            k=k,
            sim_threshold=sim_threshold,
            seed=seed,
            n_variants=n_variants,
            few_shot_n=few_shot_n,
            seen=seen,
            styles=styles,
        )

        n_gen = len(generated_texts)
        logger.info(f"  Accepted {n_gen:,} / {target_count:,} synthetic '{label_name}' examples.")

        if n_gen > 0:
            synthetic_df = pd.DataFrame({
                "text":         generated_texts,
                "label":        [label_int] * n_gen,
                "label_name":   [label_name] * n_gen,
                "is_synthetic": [True] * n_gen,
            })
            all_synthetic_rows.append(synthetic_df)

    # ── Free GPU memory ──────────────────────────────────────────────────────
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Compose augmented DataFrame ──────────────────────────────────────────
    real_df = train_df.copy()
    real_df["is_synthetic"] = False

    parts = [real_df] + all_synthetic_rows
    augmented_df = pd.concat(parts, ignore_index=True)

    # Total synthetic examples across all augmented classes
    n_total_synthetic = sum(len(d) for d in all_synthetic_rows)

    # ── Log final class distribution ─────────────────────────────────────────
    final_counts = augmented_df["label"].value_counts()
    logger.info("\n" + "=" * 55)
    logger.info("  CLASS DISTRIBUTION — augmented training set")
    logger.info("=" * 55)
    for label_int, label_name in LABEL_MAP.items():
        n = final_counts.get(label_int, 0)
        logger.info(f"  {label_name:<25} {n:>6,}")
    final_imbalance = final_counts.max() / max(final_counts.min(), 1)
    logger.info(f"  Imbalance ratio (max/min)  : {final_imbalance:.2f}×  (was {train_df['label'].value_counts().max() / max(train_df['label'].value_counts().min(), 1):.1f}×)")
    logger.info(f"  Total training samples     : {len(augmented_df):,}")
    logger.info(f"    Real                     : {len(real_df):,}")
    logger.info(f"    Synthetic (all classes)  : {n_total_synthetic:,}")
    logger.info("=" * 55)

    return augmented_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advanced Few-Shot Data Augmentation — class-equalising pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m augment.generator                   # equalize to majority class\n"
            "  python -m augment.generator --balance mean    # equalize to class mean\n"
            "  python -m augment.generator --balance 5000    # generate exactly 5000\n"
        ),
    )
    p.add_argument(
        "--balance",
        type=str,
        default="majority",
        help=(
            "Class equalisation strategy. "
            "'majority': match the largest class count (default). "
            "'mean': match the mean class count. "
            "An integer: generate exactly that many synthetic examples."
        ),
    )
    p.add_argument("--llm",       type=str,   default=DEFAULT_LLM_ID,
                   help="HuggingFace model ID for the generative LLM.")
    p.add_argument("--k",         type=int,   default=6,
                   help="Number of semantic clusters (Etapa 1).")
    p.add_argument("--threshold", type=float, default=SIM_THRESHOLD,
                   help="Cosine similarity threshold for Etapa 7 (default 0.70).")
    p.add_argument("--seed",      type=int,   default=SEED)
    p.add_argument("--data_dir",  type=str,   default="data")
    p.add_argument("--output",    type=str,   default="data/train_augmented.csv")
    return p.parse_args()


def _parse_balance(raw: str) -> Union[str, int]:
    """Coerce CLI --balance value to str or int."""
    if raw in ("majority", "mean"):
        return raw
    try:
        return int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--balance must be 'majority', 'mean', or an integer. Got: {raw!r}"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    balance = _parse_balance(args.balance)

    logger.info("=" * 60)
    logger.info("  ADVANCED FEW-SHOT AUGMENTATION — 7-STEP PIPELINE")
    logger.info(f"  Balance strategy : {balance}")
    logger.info("=" * 60)

    df = load_raw_data(args.data_dir)
    train_df, _, _ = split_data(df, seed=args.seed)
    logger.info(f"Training set loaded: {len(train_df):,} samples")

    augmented_df = advanced_augment(
        train_df,
        balance=balance,
        llm_id=args.llm,
        k=args.k,
        sim_threshold=args.threshold,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(out_path, index=False)
    logger.info(f"Augmented dataset saved → {out_path}")


if __name__ == "__main__":
    main()
