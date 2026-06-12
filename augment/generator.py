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

from augment.cluster import cluster_hate_speech, stratified_sample
from augment.templates import build_prompt, style_combinations
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


# ---------------------------------------------------------------------------
# Target computation — equalize classes
# ---------------------------------------------------------------------------

def compute_target(
    train_df: pd.DataFrame,
    balance: Union[str, int] = "majority",
) -> int:
    """
    Compute how many synthetic hate_speech examples are needed.

    Parameters
    ----------
    balance : 'majority' | 'mean' | int
        'majority' — bring hate_speech count up to the largest class count.
        'mean'     — bring hate_speech count up to the mean class count.
        int        — fixed number (legacy mode, ignores class distribution).

    Returns
    -------
    Number of synthetic examples to generate (always ≥ 0).
    """
    counts = train_df["label"].value_counts()
    hate_n = counts.get(HATE_LABEL, 0)

    if isinstance(balance, int):
        target = balance
    elif balance == "majority":
        target = int(counts.max()) - hate_n
    elif balance == "mean":
        target = int(counts.mean()) - hate_n
    else:
        raise ValueError(f"balance must be 'majority', 'mean', or an int. Got: {balance!r}")

    target = max(0, target)

    logger.info("=" * 55)
    logger.info("  CLASS DISTRIBUTION — training set")
    logger.info("=" * 55)
    for label_int, label_name in LABEL_MAP.items():
        n = counts.get(label_int, 0)
        logger.info(f"  {label_name:<25} {n:>6,}")
    logger.info(f"  {'─'*40}")
    logger.info(f"  Imbalance ratio (max/min) : {counts.max() / max(counts.min(), 1):.1f}×")
    logger.info(f"  Balance strategy          : '{balance}'")
    logger.info(f"  Synthetic target          : {target:,} hate_speech examples")
    logger.info("=" * 55)

    return target


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
    Full 7-step few-shot augmentation pipeline.

    Generates synthetic hate_speech examples until the class distribution in
    the training set is equalised according to the `balance` strategy.

    Parameters
    ----------
    train_df   : Original (unaugmented) training DataFrame.
    balance    : Target strategy — 'majority', 'mean', or an int (fixed count).
    llm_id     : HuggingFace model ID for the causal LLM generator.
    k          : Number of semantic clusters (Etapa 1).
    sim_threshold : Cosine similarity threshold for Etapa 7 (default 0.70).
    seed       : Random seed for full reproducibility.
    n_variants : Number of examples to request per LLM call (Etapa 4).
    few_shot_n : Number of real examples to include in each prompt (Etapa 3).

    Returns
    -------
    Augmented training DataFrame (real data + synthetic hate_speech rows).
    The synthetic rows have an `is_synthetic = True` column for tracking.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Compute target ───────────────────────────────────────────────────────
    target_count = compute_target(train_df, balance=balance)

    if target_count == 0:
        logger.info("Classes already balanced — no augmentation needed.")
        result = train_df.copy()
        result["is_synthetic"] = False
        return result

    # ── Etapa 1 — Estratificação Semântica ───────────────────────────────────
    logger.info("[Etapa 1] Semantic clustering of hate speech examples …")
    hate_df = train_df[train_df["label"] == HATE_LABEL].copy()
    logger.info(f"  Real hate_speech examples available: {len(hate_df):,}")

    clustered_df, _embeddings, _kmeans = cluster_hate_speech(
        hate_df, k=k, seed=seed
    )
    logger.info(f"  Clusters formed: {k}")
    for name, grp in clustered_df.groupby("cluster_name"):
        logger.info(f"    {name}: {len(grp):,} examples")

    # ── Etapa 3 — Amostragem Estratificada ──────────────────────────────────
    logger.info("[Etapa 3] Stratified sampling of few-shot seeds …")
    seeds_per_cluster = stratified_sample(
        clustered_df, target_per_cluster=few_shot_n * 4, seed=seed
    )

    # ── Etapas 6 & 7 — Setup filters ────────────────────────────────────────
    sem_filter = SemanticFilter(threshold=sim_threshold)
    sem_filter.fit(hate_df["text"].tolist())

    # ── Load LLM (Etapa 4) ──────────────────────────────────────────────────
    model, tokenizer = load_llm(llm_id, device)

    # ── Etapas 4+5 — Few-Shot Prompting + Diversidade Linguística ───────────
    logger.info(
        f"[Etapas 4+5] Generating {target_count:,} synthetic examples "
        f"across {k} clusters …"
    )
    styles = style_combinations(n=20)

    all_generated: List[str] = []
    seen: Set[str] = set(hate_df["text"].str.lower().tolist())

    # Distribute target evenly across clusters; remainder goes to last cluster
    base_per_cluster = target_count // k
    remainder        = target_count % k
    cluster_names    = list(seeds_per_cluster.keys())

    pbar = tqdm(total=target_count, desc="Synthetic hate speech", unit="ex")

    for cluster_idx, (cluster_name, seed_df) in enumerate(seeds_per_cluster.items()):
        # Last cluster absorbs the remainder
        cluster_target = base_per_cluster + (remainder if cluster_idx == len(cluster_names) - 1 else 0)
        cluster_generated: List[str] = []
        style_idx = 0

        while len(cluster_generated) < cluster_target:
            # Etapa 2 — Template semântico com variação de estilo
            style = styles[style_idx % len(styles)]
            style_idx += 1

            few_shot_examples = seed_df["text"].sample(
                n=min(few_shot_n, len(seed_df)),
                random_state=seed + style_idx,
            ).tolist()

            prompt = build_prompt(
                category=cluster_name,
                few_shot_examples=few_shot_examples,
                n_variants=n_variants,
                tone=style["tone"],
                length=style["length"],
                formality=style["formality"],
            )

            # Etapa 4 — Generate
            raw_lines = generate_batch(prompt, model, tokenizer, device)

            # Etapa 6 — Filter artefacts & duplicates
            clean = filter_generated(raw_lines, seen=seen)

            # Etapa 7 — Semantic similarity filter
            if clean:
                clean = sem_filter.filter(clean)

            cluster_generated.extend(clean)
            pbar.update(len(clean))

            if style_idx > 300:   # safety valve per cluster
                logger.warning(
                    f"  Safety limit (300 calls) reached for cluster '{cluster_name}'. "
                    f"Collected {len(cluster_generated):,}/{cluster_target:,}."
                )
                break

        all_generated.extend(cluster_generated[:cluster_target])

    pbar.close()

    # ── Free GPU memory ──────────────────────────────────────────────────────
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Final deduplication ──────────────────────────────────────────────────
    logger.info("[Etapa 6] Final deduplication …")
    all_generated = deduplicate(all_generated)[:target_count]

    n_generated = len(all_generated)
    logger.info(f"  Synthetic examples accepted : {n_generated:,} / {target_count:,} target")

    # ── Compose augmented DataFrame ──────────────────────────────────────────
    synthetic_df = pd.DataFrame({
        "text":         all_generated,
        "label":        [HATE_LABEL] * n_generated,
        "label_name":   ["hate_speech"] * n_generated,
        "is_synthetic": [True] * n_generated,
    })

    real_df = train_df.copy()
    real_df["is_synthetic"] = False

    augmented_df = pd.concat([real_df, synthetic_df], ignore_index=True)

    # ── Log final class distribution ─────────────────────────────────────────
    final_counts = augmented_df["label"].value_counts()
    logger.info("=" * 55)
    logger.info("  CLASS DISTRIBUTION — augmented training set")
    logger.info("=" * 55)
    for label_int, label_name in LABEL_MAP.items():
        n = final_counts.get(label_int, 0)
        logger.info(f"  {label_name:<25} {n:>6,}")
    final_imbalance = final_counts.max() / max(final_counts.min(), 1)
    logger.info(f"  Imbalance ratio (max/min)  : {final_imbalance:.2f}×  (was {train_df['label'].value_counts().max() / max(train_df['label'].value_counts().min(), 1):.1f}×)")
    logger.info(f"  Total training samples     : {len(augmented_df):,}")
    logger.info(f"    Real                     : {len(real_df):,}")
    logger.info(f"    Synthetic (hate_speech)  : {n_generated:,}")
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
