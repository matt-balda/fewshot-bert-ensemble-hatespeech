import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import get_logger, SEED

logger = get_logger(__name__, "results/statistical_tests.log")


# Wilcoxon signed-rank test

def wilcoxon_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = 0.05,
) -> Dict:
    
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)
    diff      = correct_a - correct_b

    n_ties     = int((diff == 0).sum())
    n_a_better = int((diff >  0).sum())
    n_b_better = int((diff <  0).sum())

    result = {
        "comparison":  f"{name_a} vs {name_b}",
        "n_samples":   len(y_true),
        "n_a_better":  n_a_better,
        "n_b_better":  n_b_better,
        "n_ties":      n_ties,
    }

    # Need at least one non-zero difference to run the test
    if n_a_better + n_b_better < 1:
        result.update({
            "statistic":   None,
            "p_value":     1.0,
            "significant": False,
            "winner":      "tie (identical predictions)",
        })
        return result

    try:
        stat, p = wilcoxon(correct_a, correct_b, alternative="two-sided")
    except ValueError as exc:
        result.update({
            "statistic":   None,
            "p_value":     None,
            "significant": False,
            "winner":      f"error: {exc}",
        })
        return result

    significant = p < alpha

    if not significant:
        winner = "no significant difference"
    elif n_a_better > n_b_better:
        winner = name_a
    else:
        winner = name_b

    result.update({
        "statistic":   float(stat),
        "p_value":     float(p),
        "significant": significant,
        "winner":      winner,
    })
    return result


def pairwise_wilcoxon(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    
    pairs   = list(combinations(preds.keys(), 2))
    results = [
        wilcoxon_test(y_true, preds[a], preds[b], name_a=a, name_b=b, alpha=alpha)
        for a, b in pairs
    ]
    return pd.DataFrame(results)


# Data loading

def load_predictions(results_dir: str) -> Dict[str, np.ndarray]:
    
    preds: Dict[str, np.ndarray] = {}
    for f in sorted(Path(results_dir).rglob("*_predictions.csv")):
        df = pd.read_csv(f)
        if "pred_label" not in df.columns:
            continue
        parent = f.parent.name
        stem   = f.stem.replace("_predictions", "")
        label  = f"{parent}/{stem}" if parent != Path(results_dir).name else stem
        preds[label] = df["pred_label"].values
    return preds


def load_y_true(results_dir: str) -> Optional[np.ndarray]:
    
    for f in sorted(Path(results_dir).rglob("*_predictions.csv")):
        df = pd.read_csv(f)
        if "true_label" in df.columns:
            return df["true_label"].values
    return None

# Text report

def build_report(df: pd.DataFrame, alpha: float) -> str:
    
    sig  = df[df["significant"] == True]
    nsig = df[df["significant"] == False]

    lines = [
        "=" * 70,
        "  WILCOXON SIGNED-RANK TEST — PAIRWISE COMPARISON",
        "=" * 70,
        f"  Significance level (α) : {alpha}",
        f"  Test                   : Two-sided Wilcoxon signed-rank",
        f"  Pairs tested           : {len(df)}",
        f"  Significant pairs      : {len(sig)}",
        f"  Non-significant pairs  : {len(nsig)}",
        "",
        "── Results ──────────────────────────────────────────────────────────",
    ]

    display_cols = ["comparison", "n_a_better", "n_b_better", "n_ties",
                    "statistic", "p_value", "significant", "winner"]
    available = [c for c in display_cols if c in df.columns]
    lines.append(df[available].to_string(index=False))

    lines += ["", "── Significant differences ──────────────────────────────────────────"]
    if sig.empty:
        lines.append("  → No pair showed a statistically significant difference (α={}).".format(alpha))
    else:
        for _, row in sig.iterrows():
            p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "N/A"
            lines.append(
                f"  ✓ {row['comparison']:<45}  p={p_str}  winner={row['winner']}"
            )

    lines += ["", "=" * 70]
    return "\n".join(lines)


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairwise Wilcoxon signed-rank tests for the hate speech benchmark."
    )
    p.add_argument("--results_dir", type=str,  default="results",
                   help="Root directory with *_predictions.csv files (scans subdirs).")
    p.add_argument("--alpha",       type=float, default=0.05,
                   help="Significance level (default 0.05).")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  WILCOXON SIGNED-RANK STATISTICAL ANALYSIS")
    logger.info("=" * 60)

    # Load predictions
    preds  = load_predictions(args.results_dir)
    y_true = load_y_true(args.results_dir)

    if y_true is None or len(preds) < 2:
        logger.error(
            "Need ≥ 2 *_predictions.csv files. "
            "Run evaluate.py and ensemble.py first."
        )
        return

    logger.info(f"  Systems found ({len(preds)}):")
    for name in preds:
        logger.info(f"    • {name}")

    # Pairwise Wilcoxon
    logger.info(f"\n--- Pairwise Wilcoxon Signed-Rank Tests (α={args.alpha}) ---")
    wilcoxon_df = pairwise_wilcoxon(y_true, preds, alpha=args.alpha)

    display_cols = ["comparison", "n_a_better", "n_b_better", "n_ties",
                    "statistic", "p_value", "significant", "winner"]
    logger.info("\n" + wilcoxon_df[[c for c in display_cols if c in wilcoxon_df.columns]].to_string(index=False))

    wilcoxon_df.to_csv(out_dir / "wilcoxon_tests.csv", index=False)

    # Text report
    report = build_report(wilcoxon_df, args.alpha)
    report_path = out_dir / "wilcoxon_report.txt"
    report_path.write_text(report)
    logger.info(f"\nFull report: {report_path}")

    # JSON summary
    summary = {
        "test":      "Wilcoxon signed-rank (two-sided)",
        "alpha":     args.alpha,
        "n_systems": len(preds),
        "systems":   list(preds.keys()),
        "results":   wilcoxon_df.to_dict(orient="records"),
    }
    (out_dir / "statistical_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(f"\nStatistical analysis complete. Results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
