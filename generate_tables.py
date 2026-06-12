"""
generate_tables.py — Geração de Tabelas de Resultados para Publicação

Consolida todos os arquivos *_metrics.json dos diretórios de resultado em:
  • results/main_table.csv          — tabela completa em CSV
  • results/main_table.tex          — tabela LaTeX pronta para publicação
  • results/hate_speech_table.csv   — foco na classe minoritária (hate speech)
  • results/hate_speech_table.tex   — idem, em LaTeX
  • results/all_scenarios_comparison.png — gráfico agrupado de todos os cenários

O script varre automaticamente:
  • results/scenario_A/*_metrics.json  → Cenário A (baselines)
  • results/scenario_B/*_metrics.json  → Cenário B (ensemble)
  • results/scenario_C/*_metrics.json  → Cenário C (few-shot + aug)
  • results/*_metrics.json             → qualquer resultado avulso

Usage:
    python generate_tables.py
    python generate_tables.py --results_dir results --latex_bold
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils import (
    get_logger,
    CLASS_NAMES,
    plot_all_scenarios_bar,
)

logger = get_logger(__name__, "results/generate_tables.log")

# ---------------------------------------------------------------------------
# Scenario detection
# ---------------------------------------------------------------------------

# Maps (directory_name, file_stem_key) → (scenario_id, display_name, group)
SCENARIO_MAP: Dict[str, tuple] = {
    # Baselines — Cenário A
    "bert-base-uncased":            ("A1", "BERT-Base",         "A — Baseline"),
    "roberta-base":                 ("A2", "RoBERTa-Base",      "A — Baseline"),
    "GroNLP_hateBERT":              ("A3", "HateBERT",          "A — Baseline"),
    # Ensemble sem aug — Cenário B
    "ensemble_hard":                ("B1", "Hard Voting",       "B — Ensemble"),
    "ensemble_soft":                ("B2", "Soft Voting",       "B — Ensemble"),
    # Aug + re-treino — Cenário C (individuais)
    "bert-base-uncased_aug":        ("C1", "BERT-Base+Aug",     "C — Few-Shot+Aug"),
    "roberta-base_aug":             ("C2", "RoBERTa-Base+Aug",  "C — Few-Shot+Aug"),
    "GroNLP_hateBERT_aug":          ("C3", "HateBERT+Aug",      "C — Few-Shot+Aug"),
    # Aug + ensemble — Cenário C (ensemble)
    "ensemble_hard_aug":            ("C4", "Hard Voting+Aug",   "C — Few-Shot+Ensemble"),
    "ensemble_soft_aug":            ("C5", "Soft Voting+Aug",   "C — Few-Shot+Ensemble"),
}

# Fallback: detect scenario from parent directory name
_DIR_TO_GROUP = {
    "scenario_a": "A — Baseline",
    "scenario_b": "B — Ensemble",
    "scenario_c": "C — Few-Shot+Aug",
}


def _stem_to_key(stem: str) -> str:
    """Normalise a file stem to a SCENARIO_MAP lookup key."""
    key = stem.replace("_metrics", "").strip("_")
    # Collapse common variations
    key = re.sub(r"_(predictions|metrics|report)$", "", key)
    return key


def detect_scenario(stem: str, parent_dir: str) -> tuple:
    """
    Return (scenario_id, display_name, group) for a given metrics file.
    Falls back gracefully when no explicit mapping is found.
    """
    key = _stem_to_key(stem)
    if key in SCENARIO_MAP:
        return SCENARIO_MAP[key]

    # Try fuzzy match
    for map_key, meta in SCENARIO_MAP.items():
        if map_key in key:
            return meta

    # Derive from parent directory
    group = _DIR_TO_GROUP.get(parent_dir.lower(), "Unknown")
    display = key.replace("_", " ").title()
    return ("??", display, group)


# ---------------------------------------------------------------------------
# Metric column definitions
# ---------------------------------------------------------------------------

MAIN_METRICS: List[tuple] = [
    ("accuracy",            "Accuracy"),
    ("precision_macro",     "Prec. Macro"),
    ("recall_macro",        "Rec. Macro"),
    ("f1_macro",            "F1 Macro"),
    ("precision_weighted",  "Prec. Weighted"),
    ("recall_weighted",     "Rec. Weighted"),
    ("f1_weighted",         "F1 Weighted"),
    ("roc_auc_macro",       "ROC-AUC"),
    ("pr_auc_macro",        "PR-AUC"),
]

HATE_METRICS: List[tuple] = [
    ("precision_hate_speech", "Prec. (HS)"),
    ("recall_hate_speech",    "Rec. (HS)"),
    ("f1_hate_speech",        "F1 (HS)"),
]

# Metrics to include in the multi-scenario grouped bar chart
CHART_METRICS = [
    "f1_macro", "f1_weighted", "accuracy",
    "f1_hate_speech", "roc_auc_macro",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _scan_metrics_files(root: Path) -> List[Path]:
    """
    Return all *_metrics.json files under root, including in scenario_A/B/C
    subdirectories. Excludes aggregate files like all_metrics.json.
    """
    files: List[Path] = []
    for f in sorted(root.rglob("*_metrics.json")):
        if f.name.startswith("all_"):
            continue
        if "statistical" in f.name:
            continue
        files.append(f)
    return files


def load_all_metrics(results_dir: str) -> pd.DataFrame:
    """
    Walk the results directory tree, load every individual *_metrics.json
    and build a consolidated DataFrame with scenario metadata columns.
    """
    root  = Path(results_dir)
    files = _scan_metrics_files(root)

    if not files:
        logger.warning(f"No *_metrics.json found under {results_dir}/")
        return pd.DataFrame()

    rows = []
    for f in files:
        parent_dir = f.parent.name   # e.g. "scenario_A", "results"
        stem       = f.stem

        try:
            m = json.loads(f.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {f}. Skipping.")
            continue

        scenario_id, display_name, group = detect_scenario(stem, parent_dir)

        row: Dict = {
            "Scenario":   scenario_id,
            "Method":     display_name,
            "Group":      group,
            "_source_dir": str(f.parent),
        }

        for metric_key, _ in MAIN_METRICS + HATE_METRICS:
            val = m.get(metric_key)
            row[metric_key] = float(val) if val is not None else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["Group", "Scenario"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rename metric columns to display labels and drop internal columns."""
    metric_keys = [mk for mk, _ in MAIN_METRICS if mk in df.columns]
    cols = ["Scenario", "Method", "Group"] + metric_keys
    sub  = df[[c for c in cols if c in df.columns]].copy()
    rename = {mk: dn for mk, dn in MAIN_METRICS if mk in sub.columns}
    return sub.rename(columns=rename)


def build_hate_speech_table(df: pd.DataFrame) -> pd.DataFrame:
    """Focus table: macro metrics + per-class hate speech metrics."""
    priority_keys = [mk for mk, _ in MAIN_METRICS[:4] if mk in df.columns]
    hs_keys       = [mk for mk, _ in HATE_METRICS    if mk in df.columns]
    cols = ["Scenario", "Method", "Group"] + priority_keys + hs_keys
    sub  = df[[c for c in cols if c in df.columns]].copy()
    rename = {mk: dn for mk, dn in MAIN_METRICS + HATE_METRICS if mk in sub.columns}
    return sub.rename(columns=rename)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

def _fmt(val) -> str:
    """Format a single float value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val):.4f}"


def _bold_max(series: pd.Series) -> List[str]:
    """Return list of formatted strings; best value is bold."""
    numeric = pd.to_numeric(series, errors="coerce")
    max_val = numeric.max()
    result  = []
    for v in numeric:
        if np.isnan(v):
            result.append("—")
        elif v == max_val:
            result.append(f"\\textbf{{{v:.4f}}}")
        else:
            result.append(f"{v:.4f}")
    return result


def to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    bold_best: bool = True,
) -> str:
    """
    Convert a results DataFrame to a publication-quality LaTeX booktabs table.

    Groups rows with \\midrule separators whenever the 'Group' column changes.
    Requires: \\usepackage{booktabs} in the LaTeX preamble.
    """
    # Identify float columns (all except the first 3 identifier columns)
    id_cols    = ["Scenario", "Method", "Group"]
    float_cols = [c for c in df.columns if c not in id_cols]

    col_fmt = "l" * len(id_cols) + "c" * len(float_cols)
    all_cols = id_cols + float_cols

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{c}}}" for c in all_cols) + r" \\",
        r"\midrule",
    ]

    # Format float columns
    formatted = df.copy()
    for col in float_cols:
        if col in formatted.columns:
            if bold_best:
                formatted[col] = _bold_max(pd.to_numeric(formatted[col], errors="coerce"))
            else:
                formatted[col] = pd.to_numeric(formatted[col], errors="coerce").apply(_fmt)

    prev_group = None
    for _, row in formatted.iterrows():
        curr_group = row.get("Group", "")
        if curr_group != prev_group and prev_group is not None:
            lines.append(r"\midrule")
        prev_group = curr_group

        cells = []
        for c in all_cols:
            cells.append(str(row[c]) if c in row.index else "—")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication-ready result tables")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Root results directory (will scan all subdirs).")
    p.add_argument("--latex_bold",  action="store_true",
                   help="Bold the best value in each column of the LaTeX tables.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  GENERATING RESULT TABLES")
    logger.info("=" * 60)

    df = load_all_metrics(args.results_dir)
    if df.empty:
        logger.error(
            "No metrics found. Run evaluate.py and ensemble.py first, then re-run."
        )
        return

    logger.info(f"  Found {len(df)} experiment entries across all scenarios.\n")

    # ── Main table ──────────────────────────────────────────────────────────
    main_tbl = build_main_table(df)
    main_tbl.to_csv(out_dir / "main_table.csv", index=False)

    latex_main = to_latex(
        main_tbl,
        caption=(
            "Comparison of all experimental scenarios on the Davidson Hate Speech "
            "and Offensive Language dataset (Davidson et al., 2017). "
            "HS = Hate Speech class. Best result per column in \\textbf{bold}."
        ),
        label="tab:main_results",
        bold_best=args.latex_bold,
    )
    (out_dir / "main_table.tex").write_text(latex_main)

    logger.info("=== MAIN RESULTS TABLE ===")
    logger.info("\n" + main_tbl.to_string(index=False))

    # ── Hate Speech (minority class) table ───────────────────────────────────
    hs_tbl = build_hate_speech_table(df)
    hs_tbl.to_csv(out_dir / "hate_speech_table.csv", index=False)

    latex_hs = to_latex(
        hs_tbl,
        caption=(
            "Per-class results for the \\emph{hate speech} (minority) class across "
            "all experimental scenarios. Best result per column in \\textbf{bold}."
        ),
        label="tab:hatespeech_results",
        bold_best=args.latex_bold,
    )
    (out_dir / "hate_speech_table.tex").write_text(latex_hs)

    logger.info("\n=== HATE SPEECH TABLE ===")
    logger.info("\n" + hs_tbl.to_string(index=False))

    # ── Grouped bar chart ────────────────────────────────────────────────────
    # Build {display_name: {metric: value}} for the chart
    chart_data: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = f"{row['Scenario']} {row['Method']}"
        chart_data[name] = {
            mk: row.get(mk, np.nan) for mk in CHART_METRICS
        }

    chart_metrics_present = [m for m in CHART_METRICS if m in df.columns]
    if chart_data and chart_metrics_present:
        plot_all_scenarios_bar(
            scenarios=chart_data,
            metrics=chart_metrics_present,
            save_path=str(out_dir / "all_scenarios_comparison.png"),
        )
        logger.info(f"\nGrouped chart saved: {out_dir / 'all_scenarios_comparison.png'}")

    logger.info(f"\nAll tables saved to {args.results_dir}/")
    logger.info("  main_table.csv / main_table.tex")
    logger.info("  hate_speech_table.csv / hate_speech_table.tex")


if __name__ == "__main__":
    main()
