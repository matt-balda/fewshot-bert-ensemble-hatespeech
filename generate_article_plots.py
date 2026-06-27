"""
generate_article_plots.py

Generates three publication-quality figures for the article:

  Plot 1 — ROC Multiclass OvR (3 panels: one per class)
  Plot 2 — Wilcoxon Significance Heatmap (10×10)
  Plot 3 — Precision-Recall Curves (focus: hate_speech class)

All figures saved to results/article_plots/
Usage:
    python generate_article_plots.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path("results/article_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR    = Path("results/all_predictions")

CLASS_NAMES = ["hate_speech", "offensive_language", "neither"]
CLASS_LABELS = {
    "hate_speech":          "Hate Speech",
    "offensive_language":   "Offensive Language",
    "neither":              "Neither",
}

DPI = 300

# ── System registry ──────────────────────────────────────────────────────────
SYSTEMS = {
    # (file_stem, display_name, scenario_group, color, linestyle, marker)
    "A-BERT":     ("scenario_A_bert-base-uncased",  "A · BERT-Base",      "A",
                   "#2196F3", "-",  "o"),
    "A-RoBERTa":  ("scenario_A_roberta-base",        "A · RoBERTa-Base",   "A",
                   "#03A9F4", "--", "s"),
    "A-HateBERT": ("scenario_A_GroNLP_hateBERT",     "A · HateBERT",       "A",
                   "#0D47A1", ":",  "^"),
    "B-Hard":     ("scenario_B_ensemble_hard",       "B · Hard Voting",    "B",
                   "#FF5722", "-",  "D"),
    "B-Soft":     ("scenario_B_ensemble_soft",       "B · Soft Voting",    "B",
                   "#FF9800", "--", "P"),
    "C-BERT":     ("scenario_C_bert-base-uncased",   "C · BERT+Aug",       "C",
                   "#4CAF50", "-",  "o"),
    "C-RoBERTa":  ("scenario_C_roberta-base",        "C · RoBERTa+Aug",    "C",
                   "#8BC34A", "--", "s"),
    "C-HateBERT": ("scenario_C_GroNLP_hateBERT",     "C · HateBERT+Aug",   "C",
                   "#2E7D32", ":",  "^"),
    "C-Hard":     ("scenario_C_ensemble_hard",       "C · Hard+Aug",       "C",
                   "#795548", "-.", "D"),
    "C-Soft":     ("scenario_C_ensemble_soft",       "C · Soft+Aug",       "C",
                   "#9C27B0", (0, (3,1,1,1)), "P"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_all_predictions():
    """Load prediction DataFrames for all 10 systems."""
    data = {}
    for key, (stem, _, _, _, _, _) in SYSTEMS.items():
        path = PRED_DIR / f"{stem}_predictions.csv"
        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        # For hard voting, prob columns may be one-hot — keep as-is
        data[key] = df
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — ROC Multiclass OvR
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_multiclass(data: dict, save_path: Path) -> None:
    print("Generating Plot 1: ROC Multiclass OvR …")

    # Use global matplotlib style
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.alpha":         0.4,
        "grid.linewidth":     0.6,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, cls_idx, cls_key in zip(axes, range(3), CLASS_NAMES):
        ax.set_facecolor("#FAFAFA")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")

        for key, df in data.items():
            _, display, _, color, ls, marker = SYSTEMS[key]
            y_true = df["true_label"].values
            y_bin  = (y_true == cls_idx).astype(int)

            prob_col = f"prob_{cls_key}"
            if prob_col not in df.columns:
                continue
            scores = df[prob_col].values

            is_hard_voting = np.all((scores == 0) | (scores == 1))

            if is_hard_voting:
                # Hard voting has no soft probabilities — plot as a single
                # operating-point marker at the (FPR, TPR) of the hard decision.
                from sklearn.metrics import confusion_matrix
                y_pred_bin = scores.astype(int)
                tn = np.sum((y_bin == 0) & (y_pred_bin == 0))
                fp = np.sum((y_bin == 0) & (y_pred_bin == 1))
                fn = np.sum((y_bin == 1) & (y_pred_bin == 0))
                tp = np.sum((y_bin == 1) & (y_pred_bin == 1))
                fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tpr_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                ax.plot(fpr_pt, tpr_pt,
                        marker="*", markersize=12, color=color,
                        linestyle="none", markeredgecolor="white",
                        markeredgewidth=0.6,
                        label=f"{display}  (op. point)", alpha=0.95, zorder=5)
            else:
                fpr, tpr, _ = roc_curve(y_bin, scores)
                roc_auc     = auc(fpr, tpr)

                lw = 1.8 if key.startswith("B") else 1.3
                ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                        label=f"{display}  (AUC={roc_auc:.3f})", alpha=0.88)

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(
            f"{CLASS_LABELS[cls_key]}",
            fontsize=13, fontweight="bold", pad=8,
        )
        ax.tick_params(labelsize=9)

        leg = ax.legend(
            loc="lower right",
            fontsize=7.5,
            frameon=True,
            framealpha=0.85,
            edgecolor="#cccccc",
            ncol=1,
        )
        leg.get_frame().set_linewidth(0.5)

    # Scenario color-band legend at top
    scenario_patches = [
        mpatches.Patch(facecolor="#2196F3", alpha=0.7, label="Scenario A: Baseline"),
        mpatches.Patch(facecolor="#FF5722", alpha=0.7, label="Scenario B: Ensemble (no aug.)"),
        mpatches.Patch(facecolor="#4CAF50", alpha=0.7, label="Scenario C: Few-Shot Augmentation"),
    ]
    fig.legend(
        handles=scenario_patches,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.suptitle(
        "ROC Curves: One-vs-Rest per Class (All Scenarios)",
        fontsize=15, fontweight="bold", y=1.08,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path}")



# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Precision-Recall Curves (focus: hate_speech)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(data: dict, save_path: Path) -> None:
    print("Generating Plot 3: Precision-Recall Curves (hate_speech focus) …")

    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.alpha":         0.4,
        "grid.linewidth":     0.6,
    })

    # 2 panels: left = hate_speech detail, right = macro-average comparison
    fig, (ax_hs, ax_macro) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#FAFAFA")

    for ax in [ax_hs, ax_macro]:
        ax.set_facecolor("#FAFAFA")

    # ── Panel A: Hate Speech class ────────────────────────────────────────────
    hs_idx = 0  # hate_speech = class 0
    prevalence_hs = None

    pr_records = []  # for ranking by AP

    for key, df in data.items():
        _, display, _, color, ls, marker = SYSTEMS[key]
        y_true = df["true_label"].values
        y_bin  = (y_true == hs_idx).astype(int)

        if prevalence_hs is None:
            prevalence_hs = y_bin.mean()

        prob_col = "prob_hate_speech"
        if prob_col not in df.columns:
            continue
        scores = df[prob_col].values
        if np.all((scores == 0) | (scores == 1)):
            continue

        prec, rec, _ = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)
        pr_records.append((ap, key, display, color, ls, marker, prec, rec))

    # Sort by AP descending for cleaner overlap rendering
    pr_records.sort(key=lambda x: x[0], reverse=False)

    for ap, key, display, color, ls, marker, prec, rec in pr_records:
        is_best = key in ("B-Soft", "B-Hard")  # highlight ensemble
        lw = 2.2 if is_best else 1.4
        alpha = 0.95 if is_best else 0.75
        ax_hs.plot(rec, prec, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                   label=f"{display}  (AP={ap:.3f})")

    # Baseline
    ax_hs.axhline(prevalence_hs, color="#9CA3AF", linestyle=":", linewidth=1.2,
                  label=f"Random (prev={prevalence_hs:.3f})")

    ax_hs.set_xlim([-0.01, 1.01])
    ax_hs.set_ylim([-0.01, 1.01])
    ax_hs.set_xlabel("Recall", fontsize=12)
    ax_hs.set_ylabel("Precision", fontsize=12)
    ax_hs.set_title("Precision–Recall: Hate Speech Class\n(minority, ~5.8% prevalence)",
                    fontsize=13, fontweight="bold")

    leg = ax_hs.legend(loc="upper right", fontsize=8, frameon=True,
                       framealpha=0.9, edgecolor="#cccccc",
                       title="System  (AP)", title_fontsize=8.5)
    leg.get_frame().set_linewidth(0.5)

    # Annotate operating point (threshold=0.5)
    ax_hs.annotate(
        "← Threshold 0.5\noperating region",
        xy=(0.45, 0.35), fontsize=8, color="#6B7280",
        ha="center",
    )

    # ── Panel B: Macro PR (average across all 3 classes) ─────────────────────
    macro_records = []
    for key, df in data.items():
        _, display, _, color, ls, marker = SYSTEMS[key]
        y_true = df["true_label"].values
        n_cls  = 3
        y_bin  = label_binarize(y_true, classes=list(range(n_cls)))

        # Compute PR per class and average
        all_prec, all_rec, all_ap = [], [], []
        prob_cols = ["prob_hate_speech", "prob_offensive_language", "prob_neither"]
        has_probs = all(c in df.columns for c in prob_cols)
        if not has_probs:
            continue

        probs = df[prob_cols].values
        # Skip degenerate
        if np.all((probs == 0) | (probs == 1)):
            continue

        # Interpolated macro PR
        mean_rec = np.linspace(0, 1, 200)
        interp_prec_per_cls = []
        for c in range(n_cls):
            prec_c, rec_c, _ = precision_recall_curve(y_bin[:, c], probs[:, c])
            # Interpolate precision at common recall grid (flip for interp)
            interp_p = np.interp(mean_rec, np.flip(rec_c), np.flip(prec_c))
            interp_prec_per_cls.append(interp_p)
            all_ap.append(average_precision_score(y_bin[:, c], probs[:, c]))

        macro_prec = np.mean(interp_prec_per_cls, axis=0)
        macro_ap   = np.mean(all_ap)
        macro_records.append((macro_ap, key, display, color, ls, marker, mean_rec, macro_prec))

    macro_records.sort(key=lambda x: x[0], reverse=False)

    for macro_ap, key, display, color, ls, marker, mean_rec, macro_prec in macro_records:
        is_best = key in ("B-Soft", "B-Hard")
        lw = 2.2 if is_best else 1.4
        alpha = 0.95 if is_best else 0.75
        ax_macro.plot(mean_rec, macro_prec, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                      label=f"{display}  (mAP={macro_ap:.3f})")

    # Macro baseline = macro prevalence
    if prevalence_hs is not None:
        macro_prev = 1 / 3
        ax_macro.axhline(macro_prev, color="#9CA3AF", linestyle=":", linewidth=1.2,
                         label=f"Random (uniform prior={macro_prev:.2f})")

    ax_macro.set_xlim([-0.01, 1.01])
    ax_macro.set_ylim([-0.01, 1.01])
    ax_macro.set_xlabel("Recall (macro-averaged)", fontsize=12)
    ax_macro.set_ylabel("Precision (macro-averaged)", fontsize=12)
    ax_macro.set_title("Precision–Recall: Macro Average\n(interpolated across all 3 classes)",
                       fontsize=13, fontweight="bold")

    leg2 = ax_macro.legend(loc="upper right", fontsize=8, frameon=True,
                            framealpha=0.9, edgecolor="#cccccc",
                            title="System  (mAP)", title_fontsize=8.5)
    leg2.get_frame().set_linewidth(0.5)

    # ── Shared scenario color legend ──────────────────────────────────────────
    scenario_patches = [
        mpatches.Patch(facecolor="#2196F3", alpha=0.8, label="Scenario A: Baseline"),
        mpatches.Patch(facecolor="#FF5722", alpha=0.8, label="Scenario B: Ensemble"),
        mpatches.Patch(facecolor="#4CAF50", alpha=0.8, label="Scenario C: Few-Shot+Aug"),
    ]
    fig.legend(
        handles=scenario_patches,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.025),
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.suptitle(
        "Precision–Recall Analysis: All Scenarios",
        fontsize=15, fontweight="bold", y=1.07,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ARTICLE PLOTS — Publication Quality")
    print("=" * 60)

    data = load_all_predictions()
    print(f"\nLoaded {len(data)} systems: {list(data.keys())}\n")

    plot_roc_multiclass(data, OUT_DIR / "plot1_roc_multiclass.png")
    plot_pr_curves(data,  OUT_DIR / "plot3_pr_curves.png")

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
