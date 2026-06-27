# Few-Shot Data Augmentation for Hate Speech Detection via BERT Ensemble

> **Research project**: Hate speech classification on Twitter data using few-shot synthetic augmentation and ensemble of Transformer models.

---

## Overview

This project investigates whether **synthetically generated training data** can help language models better detect hate speech, a notoriously hard problem due to severe **class imbalance**: hate speech posts make up only ~5% of real-world datasets, while offensive-but-not-hateful content dominates.

The core idea: use a small language model (LLM) to generate realistic examples of the *rare* class, then train and combine three BERT-family classifiers to see if the augmented data genuinely improves detection of hate speech.

---

## The Problem

The dataset used (**Davidson et al., 2017**, Hate Speech and Offensive Language) reflects a realistic distribution from Twitter:

| Class | ~Samples | % of Total |
|---|---|---|
| `offensive_language` | ~13,400 | 77% |
| `neither` | ~3,100 | 18% |
| `hate_speech` | ~1,000 | **5%** |

This **15× imbalance** between the majority and minority class means classifiers trained naively will almost entirely ignore hate speech, maximizing accuracy by predicting the easy majority.

---

## Experimental Design

Three experimental **scenarios** are compared:

### Scenario A: Individual Baselines
Three pre-trained Transformer models are fine-tuned on the **original (unaugmented)** dataset, independently:

| Model | Pre-training Focus |
|---|---|
| **BERT-Base** (`bert-base-uncased`) | General English (Wikipedia + Books) |
| **RoBERTa-Base** (`roberta-base`) | Stronger general training (160 GB text) |
| **HateBERT** (`GroNLP/hateBERT`) | Domain-adapted: 1.5M Reddit posts from offensive communities |

### Scenario B: Ensemble (No Augmentation)
The three Scenario A models are **combined** using two voting strategies, without any data augmentation:
- **Hard Voting**: majority class wins among the three models
- **Soft Voting**: average of the predicted probability distributions

### Scenario C: Few-Shot Augmentation + Ensemble
The **7-step augmentation pipeline** is run first to generate synthetic hate speech (and neutral) examples. The three models are then **re-trained** on the expanded dataset, and the ensemble is applied again.

---

## The 7-Step Augmentation Pipeline

This is the methodological core of the research. A small open-source LLM (`Qwen/Qwen2.5-1.5B-Instruct`) is used to generate synthetic hate speech examples via **few-shot prompting**, by giving the model real examples and asking it to produce variations.

```
Step 1: Semantic Clustering
      Real hate speech examples → grouped into 6 semantic clusters
      (racial attacks, xenophobia, homophobia, sexism, religious intolerance, other)

Step 2: Prompt Templates
      Each cluster gets a category-specific prompt that preserves
      the semantic intent and type of attack

Step 3: Stratified Sampling
      Real examples are sampled proportionally from each cluster
      to serve as few-shot seeds for the LLM

Step 4: Few-Shot Generation (LLM)
      LLM is prompted with 5 real examples and asked to generate 10 new ones

Step 5: Linguistic Diversity
      20 different style combinations are cycled (tone, length,
      formality, syntax) to avoid repetitive outputs

Step 6: Automatic Filtering
      Generated texts are cleaned and rejected if they contain:
      LLM refusals / apologies / disclaimers, exact duplicates,
      or texts outside Twitter's length bounds (5–280 chars)

Step 7: Semantic Similarity Check
      Each generated text is accepted only if its cosine similarity
      to real examples in its cluster exceeds a threshold (0.45),
      preventing semantic drift
```

**Target**: Augment hate speech from ~1,000 to ~13,400 examples (matching the majority class). Both the `hate_speech` and `neither` classes are augmented.

**Fallback guarantee**: If the LLM pipeline falls short of the target, remaining slots are filled by oversampling real texts with minor word-level perturbation (drop/duplicate/swap one word), ensuring the exact target count is always reached.

---

## Training Protocol

All three models follow an **identical training protocol** to ensure fair comparison:

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for BERT fine-tuning |
| Learning rate | 1e-5 | Canonical LR for BERT-family models |
| Batch size | 16 | Balanced for GPU memory |
| Max epochs | 200 | Upper bound (early stopping kicks in earlier) |
| Early stopping | Patience = 12 epochs | Monitors validation macro F1 |
| Scheduler | Linear warmup (10%) + cosine decay | Prevents early LR spikes |
| Loss function | Weighted cross-entropy | Inverse-frequency class weights to counter imbalance |
| Gradient clipping | 1.0 | Training stability |
| Random seed | 42 | Full reproducibility |

**Regularization**: Embedding layer and first 3 encoder layers are frozen during training to reduce overfitting on a small fine-tuning dataset.

**Data split**: 70% train / 15% validation / 15% test, stratified to preserve class proportions.

---

## Evaluation Metrics

The evaluation focuses on **macro-averaged** metrics, which treat all classes equally regardless of size, making them appropriate for imbalanced classification:

- **F1 Macro**: primary ranking metric
- **Precision / Recall / F1 per class**: especially relevant for `hate_speech` (minority)
- **ROC-AUC Macro** (One-vs-Rest)
- **PR-AUC Macro** (Area under Precision-Recall curve)
- **Accuracy**: reported but not the primary metric (misleading under imbalance)

### Statistical Validation
All pairwise model comparisons are validated using the **Wilcoxon Signed-Rank Test** (two-sided, α = 0.05) on per-sample correctness, a non-parametric test appropriate for this setting.

---

## Project Structure

```
fewshot-bert-ensemble-hatespeech/
│
├── data/                          # Dataset files
│   ├── davidson_raw.csv           # Full Davidson et al. dataset (~24,000 tweets)
│   ├── train_augmented.csv        # Augmented training set (real + synthetic)
│   └── test.csv                   # Held-out test set (never seen during training)
│
├── augment/                       # 7-step augmentation pipeline
│   ├── generator.py               # Main pipeline orchestrator (Steps 1–7)
│   ├── cluster.py                 # Step 1 & 3: Semantic clustering (K-Means)
│   ├── templates.py               # Step 2: Category-specific LLM prompt templates
│   ├── filter.py                  # Step 6: LLM artefact removal & deduplication
│   └── similarity.py              # Step 7: Cosine similarity filter (SemanticFilter)
│
├── train.py                       # Model fine-tuning (Scenarios A and C)
├── evaluate.py                    # Inference + full metric computation
├── ensemble.py                    # Voting strategies (Scenarios B and C)
├── statistical_tests.py           # Pairwise Wilcoxon signed-rank tests
├── generate_tables.py             # Results tables (CSV + LaTeX)
├── generate_article_plots.py      # Publication-quality figures (ROC, PR curves)
│
├── data_loader.py                 # Dataset loading, preprocessing, PyTorch Dataset
├── utils.py                       # Metrics, logging, visualization utilities
│
├── run_experiment.sh              # Full reproducible pipeline (all scenarios)
└── pyproject.toml                 # Dependencies
```

---

## Reproducing the Experiment

**Requirements**: Python ≥ 3.12, CUDA-compatible GPU recommended (CPU possible but slow).

### Full pipeline (all scenarios)
```bash
# Install dependencies
uv sync

# Run all scenarios end-to-end
bash run_experiment.sh
```

### Run individual scenarios
```bash
# Scenario A: Train and evaluate baselines
bash run_experiment.sh --scenario A

# Scenario B: Ensemble (requires Scenario A models)
bash run_experiment.sh --scenario B

# Scenario C: Augmentation + retrain + ensemble
bash run_experiment.sh --scenario C

# Statistical tests
bash run_experiment.sh --scenario stats

# Generate result tables (CSV + LaTeX)
bash run_experiment.sh --scenario tables
```

### Skip augmentation (use existing data)
```bash
bash run_experiment.sh --skip-aug
```

### Generate publication figures
```bash
python generate_article_plots.py
# Output: results/article_plots/
```

---

## Key Dependencies

| Package | Role |
|---|---|
| `transformers` | BERT, RoBERTa, HateBERT, and Qwen2.5 model loading |
| `torch` | Training and inference |
| `sentence-transformers` | Semantic embeddings for clustering and similarity filtering |
| `scikit-learn` | Metrics, K-Means clustering, data splitting |
| `scipy` | Wilcoxon statistical test |
| `datasets` | HuggingFace dataset loading |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualizations |

---

## Dataset Reference

> T. Davidson, D. Warmsley, M. Macy, and I. Weber (2017). **"Automated Hate Speech Detection and the Problem of Offensive Language."** *Proceedings of the 11th AAAI International Conference on Web and Social Media (ICWSM).*

Available at: [HuggingFace Hub: tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)

---

## Models Used

| Model | HuggingFace ID | Parameters |
|---|---|---|
| BERT-Base | `bert-base-uncased` | ~110M |
| RoBERTa-Base | `roberta-base` | ~125M |
| HateBERT | `GroNLP/hateBERT` | ~110M |
| Generator LLM | `Qwen/Qwen2.5-1.5B-Instruct` | ~1.5B |

> **HateBERT** (Caselli et al., 2021) is a domain-adapted BERT variant continued pre-trained on 1.5M Reddit posts from offensive and banned communities. It is the only model here pre-trained on hate-related content, making it the strongest individual baseline hypothesis.

---

## Reproducibility Notes

- All experiments use **seed = 42** fixed across Python, NumPy, PyTorch, and CUDA.
- The test set (`data/test.csv`) is split before any augmentation and **never used during training**.
- Synthetic data is flagged with `is_synthetic=True` in the augmented CSV, allowing controlled experiments with sample weighting.
- Logit adjustment at inference time is supported via `--logit_adjust` flag to correct for the prior shift introduced by augmentation.

---

## License

MIT License. See [`LICENSE`](LICENSE).