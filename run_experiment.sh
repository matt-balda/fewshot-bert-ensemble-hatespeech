#!/usr/bin/env bash
# run_experiment.sh — Full reproducible pipeline
# Usage: bash run_experiment.sh

set -euo pipefail

echo "========================================"
echo "Few-shot Learning for an Ensemble of BERT Models (BERT, RoBERTa, and HateBERT)"
echo "Dataset: hatespeech"
echo "========================================"

# Install/sync dependencies using uv
uv sync

# Train all three models
uv run python train.py \
    --epochs 150 \
    --batch_size 16 \
    --lr 2e-5 \
    --seed 42 \
    --data_dir data \
    --results_dir results \
    --models_dir models

# Evaluate all models and generate comparison table
uv run python evaluate.py \
    --data_dir data \
    --results_dir results \
    --models_dir models \
    --batch_size 64 \
    --seed 42

echo "========================================"
echo " Experiment complete. See results/"
echo "========================================"
