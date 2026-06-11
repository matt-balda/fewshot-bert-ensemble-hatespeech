#!/usr/bin/env bash
# run_experiment.sh — Full reproducible pipeline
# Usage: bash run_experiment.sh

set -euo pipefail

echo "========================================"
echo " Hate Speech Detection Benchmark"
echo " Davidson et al. (2017)"
echo "========================================"

# Install dependencies
pip install -r requirements.txt

# Train all three models
python train.py \
    --epochs 150 \
    --batch_size 16 \
    --lr 2e-5 \
    --seed 42 \
    --data_dir data \
    --results_dir results \
    --models_dir models

# Evaluate all models and generate comparison table
python evaluate.py \
    --data_dir data \
    --results_dir results \
    --models_dir models \
    --batch_size 64 \
    --seed 42

echo "========================================"
echo " Experiment complete. See results/"
echo "========================================"
