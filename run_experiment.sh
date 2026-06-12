#!/usr/bin/env bash
# run_experiment.sh — Pipeline experimental completo e reproduzível
#
# Executa todos os cenários do protocolo experimental:
#   Cenário A — Baselines individuais (BERT, RoBERTa, HateBERT)
#   Cenário B — Ensemble (Hard Voting + Soft Voting)
#   Cenário C — Few-Shot Augmentation (pipeline 7 etapas) + re-treino + ensemble
#
# Usage:
#   bash run_experiment.sh              # pipeline completo
#   bash run_experiment.sh --skip-aug   # pula augmentação (usa train_augmented.csv existente)
#   bash run_experiment.sh --scenario A # executa apenas um cenário
#
# Flags:
#   --skip-aug       Pula a geração de dados sintéticos (útil se já existe train_augmented.csv)
#   --scenario [A|B|C|stats|tables]  Executa somente o cenário especificado

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
SEED=42
EPOCHS=100
BATCH_SIZE=16
LR=1e-5
DATA_DIR="data"
RESULTS_A="results/scenario_A"
RESULTS_B="results/scenario_B"
RESULTS_C="results/scenario_C"
MODELS_A="models/scenario_A"
MODELS_C="models/scenario_C"
SKIP_AUG=false
SCENARIO="all"

# ---------------------------------------------------------------------------
# Parse de argumentos
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-aug)   SKIP_AUG=true;      shift ;;
    --scenario)   SCENARIO="$2";      shift 2 ;;
    *)            echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "========================================================"
echo " Hate Speech BERT Ensemble + Few-Shot Experiment"
echo " Dataset : Davidson et al. (2017)"
echo " Seed    : $SEED"
echo " Scenario: $SCENARIO"
echo "========================================================"

# ---------------------------------------------------------------------------
# Sync dependencies
# ---------------------------------------------------------------------------
uv sync

# ---------------------------------------------------------------------------
# Cenário A — Baselines individuais
# ---------------------------------------------------------------------------
run_scenario_a() {
  echo ""
  echo "----------------------------------------------------"
  echo " Cenário A — Baseline (individual models)"
  echo "----------------------------------------------------"
  mkdir -p "$RESULTS_A"

  uv run python train.py \
    --model all \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_A" \
    --models_dir "$MODELS_A"

  uv run python evaluate.py \
    --model all \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_A" \
    --models_dir "$MODELS_A" \
    --batch_size 64 \
    --seed "$SEED"
}

# ---------------------------------------------------------------------------
# Cenário B — Ensemble (sem augmentação)
# ---------------------------------------------------------------------------
run_scenario_b() {
  echo ""
  echo "----------------------------------------------------"
  echo " Cenário B — Ensemble (Hard + Soft Voting)"
  echo "----------------------------------------------------"
  mkdir -p "$RESULTS_B"

  uv run python ensemble.py \
    --strategy both \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_B" \
    --models_dir "$MODELS_A" \
    --batch_size 64 \
    --seed "$SEED"
}

# ---------------------------------------------------------------------------
# Cenário C — Few-Shot Augmentation + re-treino + ensemble
# ---------------------------------------------------------------------------
run_scenario_c() {
  echo ""
  echo "----------------------------------------------------"
  echo " Cenário C — Few-Shot Augmentation"
  echo "----------------------------------------------------"
  mkdir -p "$RESULTS_C"

  # Etapa C.0 — Geração de dados sintéticos
  if [ "$SKIP_AUG" = false ]; then
    echo " [C.0] Running 7-step augmentation pipeline (equalising classes to majority) …"
    uv run python -m augment.generator \
      --balance majority \
      --k 6 \
      --threshold 0.70 \
      --seed "$SEED" \
      --data_dir "$DATA_DIR" \
      --output "$DATA_DIR/train_augmented.csv"
  else
    echo " [C.0] Skipping augmentation (--skip-aug). Using existing train_augmented.csv"
    if [ ! -f "$DATA_DIR/train_augmented.csv" ]; then
      echo "ERROR: $DATA_DIR/train_augmented.csv not found. Remove --skip-aug or run augmentation first."
      exit 1
    fi
  fi

  # Etapa C.1 — Re-treino com dados augmentados
  echo " [C.1] Training models on augmented data …"
  uv run python train.py \
    --model all \
    --use_augmented \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_C" \
    --models_dir "$MODELS_C"

  # Etapa C.2 — Avaliação individual
  echo " [C.2] Evaluating augmented models …"
  uv run python evaluate.py \
    --model all \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_C" \
    --models_dir "$MODELS_C" \
    --batch_size 64 \
    --seed "$SEED"

  # Etapa C.3 — Ensemble com modelos augmentados
  echo " [C.3] Running ensemble on augmented models …"
  uv run python ensemble.py \
    --strategy both \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULTS_C" \
    --models_dir "$MODELS_C" \
    --batch_size 64 \
    --seed "$SEED"
}

# ---------------------------------------------------------------------------
# Análise estatística
# ---------------------------------------------------------------------------
run_stats() {
  echo ""
  echo "----------------------------------------------------"
  echo " Análise Estatística (McNemar + Bootstrap CI)"
  echo "----------------------------------------------------"

  # Consolidate predictions from all scenarios into one results dir for comparison
  mkdir -p results/all_predictions

  for f in "$RESULTS_A"/*_predictions.csv "$RESULTS_B"/*_predictions.csv "$RESULTS_C"/*_predictions.csv; do
    [ -f "$f" ] || continue
    scenario=$(basename "$(dirname "$f")")
    stem=$(basename "$f" .csv)
    cp "$f" "results/all_predictions/${scenario}_${stem}.csv"
  done

  uv run python statistical_tests.py \
    --results_dir results/all_predictions \
    --alpha 0.05 \
    --n_boot 1000
}

# ---------------------------------------------------------------------------
# Tabelas de resultados
# ---------------------------------------------------------------------------
run_tables() {
  echo ""
  echo "----------------------------------------------------"
  echo " Gerando Tabelas de Resultados (CSV + LaTeX)"
  echo "----------------------------------------------------"
  uv run python generate_tables.py \
    --results_dir results \
    --latex_bold
}

# ---------------------------------------------------------------------------
# Seleção de cenário
# ---------------------------------------------------------------------------
case "$SCENARIO" in
  "A")      run_scenario_a ;;
  "B")      run_scenario_b ;;
  "C")      run_scenario_c ;;
  "stats")  run_stats ;;
  "tables") run_tables ;;
  "all")
    run_scenario_a
    run_scenario_b
    run_scenario_c
    run_stats
    run_tables
    ;;
  *)
    echo "Unknown scenario: $SCENARIO. Use A, B, C, stats, tables, or all."
    exit 1
    ;;
esac

echo ""
echo "========================================================"
echo " Experiment complete!"
echo " Results saved to: results/"
echo "========================================================"
