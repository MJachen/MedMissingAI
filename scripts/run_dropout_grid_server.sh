#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"

DATA_CONFIG="configs/data_server.yaml"
MODEL_CONFIG="configs/experiments/model_zero_fill_mask.yaml"
DROPOUT_TAGS=(005 010 015 020 030)
SEEDS=(42 43 44)
FORCE="${FORCE:-0}"

for tag in "${DROPOUT_TAGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    train_config="configs/experiments/train_server_dropout${tag}_seed${seed}.yaml"
    run_dir="outputs/server_dropout${tag}_seed${seed}"

    if [[ ! -f "${run_dir}/best.pt" || "${FORCE}" == "1" ]]; then
      echo "Training ${train_config}"
      python scripts/train.py \
        --data-config "${DATA_CONFIG}" \
        --train-config "${train_config}" \
        --model-config "${MODEL_CONFIG}"
    else
      echo "Skipping training for ${run_dir}; best.pt already exists"
    fi

    echo "Evaluating ${run_dir}"
    python scripts/evaluate.py \
      --data-config "${DATA_CONFIG}" \
      --train-config "${train_config}" \
      --model-config "${MODEL_CONFIG}" \
      --checkpoint "${run_dir}/best.pt" \
      --split test

    python scripts/export_predictions.py \
      --data-config "${DATA_CONFIG}" \
      --train-config "${train_config}" \
      --model-config "${MODEL_CONFIG}" \
      --checkpoint "${run_dir}/best.pt" \
      --split val

    python scripts/export_predictions.py \
      --data-config "${DATA_CONFIG}" \
      --train-config "${train_config}" \
      --model-config "${MODEL_CONFIG}" \
      --checkpoint "${run_dir}/best.pt" \
      --split test

    python scripts/analyze_predictions.py \
      --predictions "${run_dir}/predictions_test.csv" \
      --calibration-predictions "${run_dir}/predictions_val.csv" \
      --threshold-metric balanced_accuracy
  done
done

python scripts/summarize_experiment_matrix.py
