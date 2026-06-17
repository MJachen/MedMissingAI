#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"

DATA_CONFIG="configs/data_server.yaml"
MODEL_CONFIG="configs/experiments/model_zero_fill_mask.yaml"
DROPOUT_TAGS=(005 010 015 020 030)
SEEDS=(42 43 44)
FORCE="${FORCE:-0}"
DEVICE="${DEVICE:-cuda:0}"

dropout_value() {
  case "$1" in
    005) echo "0.05" ;;
    010) echo "0.10" ;;
    015) echo "0.15" ;;
    020) echo "0.20" ;;
    030) echo "0.30" ;;
    *)
      echo "Unknown dropout tag: $1" >&2
      exit 2
      ;;
  esac
}

ensure_train_config() {
  local tag="$1"
  local seed="$2"
  local train_config="$3"
  local run_dir="$4"
  local prob

  prob="$(dropout_value "${tag}")"
  mkdir -p "$(dirname "${train_config}")"
  cat > "${train_config}" <<YAML
seed: ${seed}

training:
  save_dir: ${run_dir}
  epochs: 100
  batch_size: 2
  num_workers: 4
  lr: 0.0001
  weight_decay: 0.0001
  checkpoint_metric: balanced_accuracy
  class_weights: balanced
  modality_dropout_prob: ${prob}
  device: ${DEVICE}
YAML
}

for tag in "${DROPOUT_TAGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    train_config="configs/experiments/train_server_dropout${tag}_seed${seed}.yaml"
    run_dir="outputs/server_dropout${tag}_seed${seed}"
    ensure_train_config "${tag}" "${seed}" "${train_config}" "${run_dir}"

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

if [[ -f "scripts/summarize_experiment_matrix.py" ]]; then
  python scripts/summarize_experiment_matrix.py
else
  echo "Skipping matrix summary; scripts/summarize_experiment_matrix.py is not present"
fi
