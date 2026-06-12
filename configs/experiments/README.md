# Server Experiment Matrix

Use the same command template for every run. Change only `--train-config`
and, for the missing-token ablation, `--model-config`.

All runs keep:

- `class_weights: balanced`
- `checkpoint_metric: balanced_accuracy`
- `data-config configs/data_server.yaml`
- 3 seeds: 42, 43, 44

Recommended order:

1. Balanced-loss + balanced-checkpoint baseline:
   - `configs/experiments/train_server_balanced_seed42.yaml`
   - `configs/experiments/train_server_balanced_seed43.yaml`
   - `configs/experiments/train_server_balanced_seed44.yaml`
   - model: `configs/experiments/model_zero_fill_mask.yaml`

2. Modality dropout ablation, only changing `modality_dropout_prob`:
   - `configs/experiments/train_server_dropout015_seed42.yaml`
   - `configs/experiments/train_server_dropout015_seed43.yaml`
   - `configs/experiments/train_server_dropout015_seed44.yaml`
   - model: `configs/experiments/model_zero_fill_mask.yaml`

3. Learnable missing-token ablation, only changing model missing-token use:
   - `configs/experiments/train_server_missing_token_seed42.yaml`
   - `configs/experiments/train_server_missing_token_seed43.yaml`
   - `configs/experiments/train_server_missing_token_seed44.yaml`
   - model: `configs/experiments/model_missing_token.yaml`

Primary comparison target:

- `*_main_table.csv`
- `*_metrics_by_availability.csv`
- `*_metrics_by_availability_calibrated.csv`

Train one run:

```bash
export PYTHONPATH=src

python scripts/train.py \
  --data-config configs/data_server.yaml \
  --train-config configs/experiments/train_server_balanced_seed42.yaml \
  --model-config configs/experiments/model_zero_fill_mask.yaml
```

Evaluate one run by replacing `RUN_DIR` with that config's `training.save_dir`:

```bash
python scripts/evaluate.py \
  --data-config configs/data_server.yaml \
  --train-config configs/experiments/train_server_balanced_seed42.yaml \
  --model-config configs/experiments/model_zero_fill_mask.yaml \
  --checkpoint RUN_DIR/best.pt \
  --split test

python scripts/export_predictions.py \
  --data-config configs/data_server.yaml \
  --train-config configs/experiments/train_server_balanced_seed42.yaml \
  --model-config configs/experiments/model_zero_fill_mask.yaml \
  --checkpoint RUN_DIR/best.pt \
  --split val

python scripts/export_predictions.py \
  --data-config configs/data_server.yaml \
  --train-config configs/experiments/train_server_balanced_seed42.yaml \
  --model-config configs/experiments/model_zero_fill_mask.yaml \
  --checkpoint RUN_DIR/best.pt \
  --split test

python scripts/analyze_predictions.py \
  --predictions RUN_DIR/predictions_test.csv \
  --calibration-predictions RUN_DIR/predictions_val.csv \
  --threshold-metric balanced_accuracy
```
