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

4. Dropout grid, only changing `modality_dropout_prob`:
   - `0.05`: `train_server_dropout005_seed42.yaml`, `train_server_dropout005_seed43.yaml`, `train_server_dropout005_seed44.yaml`
   - `0.10`: `train_server_dropout010_seed42.yaml`, `train_server_dropout010_seed43.yaml`, `train_server_dropout010_seed44.yaml`
   - `0.15`: `train_server_dropout015_seed42.yaml`, `train_server_dropout015_seed43.yaml`, `train_server_dropout015_seed44.yaml`
   - `0.20`: `train_server_dropout020_seed42.yaml`, `train_server_dropout020_seed43.yaml`, `train_server_dropout020_seed44.yaml`
   - `0.30`: `train_server_dropout030_seed42.yaml`, `train_server_dropout030_seed43.yaml`, `train_server_dropout030_seed44.yaml`
   - model: `configs/experiments/model_zero_fill_mask.yaml`

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

Summarize synced local results:

```bash
python scripts/summarize_experiment_matrix.py
```

Run the full dropout grid on the Linux server:

```bash
export PYTHONPATH=src
CUDA_VISIBLE_DEVICES=6 DEVICE=cuda:0 nohup bash scripts/run_dropout_grid_server.sh > outputs/dropout_grid.log 2>&1 &
tail -f outputs/dropout_grid.log
```

By default, the script writes the dropout train configs before running, so the
server only needs this script plus the base repo files. It also skips a run if
`RUN_DIR/best.pt` already exists. To rerun all dropout-grid training jobs from
scratch, use:

```bash
CUDA_VISIBLE_DEVICES=6 DEVICE=cuda:0 FORCE=1 bash scripts/run_dropout_grid_server.sh
```
