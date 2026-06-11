from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medmissingai.training.metrics import binary_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction CSV metrics.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--calibration-predictions", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--by-availability-csv", default=None)
    parser.add_argument("--by-availability-calibrated-csv", default=None)
    parser.add_argument(
        "--threshold-metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "macro_f1", "f1", "sensitivity", "specificity"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    predictions = pd.read_csv(predictions_path)
    output_json = Path(args.output_json or _default_output(predictions_path, "_metrics_detailed.json"))
    by_availability_csv = Path(
        args.by_availability_csv or _default_output(predictions_path, "_metrics_by_availability.csv")
    )
    by_availability_calibrated_csv = Path(
        args.by_availability_calibrated_csv
        or _default_output(predictions_path, "_metrics_by_availability_calibrated.csv")
    )

    default_threshold = 0.5
    default_metrics = metrics_from_frame(predictions, default_threshold)
    payload: dict[str, Any] = {
        "predictions": str(predictions_path),
        "metrics_at_threshold_0_5": default_metrics,
        "calibration": None,
    }

    write_by_availability(predictions, default_threshold, by_availability_csv)

    if args.calibration_predictions is not None:
        calibration_path = Path(args.calibration_predictions)
        calibration = pd.read_csv(calibration_path)
        threshold, calibration_metrics = calibrate_threshold(calibration, args.threshold_metric)
        calibrated_metrics = metrics_from_frame(predictions, threshold)
        payload["calibration"] = {
            "predictions": str(calibration_path),
            "threshold_metric": args.threshold_metric,
            "selected_threshold": threshold,
            "calibration_metrics": calibration_metrics,
            "evaluation_metrics_at_calibrated_threshold": calibrated_metrics,
        }
        write_by_availability(predictions, threshold, by_availability_calibrated_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {by_availability_csv}")
    if args.calibration_predictions is not None:
        print(f"Wrote {by_availability_calibrated_csv}")


def metrics_from_frame(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    _require_columns(frame, {"label", "prob_class_1"})
    return binary_classification_metrics(
        labels=frame["label"].to_numpy(),
        positive_probs=frame["prob_class_1"].to_numpy(),
        threshold=threshold,
    )


def calibrate_threshold(frame: pd.DataFrame, metric_name: str) -> tuple[float, dict[str, Any]]:
    _require_columns(frame, {"label", "prob_class_1"})
    probs = np.sort(frame["prob_class_1"].astype(float).unique())
    candidates = {0.0, 0.5, 1.0}
    candidates.update(float(value) for value in probs)
    if len(probs) > 1:
        candidates.update(float((left + right) / 2.0) for left, right in zip(probs[:-1], probs[1:]))

    best_threshold = 0.5
    best_metrics = metrics_from_frame(frame, best_threshold)
    best_score = _score(best_metrics, metric_name)

    for threshold in sorted(candidates):
        metrics = metrics_from_frame(frame, threshold)
        score = _score(metrics, metric_name)
        if score > best_score or (
            score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return best_threshold, best_metrics


def write_by_availability(frame: pd.DataFrame, threshold: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "availability" not in frame.columns:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    rows = []
    for availability, group in frame.groupby("availability", dropna=False):
        metrics = metrics_from_frame(group, threshold)
        rows.append(
            {
                "availability": availability,
                "threshold": metrics["threshold"],
                "n": int(len(group)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "roc_auc": metrics.get("roc_auc"),
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )

    pd.DataFrame(rows).sort_values("availability").to_csv(output_path, index=False)


def _score(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "f1":
        return float(_positive_class_f1(metrics))
    value = metrics.get(metric_name)
    if value is None:
        return float("-inf")
    return float(value)


def _positive_class_f1(metrics: dict[str, Any]) -> float:
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _default_output(predictions_path: Path, suffix: str) -> str:
    stem = predictions_path.stem
    if stem.startswith("predictions_"):
        stem = stem.removeprefix("predictions_")
    return str(predictions_path.with_name(f"{stem}{suffix}"))


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")


if __name__ == "__main__":
    main()
