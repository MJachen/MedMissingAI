from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


RUN_RE = re.compile(r"^server_(?P<method>.+)_seed(?P<seed>\d+)$")
METRICS = ("auc", "balanced_accuracy", "sensitivity", "specificity", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize server experiment matrix outputs.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--report", default="outputs/analysis_report_latest.md")
    parser.add_argument("--summary-csv", default="outputs/experiment_matrix_summary.csv")
    parser.add_argument(
        "--availability-csv",
        default="outputs/experiment_matrix_by_availability.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    run_rows = collect_run_rows(outputs_dir)
    if not run_rows:
        raise SystemExit(f"No server experiment outputs found under {outputs_dir}")

    summary_rows = summarize_runs(run_rows)
    write_csv(Path(args.summary_csv), summary_rows)

    availability_rows = collect_availability_rows(outputs_dir)
    availability_summary = summarize_availability(availability_rows)
    write_csv(Path(args.availability_csv), availability_summary)

    write_report(
        report_path=Path(args.report),
        summary_rows=summary_rows,
        availability_summary=availability_summary,
        run_rows=run_rows,
    )


def collect_run_rows(outputs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        table_path = run_dir / "test_main_table.csv"
        if not table_path.exists():
            continue
        for row in read_csv(table_path):
            rows.append(
                {
                    "run": run_dir.name,
                    "method": match.group("method"),
                    "seed": int(match.group("seed")),
                    "threshold_strategy": row["threshold_strategy"],
                    "threshold": as_float(row.get("threshold")),
                    **{metric: as_float(row.get(metric)) for metric in METRICS},
                }
            )
    return rows


def summarize_runs(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["threshold_strategy"]))].append(row)

    summary: list[dict[str, object]] = []
    for (method, strategy), group in sorted(grouped.items()):
        output: dict[str, object] = {
            "method": method,
            "threshold_strategy": strategy,
            "n_seeds": len(group),
            "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            output[f"{metric}_mean"] = mean(values) if values else None
            output[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(output)
    return summary


def collect_availability_rows(outputs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        table_path = run_dir / "test_metrics_by_availability.csv"
        if not table_path.exists():
            continue
        for row in read_csv(table_path):
            rows.append(
                {
                    "run": run_dir.name,
                    "method": match.group("method"),
                    "seed": int(match.group("seed")),
                    "availability": row["availability"],
                    "n": as_float(row.get("n")),
                    "auc": as_float(row.get("roc_auc")),
                    "balanced_accuracy": as_float(row.get("balanced_accuracy")),
                    "sensitivity": as_float(row.get("sensitivity")),
                    "specificity": as_float(row.get("specificity")),
                    "macro_f1": as_float(row.get("macro_f1")),
                }
            )
    return rows


def summarize_availability(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["availability"]))].append(row)

    summary: list[dict[str, object]] = []
    for (method, availability), group in sorted(grouped.items()):
        output: dict[str, object] = {
            "method": method,
            "availability": availability,
            "n_seeds": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            output[f"{metric}_mean"] = mean(values) if values else None
            output[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(output)
    return summary


def write_report(
    report_path: Path,
    summary_rows: list[dict[str, object]],
    availability_summary: list[dict[str, object]],
    run_rows: list[dict[str, object]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    default_rows = [
        row for row in summary_rows if row["threshold_strategy"] == "default_0_5"
    ]
    calibrated_rows = [
        row for row in summary_rows if row["threshold_strategy"] == "validation_calibrated"
    ]
    best_default = max(default_rows, key=lambda row: float(row["balanced_accuracy_mean"]))
    best_calibrated = max(
        calibrated_rows,
        key=lambda row: float(row["balanced_accuracy_mean"]),
    )

    lines = [
        "# MedMissingAI Latest Result Analysis",
        "",
        "This report was generated from the current local `outputs/server_*_seed*/` result folders.",
        "It supersedes older analysis that predated synced test metrics and prediction CSV files.",
        "",
        "## Current Evidence",
        "",
        f"- Experiment runs found: {len({row['run'] for row in run_rows})}.",
        "- Compared methods: balanced baseline, modality dropout, and learnable missing token.",
        "- Main metrics: AUC, balanced accuracy, sensitivity, specificity, and macro-F1.",
        "",
        "## Mean/Std by Method",
        "",
        "### Default threshold 0.5",
        "",
        markdown_summary_table(default_rows),
        "",
        "### Validation-calibrated threshold",
        "",
        markdown_summary_table(calibrated_rows),
        "",
        "## Practical Interpretation",
        "",
        f"- Best default-threshold mean balanced accuracy: `{best_default['method']}` "
        f"({float(best_default['balanced_accuracy_mean']):.4f}).",
        f"- Best validation-calibrated mean balanced accuracy: `{best_calibrated['method']}` "
        f"({float(best_calibrated['balanced_accuracy_mean']):.4f}).",
        "- Modality dropout is promising under the default threshold, but the calibrated-threshold view still favors the balanced baseline.",
        "- Learnable missing token is not stable enough to become the main direction yet.",
        "",
        "## Hard Availability Groups",
        "",
        markdown_hard_groups(availability_summary),
        "",
        "## Next Actions",
        "",
        "1. Keep the balanced zero-fill-plus-mask model as the reference baseline.",
        "2. Run a dropout grid over probabilities 0.05, 0.10, 0.15, 0.20, and 0.30 with seeds 42, 43, and 44.",
        "3. Use validation-calibrated threshold as the main reporting view, with default 0.5 as an auxiliary table.",
        "4. Do not promote learnable missing token until it is retested with dropout or a stronger fusion design.",
        "5. After the dropout grid, implement a mask-aware late-fusion model only if dropout does not give a stable gain.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def markdown_summary_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| method | n | AUC | BAC | Sens | Spec | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: str(item["method"])):
        lines.append(
            "| {method} | {n} | {auc} | {bac} | {sens} | {spec} | {f1} |".format(
                method=row["method"],
                n=row["n_seeds"],
                auc=fmt_mean_std(row, "auc"),
                bac=fmt_mean_std(row, "balanced_accuracy"),
                sens=fmt_mean_std(row, "sensitivity"),
                spec=fmt_mean_std(row, "specificity"),
                f1=fmt_mean_std(row, "macro_f1"),
            )
        )
    return "\n".join(lines)


def markdown_hard_groups(rows: list[dict[str, object]]) -> str:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    lines: list[str] = []
    for method in sorted(by_method):
        hardest = sorted(
            by_method[method],
            key=lambda row: float(row["balanced_accuracy_mean"]),
        )[:5]
        details = ", ".join(
            f"{row['availability']}={float(row['balanced_accuracy_mean']):.3f}"
            for row in hardest
        )
        lines.append(f"- `{method}` weakest BAC groups: {details}.")
    return "\n".join(lines)


def fmt_mean_std(row: dict[str, object], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


if __name__ == "__main__":
    main()
