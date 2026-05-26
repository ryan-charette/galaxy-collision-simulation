"""Evaluate a saved ML model artifact against a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from python.ml.datasets import (
    completed_rows,
    finite_target_rows,
    load_dataset,
    load_model_bundle,
    numeric_matrix,
)
from python.ml.evaluation import (
    baseline_metrics,
    markdown_report,
    model_beats_baseline,
    regression_metrics,
    solver_selection_accuracy,
)
from python.ml.features import predict_bundle


def evaluate(args: argparse.Namespace) -> None:
    bundle = load_model_bundle(args.model)
    targets = list(bundle["targets"])
    frame = finite_target_rows(completed_rows(load_dataset(args.data)), targets)
    if frame.empty:
        raise ValueError("No completed rows with finite targets are available for evaluation")

    y_true = numeric_matrix(frame, targets)
    predictions = predict_bundle(bundle, frame)
    baseline_predictions = bundle["mean_baseline"].predict(len(frame))
    metrics = regression_metrics(y_true, predictions, targets)
    mean_metrics = baseline_metrics(y_true, baseline_predictions, targets)
    beats = model_beats_baseline(metrics, mean_metrics)
    selection_accuracy = solver_selection_accuracy(
        frame,
        predictions,
        targets,
        "median_step_time",
    )

    metadata = {
        "model_path": str(args.model),
        "dataset_path": str(args.data),
        "model_kind": bundle.get("model_kind", ""),
        "model_type": bundle.get("model_type", ""),
        "rows": len(frame),
        "targets": ", ".join(targets),
    }
    report = markdown_report(
        "Model Evaluation",
        metadata,
        metrics,
        mean_metrics,
        beats,
        selection_accuracy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
