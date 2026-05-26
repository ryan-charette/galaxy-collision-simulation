"""Train supervised models for approximate-solver force error."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from python.ml import DATASET_SCHEMA_VERSION
from python.ml.datasets import (
    completed_rows,
    finite_target_rows,
    load_dataset,
    numeric_matrix,
    save_model_bundle,
    train_test_split_indices,
    write_json,
)
from python.ml.evaluation import baseline_metrics, markdown_report, model_beats_baseline, regression_metrics
from python.ml.features import (
    FeatureTransformer,
    MeanRegressor,
    StandardScaler,
    force_feature_columns,
    make_regressor,
)


DEFAULT_TARGETS = ["force_rmse", "relative_force_rmse"]


def train(args: argparse.Namespace) -> None:
    frame = finite_target_rows(completed_rows(load_dataset(args.data)), args.targets)
    if len(frame) < 2:
        raise ValueError("Need at least two completed rows with finite force-error targets")

    numeric_features, categorical_features = force_feature_columns(frame)
    transformer = FeatureTransformer(numeric_features, categorical_features).fit(frame)
    features = transformer.transform(frame)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    targets = numeric_matrix(frame, args.targets)

    split = train_test_split_indices(len(frame), args.test_fraction, args.seed)
    x_train, y_train = features[split.train_indices], targets[split.train_indices]
    x_test, y_test = features[split.test_indices], targets[split.test_indices]

    model = make_regressor(args.model, args.seed)
    model.fit(x_train, y_train)
    baseline = MeanRegressor().fit(y_train)

    predictions = model.predict(x_test)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    baseline_predictions = baseline.predict(len(x_test))
    metrics = regression_metrics(y_test, predictions, args.targets)
    mean_metrics = baseline_metrics(y_test, baseline_predictions, args.targets)
    beats = model_beats_baseline(metrics, mean_metrics)

    metadata = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_type": "force_error",
        "model_kind": "force_error",
        "model_type": args.model,
        "targets": args.targets,
        "feature_names": transformer.feature_names,
        "train_rows": int(len(split.train_indices)),
        "test_rows": int(len(split.test_indices)),
        "dataset_path": str(args.data),
        "trained_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    bundle = {
        **metadata,
        "feature_transformer": transformer,
        "feature_scaler": scaler,
        "model": model,
        "mean_baseline": baseline,
        "metrics": metrics,
        "mean_baseline_metrics": mean_metrics,
        "beats_mean_baseline": beats,
        "solver_selection_accuracy": None,
    }
    save_model_bundle(args.output, bundle)

    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.md")
    report = markdown_report(
        "Force Error Model Evaluation",
        metadata,
        metrics,
        mean_metrics,
        beats,
        None,
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report, encoding="utf-8")
    write_json(args.output.with_suffix(args.output.suffix + ".metadata.json"), metadata)
    print(f"Wrote {args.output}")
    print(f"Wrote {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--model",
        choices=["linear", "random_forest", "gradient_boosting"],
        default="linear",
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
