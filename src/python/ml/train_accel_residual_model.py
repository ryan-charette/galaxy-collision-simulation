"""Train an acceleration-residual model for approximate solver correction."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from python.ml import DATASET_SCHEMA_VERSION
from python.ml.datasets import load_dataset, numeric_matrix, save_model_bundle, write_json
from python.ml.features import make_regressor
from python.ml.residuals import (
    RESIDUAL_SCHEMA_VERSION,
    RESIDUAL_TARGETS,
    NumpyKnnRegressor,
    correction_markdown_report,
    correction_metrics,
    finite_residual_rows,
    metrics_dict,
    residual_feature_columns,
    split_by_config,
)
from python.ml.training import fit_feature_pipeline


def make_residual_regressor(args: argparse.Namespace) -> Any:
    if args.model == "knn":
        return NumpyKnnRegressor(k=args.knn_k)
    return make_regressor(args.model, args.seed)


def train(args: argparse.Namespace) -> None:
    frame = finite_residual_rows(load_dataset(args.data)).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Need at least two residual rows")

    numeric_features, categorical_features = residual_feature_columns(frame)
    transformer, scaler, features = fit_feature_pipeline(frame, numeric_features, categorical_features)
    targets = numeric_matrix(frame, RESIDUAL_TARGETS)

    train_indices, test_indices, train_groups, test_groups = split_by_config(
        frame,
        args.test_fraction,
        args.seed,
        args.group_column,
    )
    x_train, y_train = features[train_indices], targets[train_indices]
    x_test = features[test_indices]
    test_frame = frame.iloc[test_indices].copy()

    model = make_residual_regressor(args)
    model.fit(x_train, y_train)

    started = time.perf_counter()
    predictions = model.predict(x_test)
    prediction_seconds = time.perf_counter() - started
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    metrics = correction_metrics(test_frame, predictions, prediction_seconds)

    metadata = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
        "dataset_type": "acceleration_residual",
        "model_kind": "accel_residual",
        "model_type": args.model,
        "knn_k": args.knn_k if args.model == "knn" else "",
        "targets": RESIDUAL_TARGETS,
        "feature_names": transformer.feature_names,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_config_count": len(train_groups),
        "test_config_count": len(test_groups),
        "train_config_ids": train_groups,
        "test_config_ids": test_groups,
        "dataset_path": str(args.data),
        "trained_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    bundle = {
        **metadata,
        "feature_transformer": transformer,
        "feature_scaler": scaler,
        "model": model,
        "metrics": metrics_dict(metrics),
    }
    save_model_bundle(args.output, bundle)

    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.md")
    report = correction_markdown_report("Acceleration Residual Model Evaluation", metadata, metrics)
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
    parser.add_argument(
        "--model",
        choices=["knn", "linear", "random_forest", "gradient_boosting"],
        default="knn",
    )
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--group-column", default="config_id")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
