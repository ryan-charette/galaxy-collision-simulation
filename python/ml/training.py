"""Shared supervised-regression training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from python.ml import DATASET_SCHEMA_VERSION
from python.ml.datasets import (
    numeric_matrix,
    save_model_bundle,
    train_test_split_indices,
    write_json,
)
from python.ml.evaluation import (
    baseline_metrics,
    markdown_report,
    model_beats_baseline,
    regression_metrics,
    solver_selection_accuracy,
)
from python.ml.features import FeatureTransformer, MeanRegressor, StandardScaler, make_regressor


@dataclass(frozen=True)
class RegressionTrainingConfig:
    dataset_type: str
    model_kind: str
    report_title: str
    targets: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    model_type: str
    seed: int
    test_fraction: float
    dataset_path: Path
    output_path: Path
    report_path: Path | None = None
    selection_objective: str | None = None


def fit_feature_pipeline(
    frame: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[FeatureTransformer, StandardScaler, Any]:
    transformer = FeatureTransformer(numeric_features, categorical_features).fit(frame)
    features = transformer.transform(frame)
    scaler = StandardScaler().fit(features)
    return transformer, scaler, scaler.transform(features)


def train_regression_bundle(frame: Any, config: RegressionTrainingConfig) -> dict[str, Any]:
    transformer, scaler, features = fit_feature_pipeline(
        frame,
        config.numeric_features,
        config.categorical_features,
    )
    targets = numeric_matrix(frame, config.targets)

    split = train_test_split_indices(len(frame), config.test_fraction, config.seed)
    x_train, y_train = features[split.train_indices], targets[split.train_indices]
    x_test, y_test = features[split.test_indices], targets[split.test_indices]
    test_frame = frame.iloc[split.test_indices].copy()

    model = make_regressor(config.model_type, config.seed)
    model.fit(x_train, y_train)
    baseline = MeanRegressor().fit(y_train)

    predictions = model.predict(x_test)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    baseline_predictions = baseline.predict(len(x_test))
    metrics = regression_metrics(y_test, predictions, config.targets)
    mean_metrics = baseline_metrics(y_test, baseline_predictions, config.targets)
    beats = model_beats_baseline(metrics, mean_metrics)
    selection_accuracy = (
        solver_selection_accuracy(
            test_frame,
            predictions,
            config.targets,
            config.selection_objective,
        )
        if config.selection_objective is not None
        else None
    )

    metadata = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_type": config.dataset_type,
        "model_kind": config.model_kind,
        "model_type": config.model_type,
        "targets": config.targets,
        "feature_names": transformer.feature_names,
        "train_rows": int(len(split.train_indices)),
        "test_rows": int(len(split.test_indices)),
        "dataset_path": str(config.dataset_path),
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
        "solver_selection_accuracy": selection_accuracy,
    }
    save_model_bundle(config.output_path, bundle)

    report_path = config.report_path or config.output_path.with_suffix(config.output_path.suffix + ".report.md")
    report = markdown_report(
        config.report_title,
        metadata,
        metrics,
        mean_metrics,
        beats,
        selection_accuracy,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_json(config.output_path.with_suffix(config.output_path.suffix + ".metadata.json"), metadata)
    return {"bundle": bundle, "report_path": report_path}
