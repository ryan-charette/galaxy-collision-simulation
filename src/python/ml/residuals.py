"""Utilities for learned acceleration-residual correction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RESIDUAL_SCHEMA_VERSION = "0.1.0"

RESIDUAL_TARGETS = ["accel_error_x", "accel_error_y", "accel_error_z"]

RESIDUAL_NUMERIC_FEATURES = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "mass",
    "group_id",
    "approx_accel_x",
    "approx_accel_y",
    "approx_accel_z",
    "local_density_estimate",
    "nearest_neighbor_distance",
    "distance_from_center_of_mass",
    "tree_theta",
    "tree_leaf_capacity",
    "fmm_expansion_order",
    "leaf_particle_count",
    "tree_depth",
    "cell_size",
    "distance_to_cell_center",
    "n_particles",
    "softening",
]

RESIDUAL_CATEGORICAL_FEATURES = ["solver"]


@dataclass
class CorrectionMetrics:
    """Metrics comparing approximate and residual-corrected accelerations."""

    approximate_rmse: float
    corrected_rmse: float
    approximate_mae: float
    corrected_mae: float
    approximate_max_error: float
    corrected_max_error: float
    approximate_relative_rmse: float
    corrected_relative_rmse: float
    improvement_fraction: float
    prediction_seconds: float
    rows: int


@dataclass
class NumpyKnnRegressor:
    """Small dependency-free KNN regressor for residual baselines."""

    k: int = 8
    train_features: np.ndarray | None = None
    train_targets: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "NumpyKnnRegressor":
        if targets.ndim == 1:
            targets = targets[:, None]
        self.train_features = np.asarray(features, dtype=float)
        self.train_targets = np.asarray(targets, dtype=float)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.train_features is None or self.train_targets is None:
            raise RuntimeError("KNN regressor is not fitted")
        features = np.asarray(features, dtype=float)
        predictions = np.zeros((features.shape[0], self.train_targets.shape[1]), dtype=float)
        k = max(1, min(self.k, self.train_features.shape[0]))
        for start in range(0, features.shape[0], 512):
            stop = min(start + 512, features.shape[0])
            chunk = features[start:stop]
            distances = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                - 2.0 * chunk @ self.train_features.T
                + np.sum(self.train_features * self.train_features, axis=1)[None, :]
            )
            nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            predictions[start:stop] = self.train_targets[nearest].mean(axis=1)
        return predictions


def residual_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return available residual-model numeric and categorical feature columns."""
    numeric = [column for column in RESIDUAL_NUMERIC_FEATURES if column in frame.columns]
    categorical = [column for column in RESIDUAL_CATEGORICAL_FEATURES if column in frame.columns]
    return numeric, categorical


def finite_residual_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return residual rows with finite direct, approximate, and error accelerations."""
    filtered = frame.copy()
    for column in [*RESIDUAL_TARGETS, "direct_accel_x", "direct_accel_y", "direct_accel_z"]:
        values = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered[np.isfinite(values)]
    return filtered.copy()


def split_by_config(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int,
    group_column: str = "config_id",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Split residual rows by configuration group to avoid train/test leakage."""
    if group_column not in frame.columns:
        group_column = "run_id"
    groups = np.asarray(sorted(str(value) for value in frame[group_column].dropna().unique()))
    if len(groups) == 0:
        raise ValueError("Residual dataset has no configuration groups")
    if len(groups) == 1:
        indices = np.arange(len(frame))
        return indices, indices, groups.tolist(), groups.tolist()

    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    test_count = min(test_count, len(shuffled) - 1)
    test_groups = {str(value) for value in shuffled[:test_count]}
    train_groups = {str(value) for value in shuffled[test_count:]}
    train_indices = frame.index[frame[group_column].astype(str).isin(train_groups)].to_numpy()
    test_indices = frame.index[frame[group_column].astype(str).isin(test_groups)].to_numpy()
    return (
        train_indices,
        test_indices,
        sorted(train_groups),
        sorted(test_groups),
    )


def acceleration_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return direct acceleration, approximate acceleration, and residual-error matrices."""
    direct = frame[["direct_accel_x", "direct_accel_y", "direct_accel_z"]].to_numpy(dtype=float)
    approx = frame[["approx_accel_x", "approx_accel_y", "approx_accel_z"]].to_numpy(dtype=float)
    error = frame[RESIDUAL_TARGETS].to_numpy(dtype=float)
    return direct, approx, error


def correction_metrics(
    frame: pd.DataFrame,
    predicted_error: np.ndarray,
    prediction_seconds: float = 0.0,
) -> CorrectionMetrics:
    """Compute one-step residual-correction quality metrics."""
    direct, approx, true_error = acceleration_matrices(frame)
    corrected_error = true_error - predicted_error
    approx_norm = np.linalg.norm(direct - approx, axis=1)
    corrected_norm = np.linalg.norm(corrected_error, axis=1)
    direct_norm = np.linalg.norm(direct, axis=1)
    reference_rmse = max(float(math.sqrt(np.mean(direct_norm * direct_norm))), 1.0e-12)
    approximate_rmse = float(math.sqrt(np.mean(approx_norm * approx_norm)))
    corrected_rmse = float(math.sqrt(np.mean(corrected_norm * corrected_norm)))
    improvement = (
        (approximate_rmse - corrected_rmse) / approximate_rmse
        if approximate_rmse > 0.0
        else 0.0
    )
    return CorrectionMetrics(
        approximate_rmse=approximate_rmse,
        corrected_rmse=corrected_rmse,
        approximate_mae=float(np.mean(approx_norm)),
        corrected_mae=float(np.mean(corrected_norm)),
        approximate_max_error=float(np.max(approx_norm)),
        corrected_max_error=float(np.max(corrected_norm)),
        approximate_relative_rmse=approximate_rmse / reference_rmse,
        corrected_relative_rmse=corrected_rmse / reference_rmse,
        improvement_fraction=float(improvement),
        prediction_seconds=float(prediction_seconds),
        rows=len(frame),
    )


def metrics_dict(metrics: CorrectionMetrics) -> dict[str, Any]:
    """Convert correction metrics into a JSON/CSV-friendly dictionary."""
    return {
        "rows": metrics.rows,
        "approximate_rmse": metrics.approximate_rmse,
        "corrected_rmse": metrics.corrected_rmse,
        "approximate_mae": metrics.approximate_mae,
        "corrected_mae": metrics.corrected_mae,
        "approximate_max_error": metrics.approximate_max_error,
        "corrected_max_error": metrics.corrected_max_error,
        "approximate_relative_rmse": metrics.approximate_relative_rmse,
        "corrected_relative_rmse": metrics.corrected_relative_rmse,
        "improvement_fraction": metrics.improvement_fraction,
        "prediction_seconds": metrics.prediction_seconds,
        "prediction_rows_per_second": (
            metrics.rows / metrics.prediction_seconds if metrics.prediction_seconds > 0.0 else math.inf
        ),
    }


def correction_markdown_report(
    title: str,
    metadata: dict[str, Any],
    metrics: CorrectionMetrics,
    stability: dict[str, Any] | None = None,
) -> str:
    """Render residual-correction metrics as a Markdown report."""
    lines = [f"# {title}", "", "## Metadata", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## One-Step Force Correction",
            "",
            "| Rows | Approx RMSE | Corrected RMSE | Approx relative RMSE | "
            "Corrected relative RMSE | Improvement | Prediction rows/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {metrics.rows} | {metrics.approximate_rmse:.6g} | {metrics.corrected_rmse:.6g} | "
                f"{metrics.approximate_relative_rmse:.6g} | {metrics.corrected_relative_rmse:.6g} | "
                f"{100.0 * metrics.improvement_fraction:.3f}% | "
                f"{metrics_dict(metrics)['prediction_rows_per_second']:.6g} |"
            ),
            "",
        ]
    )
    if stability is not None:
        lines.extend(["## Short Integration Sanity", ""])
        for key, value in stability.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    return "\n".join(lines)
