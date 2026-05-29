"""Evaluation helpers for supervised ML baselines."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> dict[str, dict[str, float]]:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    metrics: dict[str, dict[str, float]] = {}
    for index, target in enumerate(targets):
        residual = y_pred[:, index] - y_true[:, index]
        mae = float(np.mean(np.abs(residual)))
        rmse = float(math.sqrt(np.mean(residual * residual)))
        variance = float(np.var(y_true[:, index]))
        r2 = 1.0 - float(np.mean(residual * residual)) / variance if variance > 0.0 else float("nan")
        metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics


def baseline_metrics(y_true: np.ndarray, y_baseline: np.ndarray, targets: list[str]) -> dict[str, dict[str, float]]:
    return regression_metrics(y_true, y_baseline, targets)


def model_beats_baseline(
    model_metrics: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> dict[str, bool]:
    return {
        target: model_metrics[target]["rmse"] < baseline[target]["rmse"]
        for target in model_metrics
    }


def solver_selection_accuracy(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    targets: list[str],
    objective: str,
) -> float | None:
    if objective not in targets or "solver" not in frame.columns:
        return None

    target_index = targets.index(objective)
    working = frame.copy()
    working["_prediction"] = predictions[:, target_index]
    group_columns = [
        column
        for column in [
            "n_particles",
            "steps",
            "dt",
            "softening",
            "tree_theta",
            "tree_leaf_capacity",
            "fmm_expansion_order",
            "hardware_type",
            "output_format",
        ]
        if column in working.columns
    ]
    if not group_columns:
        return None

    total = 0
    correct = 0
    for _, group in working.groupby(group_columns, dropna=False):
        if group["solver"].nunique() < 2:
            continue
        actual_solver = group.loc[group[objective].astype(float).idxmin(), "solver"]
        predicted_solver = group.loc[group["_prediction"].astype(float).idxmin(), "solver"]
        total += 1
        correct += int(actual_solver == predicted_solver)
    if total == 0:
        return None
    return correct / total


def markdown_report(
    title: str,
    metadata: dict[str, Any],
    model_metrics: dict[str, dict[str, float]],
    mean_baseline: dict[str, dict[str, float]],
    beats_baseline: dict[str, bool],
    selection_accuracy: float | None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Regression Metrics",
            "",
            "| Target | MAE | RMSE | R2 | Mean baseline RMSE | Beats baseline |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for target, metrics in model_metrics.items():
        lines.append(
            f"| `{target}` | {metrics['mae']:.6g} | {metrics['rmse']:.6g} | "
            f"{metrics['r2']:.6g} | {mean_baseline[target]['rmse']:.6g} | "
            f"{'yes' if beats_baseline[target] else 'no'} |"
        )
    lines.extend(["", "## Selection", ""])
    if selection_accuracy is None:
        lines.append("Solver-selection accuracy: not available for this dataset/test split.")
    else:
        lines.append(f"Solver-selection accuracy: {selection_accuracy:.3f}")
    lines.append("")
    return "\n".join(lines)
