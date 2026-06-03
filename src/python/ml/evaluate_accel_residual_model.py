"""Evaluate learned acceleration-residual correction on held-out configs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from python.ml.datasets import load_dataset, load_model_bundle, write_json
from python.ml.features import predict_bundle
from python.ml.residuals import (
    RESIDUAL_TARGETS,
    correction_markdown_report,
    correction_metrics,
    finite_residual_rows,
    metrics_dict,
)


def filter_frame(args: argparse.Namespace, bundle: dict[str, Any]) -> pd.DataFrame:
    frame = finite_residual_rows(load_dataset(args.data)).reset_index(drop=True)
    if args.heldout_from_model:
        test_config_ids = {str(value) for value in bundle.get("test_config_ids", [])}
        if test_config_ids and "config_id" in frame.columns:
            frame = frame[frame["config_id"].astype(str).isin(test_config_ids)].copy()
    if args.config_id:
        wanted = set(args.config_id)
        frame = frame[frame["config_id"].astype(str).isin(wanted)].copy()
    if frame.empty:
        raise ValueError("No residual rows are available for evaluation")
    return frame.reset_index(drop=True)


def stability_check(
    frame: pd.DataFrame,
    predicted_error: np.ndarray,
    dt: float,
    steps: int,
) -> dict[str, Any]:
    if steps <= 0:
        return {"tested": False, "reason": "stability steps disabled"}

    approximate_accel = frame[["approx_accel_x", "approx_accel_y", "approx_accel_z"]].to_numpy(dtype=float)
    corrected_accel = approximate_accel + predicted_error
    masses = frame["mass"].to_numpy(dtype=float)
    positions0 = frame[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
    velocities0 = frame[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=float)

    def integrate(accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = positions0.copy()
        velocities = velocities0.copy()
        for _ in range(steps):
            velocities += accel * dt
            positions += velocities * dt
        return positions, velocities

    approx_positions, approx_velocities = integrate(approximate_accel)
    corrected_positions, corrected_velocities = integrate(corrected_accel)
    initial_ke = 0.5 * np.sum(masses * np.sum(velocities0 * velocities0, axis=1))
    approx_ke = 0.5 * np.sum(masses * np.sum(approx_velocities * approx_velocities, axis=1))
    corrected_ke = 0.5 * np.sum(masses * np.sum(corrected_velocities * corrected_velocities, axis=1))
    finite = bool(np.all(np.isfinite(corrected_positions)) and np.all(np.isfinite(corrected_velocities)))
    return {
        "tested": True,
        "steps": steps,
        "dt": dt,
        "finite": finite,
        "approx_max_radius": float(np.max(np.linalg.norm(approx_positions, axis=1))),
        "corrected_max_radius": float(np.max(np.linalg.norm(corrected_positions, axis=1))),
        "approx_kinetic_ratio": float(approx_ke / max(initial_ke, 1.0e-12)),
        "corrected_kinetic_ratio": float(corrected_ke / max(initial_ke, 1.0e-12)),
    }


def evaluate(args: argparse.Namespace) -> None:
    bundle = load_model_bundle(args.model)
    if bundle.get("model_kind") != "accel_residual":
        raise ValueError(f"Expected accel_residual model, got {bundle.get('model_kind')}")
    frame = filter_frame(args, bundle)

    started = time.perf_counter()
    predictions = predict_bundle(bundle, frame)
    prediction_seconds = time.perf_counter() - started
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    metrics = correction_metrics(frame, predictions, prediction_seconds)

    stability: dict[str, Any] | None = None
    if args.stability_steps > 0:
        if metrics.corrected_rmse <= metrics.approximate_rmse:
            stability = stability_check(frame, predictions, args.stability_dt, args.stability_steps)
        else:
            stability = {
                "tested": False,
                "reason": "one-step correction did not improve RMSE",
            }

    metadata = {
        "model_path": str(args.model),
        "dataset_path": str(args.data),
        "model_type": bundle.get("model_type", ""),
        "rows": len(frame),
        "heldout_from_model": args.heldout_from_model,
        "targets": ", ".join(RESIDUAL_TARGETS),
    }
    report = correction_markdown_report(
        "Acceleration Residual Correction Evaluation",
        metadata,
        metrics,
        stability,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    write_json(
        args.output.with_suffix(args.output.suffix + ".summary.json"),
        {
            "metadata": metadata,
            "metrics": metrics_dict(metrics),
            "stability": stability,
        },
    )
    if args.predictions_output:
        prediction_frame = frame[["config_id", "particle_id", "solver"]].copy()
        prediction_frame["predicted_accel_error_x"] = predictions[:, 0]
        prediction_frame["predicted_accel_error_y"] = predictions[:, 1]
        prediction_frame["predicted_accel_error_z"] = predictions[:, 2]
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(args.predictions_output, index=False)
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--config-id", nargs="+", default=None)
    parser.add_argument("--heldout-from-model", action="store_true")
    parser.add_argument("--stability-steps", type=int, default=0)
    parser.add_argument("--stability-dt", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
