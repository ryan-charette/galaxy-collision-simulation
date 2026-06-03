"""Train supervised models for solver runtime and drift metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from python.ml.datasets import (
    completed_rows,
    finite_target_rows,
    load_dataset,
)
from python.ml.features import (
    runtime_feature_columns,
)
from python.ml.training import RegressionTrainingConfig, train_regression_bundle


DEFAULT_TARGETS = [
    "median_step_time",
    "particle_steps_per_second",
    "energy_drift_final",
    "momentum_drift_final",
]


def train(args: argparse.Namespace) -> None:
    frame = finite_target_rows(completed_rows(load_dataset(args.data)), args.targets)
    if len(frame) < 2:
        raise ValueError("Need at least two completed rows with finite targets")

    numeric_features, categorical_features = runtime_feature_columns(frame)
    result = train_regression_bundle(
        frame,
        RegressionTrainingConfig(
            dataset_type="solver_tuning",
            model_kind="solver_cost",
            report_title="Solver Cost Model Evaluation",
            targets=args.targets,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_type=args.model,
            seed=args.seed,
            test_fraction=args.test_fraction,
            dataset_path=args.data,
            output_path=args.output,
            report_path=args.report,
            selection_objective="median_step_time",
        ),
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {result['report_path']}")


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
