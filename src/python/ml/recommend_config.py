"""Recommend simulator solver settings from trained cost and error models."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from python.ml.datasets import load_model_bundle
from python.ml.features import predict_bundle


DEFAULT_COST_MODEL = Path("experiments/ml_models/solver_cost_model.pkl")
DEFAULT_FORCE_MODEL = Path("experiments/ml_models/force_error_model.pkl")


def candidate_rows(args: argparse.Namespace) -> pd.DataFrame:
    solvers = args.solvers or (["direct", "tree", "fmm"] if args.hardware == "cpu" else ["cuda-tree", "cuda-fmm", "tree", "fmm"])
    rows: list[dict[str, Any]] = []
    for solver, theta, leaf_capacity, expansion_order in product(
        solvers,
        args.tree_theta,
        args.tree_leaf_capacity,
        args.fmm_expansion_order,
    ):
        rows.append(
            {
                "solver": solver,
                "n_particles": args.n_particles,
                "steps": args.steps,
                "dt": args.dt,
                "softening": args.softening,
                "tree_theta": theta,
                "tree_leaf_capacity": leaf_capacity,
                "fmm_expansion_order": expansion_order,
                "hardware_type": args.hardware,
                "output_format": args.output_format,
            }
        )
    return pd.DataFrame(rows)


def predict_cost(cost_model: dict[str, Any] | None, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if cost_model is None:
        heuristic = {
            "direct": 4.0,
            "tree": 1.0,
            "fmm": 1.4,
            "cuda-tree": 0.35,
            "cuda-fmm": 0.5,
        }
        frame["predicted_median_step_time"] = frame["solver"].map(heuristic).fillna(2.0)
        frame["predicted_particle_steps_per_second"] = frame["n_particles"] / frame["predicted_median_step_time"]
        return frame

    predictions = predict_bundle(cost_model, frame)
    targets = list(cost_model["targets"])
    for index, target in enumerate(targets):
        frame[f"predicted_{target}"] = predictions[:, index]
    if "predicted_median_step_time" not in frame.columns:
        frame["predicted_median_step_time"] = 1.0 / frame.get("predicted_particle_steps_per_second", 1.0)
    return frame


def add_force_predictions(
    force_model: dict[str, Any] | None,
    frame: pd.DataFrame,
    direct_runtime: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["runtime_approx"] = frame["predicted_median_step_time"] * frame["steps"]
    frame["runtime_direct"] = direct_runtime
    if force_model is None:
        frame["predicted_force_rmse"] = float("nan")
        frame["predicted_relative_force_rmse"] = float("nan")
    else:
        predictions = predict_bundle(force_model, frame)
        targets = list(force_model["targets"])
        for index, target in enumerate(targets):
            frame[f"predicted_{target}"] = predictions[:, index]

    direct_mask = frame["solver"] == "direct"
    if "predicted_force_rmse" in frame.columns:
        frame.loc[direct_mask, "predicted_force_rmse"] = 0.0
    if "predicted_relative_force_rmse" in frame.columns:
        frame.loc[direct_mask, "predicted_relative_force_rmse"] = 0.0
    return frame


def direct_runtime_estimate(frame: pd.DataFrame) -> float:
    direct_rows = frame[frame["solver"] == "direct"]
    if direct_rows.empty:
        return float(frame["predicted_median_step_time"].max() * frame["steps"].iloc[0])
    return float(direct_rows["predicted_median_step_time"].median() * direct_rows["steps"].iloc[0])


def choose_candidate(args: argparse.Namespace, frame: pd.DataFrame) -> pd.Series:
    feasible = frame.copy()
    if args.target_force_rmse is not None and "predicted_force_rmse" in feasible.columns:
        predicted = pd.to_numeric(feasible["predicted_force_rmse"], errors="coerce")
        with_predictions = feasible[predicted.notna()].copy()
        if not with_predictions.empty:
            feasible = with_predictions[predicted.loc[with_predictions.index] <= args.target_force_rmse]
            if feasible.empty:
                feasible = with_predictions
    return feasible.sort_values("predicted_median_step_time").iloc[0]


def hand_tuned_baselines(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {
            "name": "direct",
            "solver": "direct",
            "tree_theta": 0.7,
            "tree_leaf_capacity": 32,
            "fmm_expansion_order": 0,
        },
        {
            "name": "tree_default",
            "solver": "tree",
            "tree_theta": 0.7,
            "tree_leaf_capacity": 32,
            "fmm_expansion_order": 0,
        },
        {
            "name": "fmm_default",
            "solver": "fmm",
            "tree_theta": 0.6,
            "tree_leaf_capacity": 16,
            "fmm_expansion_order": 4,
        },
    ]
    for row in rows:
        row.update(
            {
                "n_particles": args.n_particles,
                "steps": args.steps,
                "dt": args.dt,
                "softening": args.softening,
                "hardware_type": args.hardware,
                "output_format": args.output_format,
            }
        )
    return pd.DataFrame(rows)


def print_toml(row: pd.Series) -> None:
    print("[simulation]")
    print(f'solver = "{row["solver"]}"')
    print(f"tree_theta = {float(row['tree_theta']):g}")
    print(f"tree_leaf_capacity = {int(row['tree_leaf_capacity'])}")
    print(f"fmm_expansion_order = {int(row['fmm_expansion_order'])}")


def recommend(args: argparse.Namespace) -> None:
    cost_model = load_model_bundle(args.cost_model) if args.cost_model.exists() else None
    force_model = load_model_bundle(args.force_model) if args.force_model.exists() else None
    candidates = predict_cost(cost_model, candidate_rows(args))
    candidates = add_force_predictions(force_model, candidates, direct_runtime_estimate(candidates))
    recommendation = choose_candidate(args, candidates)

    if args.report:
        baselines = predict_cost(cost_model, hand_tuned_baselines(args))
        baselines = add_force_predictions(force_model, baselines, direct_runtime_estimate(candidates))
        payload = {
            "recommendation": recommendation.to_dict(),
            "hand_tuned_baselines": baselines.to_dict(orient="records"),
            "cost_model": str(args.cost_model) if cost_model is not None else "heuristic",
            "force_model": str(args.force_model) if force_model is not None else "unavailable",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    print_toml(recommendation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", type=int, required=True)
    parser.add_argument("--target-force-rmse", type=float, default=None)
    parser.add_argument("--hardware", choices=["cpu", "cuda", "mpi"], default="cpu")
    parser.add_argument("--cost-model", type=Path, default=DEFAULT_COST_MODEL)
    parser.add_argument("--force-model", type=Path, default=DEFAULT_FORCE_MODEL)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--softening", type=float, default=0.02)
    parser.add_argument("--output-format", default="csv")
    parser.add_argument("--tree-theta", nargs="+", type=float, default=[0.5, 0.7, 1.0])
    parser.add_argument("--tree-leaf-capacity", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--fmm-expansion-order", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--solvers", nargs="+", default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    recommend(parse_args())


if __name__ == "__main__":
    main()
