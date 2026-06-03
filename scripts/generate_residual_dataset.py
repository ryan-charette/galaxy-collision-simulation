"""Generate paired direct-vs-approx acceleration residual datasets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from python.ml import DATASET_SCHEMA_VERSION
from python.ml.features import estimate_tree_depth
from python.ml.residuals import RESIDUAL_SCHEMA_VERSION
from python.utils.snapshots import load_acceleration_dump
from scripts.experiment_utils import (
    resolve_simulator_executable,
    run_simulator,
    safe_float_label,
    write_two_galaxy_config,
)


@dataclass(frozen=True)
class ResidualCase:
    solver: str
    n_particles: int
    tree_theta: float
    tree_leaf_capacity: int
    fmm_expansion_order: int
    softening: float


@dataclass(frozen=True)
class SimRun:
    run_id: str
    config_path: Path
    output_dir: Path
    acceleration_path: Path
    metadata: dict[str, Any]
    seconds: float


def safe_label(value: float) -> str:
    return safe_float_label(value)


def case_id(case: ResidualCase) -> str:
    return (
        f"{case.solver}_n{case.n_particles}_theta{safe_label(case.tree_theta)}_"
        f"leaf{case.tree_leaf_capacity}_p{case.fmm_expansion_order}_soft{safe_label(case.softening)}"
    )


def run_simulation(
    executable: Path,
    config_path: Path,
    output_dir: Path,
    log_path: Path,
    resume: bool,
) -> SimRun:
    acceleration_path = output_dir / "accelerations_000000.csv"
    if resume and acceleration_path.exists():
        metadata_path = output_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return SimRun(output_dir.name, config_path, output_dir, acceleration_path, metadata, 0.0)

    completed = run_simulator(
        executable,
        config_path,
        output_dir,
        cwd=REPO_ROOT,
        log_path=log_path,
    )
    if completed.exit_code != 0:
        raise RuntimeError(f"Simulation failed for {config_path}. See {log_path}")
    if not acceleration_path.exists():
        raise RuntimeError(f"Simulation did not write {acceleration_path}")
    return SimRun(
        output_dir.name,
        config_path,
        output_dir,
        acceleration_path,
        completed.metadata,
        completed.seconds,
    )


def particle_context_features(
    positions: np.ndarray,
    masses: np.ndarray,
    leaf_capacity: int,
) -> dict[str, np.ndarray | float | int]:
    count = positions.shape[0]
    total_mass = float(np.sum(masses))
    center_of_mass = np.sum(positions * masses[:, None], axis=0) / max(total_mass, 1.0e-12)
    distances_from_com = np.linalg.norm(positions - center_of_mass[None, :], axis=1)

    if count <= 1:
        nearest = np.zeros(count, dtype=float)
        density = np.zeros(count, dtype=float)
    else:
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        k = min(8, count - 1)
        kth = np.partition(distances, kth=k - 1, axis=1)[:, k - 1]
        volume = (4.0 / 3.0) * math.pi * np.maximum(kth, 1.0e-12) ** 3
        density = k / volume

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    bbox_size = max(float(np.max(maxs - mins)), 1.0e-12)
    tree_depth = estimate_tree_depth(count, leaf_capacity)
    cells_per_axis = 2**tree_depth
    cell_size = bbox_size / cells_per_axis
    cell_indices = np.floor((positions - mins[None, :]) / cell_size)
    cell_indices = np.clip(cell_indices, 0, cells_per_axis - 1)
    cell_centers = mins[None, :] + (cell_indices + 0.5) * cell_size
    distance_to_cell_center = np.linalg.norm(positions - cell_centers, axis=1)

    return {
        "local_density_estimate": density,
        "nearest_neighbor_distance": nearest,
        "distance_from_center_of_mass": distances_from_com,
        "tree_depth": tree_depth,
        "cell_size": cell_size,
        "distance_to_cell_center": distance_to_cell_center,
        "leaf_particle_count": min(leaf_capacity, count),
    }


def paired_rows(
    case: ResidualCase,
    direct_run: SimRun,
    approx_run: SimRun,
    dt: float,
    steps: int,
) -> list[dict[str, Any]]:
    direct = load_acceleration_dump(direct_run.acceleration_path)
    approx = load_acceleration_dump(approx_run.acceleration_path)
    if not np.array_equal(direct.ids, approx.ids):
        raise RuntimeError(f"Particle IDs do not match for {direct.path} and {approx.path}")
    features = particle_context_features(approx.positions, approx.masses, case.tree_leaf_capacity)
    errors = direct.accelerations - approx.accelerations
    rows: list[dict[str, Any]] = []
    config_hash = approx_run.metadata.get("config_sha256", "")
    direct_hash = direct_run.metadata.get("config_sha256", "")
    for index, particle_id in enumerate(approx.ids):
        row = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
            "status": "completed",
            "run_id": approx_run.run_id,
            "reference_run_id": direct_run.run_id,
            "config_id": case_id(case),
            "git_commit": approx_run.metadata.get("git_commit", "unavailable"),
            "config_sha256": config_hash,
            "direct_config_sha256": direct_hash,
            "solver": case.solver,
            "n_particles": case.n_particles,
            "steps": steps,
            "dt": dt,
            "softening": case.softening,
            "tree_theta": case.tree_theta,
            "tree_leaf_capacity": case.tree_leaf_capacity,
            "fmm_expansion_order": case.fmm_expansion_order,
            "particle_id": int(particle_id),
            "position_x": approx.positions[index, 0],
            "position_y": approx.positions[index, 1],
            "position_z": approx.positions[index, 2],
            "velocity_x": approx.velocities[index, 0],
            "velocity_y": approx.velocities[index, 1],
            "velocity_z": approx.velocities[index, 2],
            "mass": approx.masses[index],
            "group_id": int(approx.group_id[index]),
            "direct_accel_x": direct.accelerations[index, 0],
            "direct_accel_y": direct.accelerations[index, 1],
            "direct_accel_z": direct.accelerations[index, 2],
            "approx_accel_x": approx.accelerations[index, 0],
            "approx_accel_y": approx.accelerations[index, 1],
            "approx_accel_z": approx.accelerations[index, 2],
            "accel_error_x": errors[index, 0],
            "accel_error_y": errors[index, 1],
            "accel_error_z": errors[index, 2],
            "local_density_estimate": features["local_density_estimate"][index],
            "nearest_neighbor_distance": features["nearest_neighbor_distance"][index],
            "distance_from_center_of_mass": features["distance_from_center_of_mass"][index],
            "leaf_particle_count": features["leaf_particle_count"],
            "tree_depth": features["tree_depth"],
            "cell_size": features["cell_size"],
            "distance_to_cell_center": features["distance_to_cell_center"][index],
            "approx_seconds": approx_run.seconds,
            "direct_seconds": direct_run.seconds,
        }
        rows.append(row)
    return rows


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(path, index=False, engine="pyarrow")
            return
        except ImportError as exc:
            raise RuntimeError("Parquet output requires pyarrow. Use .csv or install dependencies.") from exc
    frame.to_csv(path, index=False)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(
        ["solver", "n_particles", "tree_theta", "tree_leaf_capacity", "fmm_expansion_order"]
    )
    lines = [
        "# Acceleration Residual Dataset",
        "",
        f"Rows: {len(frame)}",
        f"Configs: {frame['config_id'].nunique() if not frame.empty else 0}",
        "",
        "| Solver | N | Theta | Leaf | p | Rows | Approx RMSE | Relative RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, group in grouped:
        solver, particles, theta, leaf, order = key
        direct = group[["direct_accel_x", "direct_accel_y", "direct_accel_z"]].to_numpy(dtype=float)
        approx = group[["approx_accel_x", "approx_accel_y", "approx_accel_z"]].to_numpy(dtype=float)
        error_norm = np.linalg.norm(direct - approx, axis=1)
        direct_norm = np.linalg.norm(direct, axis=1)
        rmse = float(np.sqrt(np.mean(error_norm * error_norm)))
        relative = rmse / max(float(np.sqrt(np.mean(direct_norm * direct_norm))), 1.0e-12)
        lines.append(
            f"| `{solver}` | {particles} | {theta:g} | {leaf} | {order} | "
            f"{len(group)} | {rmse:.6g} | {relative:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cases(args: argparse.Namespace) -> list[ResidualCase]:
    cases = []
    for solver in args.solvers:
        for particles in args.particles:
            for softening in args.softening:
                for theta in args.theta:
                    for leaf in args.leaf_capacity:
                        orders = args.expansion_order if solver != "tree" else [args.expansion_order[0]]
                        for order in orders:
                            cases.append(ResidualCase(solver, particles, theta, leaf, order, softening))
    return cases


def generate(args: argparse.Namespace) -> None:
    executable = resolve_simulator_executable(args.executable)
    cases = build_cases(args)
    direct_runs: dict[tuple[int, float], SimRun] = {}
    rows: list[dict[str, Any]] = []
    args.run_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        direct_key = (case.n_particles, case.softening)
        if direct_key not in direct_runs:
            direct_case = ResidualCase(
                "direct",
                case.n_particles,
                case.tree_theta,
                case.tree_leaf_capacity,
                case.fmm_expansion_order,
                case.softening,
            )
            direct_id = f"direct_reference_n{case.n_particles}_soft{safe_label(case.softening)}"
            direct_output = args.run_root / "runs" / direct_id
            direct_config = args.run_root / "configs" / f"{direct_id}.toml"
            write_two_galaxy_config(
                direct_config,
                name=f"residual_direct_{case.n_particles}",
                solver="direct",
                particles=direct_case.n_particles,
                steps=args.steps,
                dt=args.dt,
                snapshot_every=1,
                output=direct_output,
                output_format="none",
                theta=direct_case.tree_theta,
                leaf_capacity=direct_case.tree_leaf_capacity,
                expansion_order=direct_case.fmm_expansion_order,
                softening=direct_case.softening,
                seed=20260526,
                acceleration_dump=True,
            )
            direct_runs[direct_key] = run_simulation(
                executable,
                direct_config,
                direct_output,
                args.run_root / "logs" / f"{direct_id}.log",
                args.resume,
            )

        label = case_id(case)
        output_dir = args.run_root / "runs" / label
        config_path = args.run_root / "configs" / f"{label}.toml"
        write_two_galaxy_config(
            config_path,
            name=f"residual_{case.solver}_{case.n_particles}",
            solver=case.solver,
            particles=case.n_particles,
            steps=args.steps,
            dt=args.dt,
            snapshot_every=1,
            output=output_dir,
            output_format="none",
            theta=case.tree_theta,
            leaf_capacity=case.tree_leaf_capacity,
            expansion_order=case.fmm_expansion_order,
            softening=case.softening,
            seed=20260526,
            acceleration_dump=True,
        )
        approx_run = run_simulation(
            executable,
            config_path,
            output_dir,
            args.run_root / "logs" / f"{label}.log",
            args.resume,
        )
        rows.extend(paired_rows(case, direct_runs[direct_key], approx_run, args.dt, args.steps))
        print(f"{label}: {case.n_particles} residual rows")

    write_dataset(args.output, rows)
    write_summary(args.summary or args.output.with_suffix(args.output.suffix + ".summary.md"), rows)
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/ml_datasets/accel_residuals.csv"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/error_correction/residual_runs"))
    parser.add_argument("--executable", type=Path, default=None)
    parser.add_argument("--solvers", nargs="+", default=["tree", "fmm"])
    parser.add_argument("--particles", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--theta", nargs="+", type=float, default=[0.4, 0.6])
    parser.add_argument("--leaf-capacity", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--expansion-order", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--softening", nargs="+", type=float, default=[0.02])
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.solvers = ["tree", "fmm"]
        args.particles = [32]
        args.theta = [0.4, 0.6]
        args.leaf_capacity = [8]
        args.expansion_order = [0, 2]
        args.softening = [0.02]
    return args


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
