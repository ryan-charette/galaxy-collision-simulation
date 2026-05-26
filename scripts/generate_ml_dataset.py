"""Generate ML-ready solver-tuning datasets from simulator sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 compatibility
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import sweep as sweep_runner


SOLVER_TUNING_COLUMNS = [
    "run_id",
    "status",
    "exit_code",
    "error",
    "git_commit",
    "config_sha256",
    "hardware_type",
    "solver",
    "n_particles",
    "steps",
    "dt",
    "softening",
    "tree_theta",
    "tree_leaf_capacity",
    "fmm_expansion_order",
    "output_format",
    "median_step_time",
    "total_wall_time",
    "particle_steps_per_second",
    "energy_drift_final",
    "momentum_drift_final",
    "max_energy_drift",
    "max_momentum_drift",
    "git_branch",
    "git_dirty",
    "build_type",
    "compiler",
    "compiler_version",
    "cuda_available",
    "cuda_device_name",
    "mpi_enabled",
    "rank_count",
    "hostname",
    "timestamp_utc",
    "config_path",
    "output_dir",
    "log_path",
]


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def vector_norm(row: dict[str, str], columns: tuple[str, str, str]) -> float:
    return math.sqrt(sum(safe_float(row.get(column), 0.0) ** 2 for column in columns))


def relative_scalar_drift(value: float, baseline: float) -> float:
    return abs(value - baseline) / max(abs(baseline), 1.0e-12)


def relative_vector_drift(
    row: dict[str, str],
    baseline: tuple[float, float, float],
    baseline_norm: float,
    columns: tuple[str, str, str],
) -> float:
    delta_squared = 0.0
    for column, baseline_value in zip(columns, baseline, strict=True):
        delta = safe_float(row.get(column), 0.0) - baseline_value
        delta_squared += delta * delta
    return math.sqrt(delta_squared) / max(baseline_norm, 1.0e-12)


def diagnostics_metrics(output_dir: Path) -> dict[str, float]:
    rows = read_csv_rows(output_dir / "diagnostics.csv")
    if not rows:
        return {
            "energy_drift_final": math.nan,
            "momentum_drift_final": math.nan,
            "max_energy_drift": math.nan,
            "max_momentum_drift": math.nan,
        }

    first = rows[0]
    last = rows[-1]
    initial_energy = safe_float(first.get("total_energy"))
    initial_momentum = (
        safe_float(first.get("momentum_x"), 0.0),
        safe_float(first.get("momentum_y"), 0.0),
        safe_float(first.get("momentum_z"), 0.0),
    )
    initial_momentum_norm = vector_norm(first, ("momentum_x", "momentum_y", "momentum_z"))

    energy_drifts = [
        relative_scalar_drift(safe_float(row.get("total_energy")), initial_energy) for row in rows
    ]
    momentum_drifts = [
        relative_vector_drift(
            row,
            initial_momentum,
            initial_momentum_norm,
            ("momentum_x", "momentum_y", "momentum_z"),
        )
        for row in rows
    ]

    return {
        "energy_drift_final": relative_scalar_drift(safe_float(last.get("total_energy")), initial_energy),
        "momentum_drift_final": relative_vector_drift(
            last,
            initial_momentum,
            initial_momentum_norm,
            ("momentum_x", "momentum_y", "momentum_z"),
        ),
        "max_energy_drift": max(energy_drifts) if energy_drifts else math.nan,
        "max_momentum_drift": max(momentum_drifts) if momentum_drifts else math.nan,
    }


def existing_summary_seconds(output_root: Path) -> dict[str, float]:
    seconds: dict[str, float] = {}
    for row in read_csv_rows(output_root / "sweep_summary.csv"):
        run_id = row.get("run_id", "")
        if run_id:
            seconds[run_id] = safe_float(row.get("seconds"), 0.0)
    return seconds


def hardware_type(metadata: dict[str, Any], solver: str) -> str:
    if solver.startswith("cuda") or bool(metadata.get("cuda_available")):
        return "cuda" if bool(metadata.get("cuda_available")) else "cpu-fallback"
    if bool(metadata.get("mpi_enabled")) or safe_int(metadata.get("rank_count"), 1) > 1:
        return "mpi"
    return "cpu"


def make_dataset_row(
    run: sweep_runner.SweepRun,
    result: sweep_runner.SweepResult,
    summary_seconds: dict[str, float],
) -> dict[str, Any]:
    metadata = result.metadata or read_json(result.output_dir / "metadata.json")
    config = run.config
    simulation = config.get("simulation", {})
    physics = config.get("physics", {})
    output = config.get("output", {})
    solver = str(simulation.get("solver", metadata.get("solver", "")))
    steps = safe_int(simulation.get("steps", metadata.get("steps")), 0)
    n_particles = safe_int(simulation.get("n_particles", metadata.get("particle_count")), 0)

    total_wall_time = result.seconds
    if total_wall_time <= 0.0:
        total_wall_time = summary_seconds.get(result.run_id, 0.0)

    particle_steps = n_particles * max(steps, 0)
    particle_steps_per_second = (
        particle_steps / total_wall_time if total_wall_time > 0.0 and particle_steps > 0 else math.nan
    )
    drift = diagnostics_metrics(result.output_dir)

    row: dict[str, Any] = {
        "run_id": result.run_id,
        "status": result.status,
        "exit_code": result.exit_code,
        "error": result.error,
        "git_commit": metadata.get("git_commit", ""),
        "config_sha256": metadata.get("config_sha256", ""),
        "hardware_type": hardware_type(metadata, solver),
        "solver": solver,
        "n_particles": n_particles,
        "steps": steps,
        "dt": safe_float(simulation.get("dt", metadata.get("dt"))),
        "softening": safe_float(physics.get("softening", metadata.get("softening"))),
        "tree_theta": safe_float(simulation.get("tree_theta", metadata.get("tree_theta"))),
        "tree_leaf_capacity": safe_int(
            simulation.get("tree_leaf_capacity", metadata.get("tree_leaf_capacity")),
            0,
        ),
        "fmm_expansion_order": safe_int(
            simulation.get("fmm_expansion_order", metadata.get("fmm_expansion_order")),
            0,
        ),
        "output_format": output.get("format", metadata.get("output_format", "")),
        "median_step_time": total_wall_time / steps if total_wall_time > 0.0 and steps > 0 else math.nan,
        "total_wall_time": total_wall_time if total_wall_time > 0.0 else math.nan,
        "particle_steps_per_second": particle_steps_per_second,
        "git_branch": metadata.get("git_branch", ""),
        "git_dirty": metadata.get("git_dirty", ""),
        "build_type": metadata.get("build_type", ""),
        "compiler": metadata.get("compiler", ""),
        "compiler_version": metadata.get("compiler_version", ""),
        "cuda_available": metadata.get("cuda_available", ""),
        "cuda_device_name": metadata.get("cuda_device_name", ""),
        "mpi_enabled": metadata.get("mpi_enabled", ""),
        "rank_count": metadata.get("rank_count", ""),
        "hostname": metadata.get("hostname", ""),
        "timestamp_utc": metadata.get("timestamp_utc", ""),
        "config_path": result.config_path.as_posix(),
        "output_dir": result.output_dir.as_posix(),
        "log_path": result.log_path.as_posix(),
    }
    row.update(drift)
    return row


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            pd.DataFrame(rows, columns=SOLVER_TUNING_COLUMNS).to_parquet(
                path,
                index=False,
                engine="pyarrow",
            )
        except ImportError as exc:
            raise RuntimeError(
                "Parquet dataset output requires pandas and pyarrow. Install project "
                "dependencies or use a .csv output path for local smoke tests."
            ) from exc
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOLVER_TUNING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, args: argparse.Namespace, output_root: Path, rows: list[dict[str, Any]]) -> None:
    status_counts = {status: sum(1 for row in rows if row["status"] == status) for status in {"completed", "failed", "planned"}}
    payload = {
        "dataset_type": "solver_tuning",
        "sweep": str(args.sweep),
        "sweep_output_root": output_root.as_posix(),
        "dataset_path": args.output.as_posix(),
        "row_count": len(rows),
        "status_counts": status_counts,
        "columns": SOLVER_TUNING_COLUMNS,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_or_plan_sweep(
    args: argparse.Namespace,
) -> tuple[Path, list[sweep_runner.SweepRun], list[sweep_runner.SweepResult], dict[str, float]]:
    sweep_path = args.sweep.resolve()
    sweep = sweep_runner.load_sweep_yaml(sweep_path)
    output_root, runs = sweep_runner.build_runs(sweep, sweep_path)
    output_root.mkdir(parents=True, exist_ok=True)
    prior_summary_seconds = existing_summary_seconds(output_root)
    if args.limit is not None:
        runs = runs[: args.limit]

    executable = sweep_runner.resolve_executable(args.executable or sweep.get("executable"))
    if args.dry_run:
        for run in runs:
            sweep_runner.write_toml(run.config_path, run.config)
        return output_root, runs, [sweep_runner.planned_result(run) for run in runs], prior_summary_seconds

    results: list[sweep_runner.SweepResult] = []
    max_workers = max(1, int(args.jobs or sweep.get("jobs", 1)))
    if max_workers == 1:
        for run in runs:
            result = sweep_runner.execute_run(executable, run, args.resume)
            results.append(result)
            print(f"{result.status:>9} {result.run_id} {result.seconds:.3f}s")
            if result.status == "failed" and args.stop_on_failure:
                break
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {
                executor.submit(sweep_runner.execute_run, executable, run, args.resume): run
                for run in runs
            }
            for future in as_completed(future_to_run):
                result = future.result()
                results.append(result)
                print(f"{result.status:>9} {result.run_id} {result.seconds:.3f}s")
                if result.status == "failed" and args.stop_on_failure:
                    break
        results.sort(key=lambda item: item.run_id)

    sweep_runner.write_csv_summary(output_root / "sweep_summary.csv", results)
    sweep_runner.write_parquet_summary(output_root / "sweep_summary.parquet", results)
    sweep_runner.write_sweep_metadata(output_root / "sweep_metadata.json", sweep_path, results, args.dry_run)
    return output_root, runs, results, prior_summary_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root, runs, results, summary_seconds = run_or_plan_sweep(args)
    current_summary_seconds = existing_summary_seconds(output_root)
    for run_id, seconds in current_summary_seconds.items():
        if seconds > 0.0:
            summary_seconds[run_id] = seconds
    run_by_id = {run.run_id: run for run in runs}
    rows = [
        make_dataset_row(run_by_id[result.run_id], result, summary_seconds)
        for result in results
        if result.run_id in run_by_id
    ]
    write_dataset(args.output, rows)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    write_manifest(manifest_path, args, output_root, rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
