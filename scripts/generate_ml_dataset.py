"""Generate versioned ML-ready datasets from reproducible simulator sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


DATASET_SCHEMA_VERSION = "0.1.0"

SOLVER_TUNING_COLUMNS = [
    "dataset_schema_version",
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

FORCE_ERROR_COLUMNS = [
    "dataset_schema_version",
    "run_id",
    "direct_run_id",
    "status",
    "error",
    "git_commit",
    "config_sha256",
    "direct_config_sha256",
    "solver",
    "n_particles",
    "tree_theta",
    "tree_leaf_capacity",
    "fmm_expansion_order",
    "softening",
    "force_rmse",
    "force_mae",
    "force_max_error",
    "relative_force_rmse",
    "runtime_direct",
    "runtime_approx",
    "speedup_vs_direct",
    "config_path",
    "direct_config_path",
    "output_dir",
    "direct_output_dir",
]

PER_STEP_DIAGNOSTICS_COLUMNS = [
    "dataset_schema_version",
    "run_id",
    "status",
    "git_commit",
    "config_sha256",
    "solver",
    "n_particles",
    "step",
    "time",
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "linear_momentum_x",
    "linear_momentum_y",
    "linear_momentum_z",
    "angular_momentum_x",
    "angular_momentum_y",
    "angular_momentum_z",
    "step_wall_time",
    "config_path",
    "output_dir",
]

DATASET_SCHEMAS = {
    "solver_tuning": {
        "columns": SOLVER_TUNING_COLUMNS,
        "required": [
            "dataset_schema_version",
            "run_id",
            "git_commit",
            "config_sha256",
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
        ],
    },
    "force_error": {
        "columns": FORCE_ERROR_COLUMNS,
        "required": [
            "dataset_schema_version",
            "run_id",
            "direct_run_id",
            "git_commit",
            "config_sha256",
            "direct_config_sha256",
            "solver",
            "n_particles",
            "tree_theta",
            "tree_leaf_capacity",
            "fmm_expansion_order",
            "softening",
            "force_rmse",
            "force_mae",
            "force_max_error",
            "relative_force_rmse",
            "runtime_direct",
            "runtime_approx",
            "speedup_vs_direct",
        ],
    },
    "per_step_diagnostics": {
        "columns": PER_STEP_DIAGNOSTICS_COLUMNS,
        "required": [
            "dataset_schema_version",
            "run_id",
            "git_commit",
            "config_sha256",
            "solver",
            "n_particles",
            "step",
            "time",
            "kinetic_energy",
            "potential_energy",
            "total_energy",
            "linear_momentum_x",
            "linear_momentum_y",
            "linear_momentum_z",
            "angular_momentum_x",
            "angular_momentum_y",
            "angular_momentum_z",
            "step_wall_time",
        ],
    },
}


@dataclass(frozen=True)
class DatasetArtifact:
    dataset_type: str
    clean_path: Path
    raw_path: Path
    summary_path: Path
    manifest_path: Path
    raw_rows: int
    clean_rows: int
    missing_counts: dict[str, int]


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


def is_missing(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() == "nan":
        return True
    return False


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


def run_metadata(result: sweep_runner.SweepResult) -> dict[str, Any]:
    return result.metadata or read_json(result.output_dir / "metadata.json")


def run_seconds(result: sweep_runner.SweepResult, summary_seconds: dict[str, float]) -> float:
    if result.seconds > 0.0:
        return result.seconds
    return summary_seconds.get(result.run_id, 0.0)


def run_settings(run: sweep_runner.SweepRun, result: sweep_runner.SweepResult) -> dict[str, Any]:
    metadata = run_metadata(result)
    simulation = run.config.get("simulation", {})
    physics = run.config.get("physics", {})
    output = run.config.get("output", {})
    return {
        "solver": str(simulation.get("solver", metadata.get("solver", ""))),
        "n_particles": safe_int(simulation.get("n_particles", metadata.get("particle_count")), 0),
        "steps": safe_int(simulation.get("steps", metadata.get("steps")), 0),
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
    }


def settings_key(settings: dict[str, Any]) -> tuple[Any, ...]:
    return (
        settings["n_particles"],
        settings["tree_theta"],
        settings["tree_leaf_capacity"],
        settings["fmm_expansion_order"],
        settings["softening"],
    )


def make_solver_tuning_rows(
    runs: list[sweep_runner.SweepRun],
    results: list[sweep_runner.SweepResult],
    summary_seconds: dict[str, float],
) -> list[dict[str, Any]]:
    run_by_id = {run.run_id: run for run in runs}
    rows: list[dict[str, Any]] = []
    for result in results:
        run = run_by_id.get(result.run_id)
        if run is None:
            continue
        metadata = run_metadata(result)
        settings = run_settings(run, result)
        total_wall_time = run_seconds(result, summary_seconds)
        particle_steps = settings["n_particles"] * max(settings["steps"], 0)
        drift = diagnostics_metrics(result.output_dir)
        row: dict[str, Any] = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "run_id": result.run_id,
            "status": result.status,
            "exit_code": result.exit_code,
            "error": result.error,
            "git_commit": metadata.get("git_commit", ""),
            "config_sha256": metadata.get("config_sha256", ""),
            "hardware_type": hardware_type(metadata, settings["solver"]),
            "solver": settings["solver"],
            "n_particles": settings["n_particles"],
            "steps": settings["steps"],
            "dt": settings["dt"],
            "softening": settings["softening"],
            "tree_theta": settings["tree_theta"],
            "tree_leaf_capacity": settings["tree_leaf_capacity"],
            "fmm_expansion_order": settings["fmm_expansion_order"],
            "output_format": settings["output_format"],
            "median_step_time": (
                total_wall_time / settings["steps"]
                if total_wall_time > 0.0 and settings["steps"] > 0
                else math.nan
            ),
            "total_wall_time": total_wall_time if total_wall_time > 0.0 else math.nan,
            "particle_steps_per_second": (
                particle_steps / total_wall_time
                if total_wall_time > 0.0 and particle_steps > 0
                else math.nan
            ),
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
        rows.append(row)
    return rows


def first_snapshot_path(output_dir: Path) -> Path | None:
    for suffix in (".csv", ".parquet"):
        path = output_dir / f"snapshot_000000{suffix}"
        if path.exists():
            return path
    return None


def load_snapshot_accelerations(path: Path) -> dict[int, tuple[float, float, float]]:
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading Parquet snapshots requires pandas and pyarrow") from exc
        frame = pd.read_parquet(path, engine="pyarrow")
        return {
            int(row.id): (float(row.ax), float(row.ay), float(row.az))
            for row in frame.itertuples(index=False)
        }

    with path.open("r", encoding="utf-8", newline="") as handle:
        first = handle.readline()
        if not first.startswith("#"):
            handle.seek(0)
        reader = csv.DictReader(handle)
        return {
            safe_int(row["id"]): (
                safe_float(row.get("ax"), 0.0),
                safe_float(row.get("ay"), 0.0),
                safe_float(row.get("az"), 0.0),
            )
            for row in reader
        }


def compare_force_snapshots(
    direct_path: Path,
    approx_path: Path,
) -> tuple[float, float, float, float]:
    direct = load_snapshot_accelerations(direct_path)
    approx = load_snapshot_accelerations(approx_path)
    common_ids = sorted(set(direct) & set(approx))
    if not common_ids or len(common_ids) != len(direct) or len(common_ids) != len(approx):
        raise RuntimeError(f"Snapshot particle IDs do not match: {direct_path} vs {approx_path}")

    squared_errors = []
    abs_errors = []
    reference_squared = []
    for particle_id in common_ids:
        dx = approx[particle_id][0] - direct[particle_id][0]
        dy = approx[particle_id][1] - direct[particle_id][1]
        dz = approx[particle_id][2] - direct[particle_id][2]
        error_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        reference_norm2 = (
            direct[particle_id][0] * direct[particle_id][0]
            + direct[particle_id][1] * direct[particle_id][1]
            + direct[particle_id][2] * direct[particle_id][2]
        )
        squared_errors.append(error_norm * error_norm)
        abs_errors.append(error_norm)
        reference_squared.append(reference_norm2)

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    mae = sum(abs_errors) / len(abs_errors)
    max_error = max(abs_errors)
    reference_rmse = math.sqrt(sum(reference_squared) / len(reference_squared))
    relative_rmse = rmse / max(reference_rmse, 1.0e-12)
    return rmse, mae, max_error, relative_rmse


def make_force_error_rows(
    runs: list[sweep_runner.SweepRun],
    results: list[sweep_runner.SweepResult],
    summary_seconds: dict[str, float],
) -> list[dict[str, Any]]:
    run_by_id = {run.run_id: run for run in runs}
    result_by_id = {result.run_id: result for result in results}
    direct_refs: dict[tuple[int, tuple[Any, ...]], sweep_runner.SweepResult] = {}
    fallback_refs: dict[tuple[Any, ...], sweep_runner.SweepResult] = {}

    for result in results:
        run = run_by_id.get(result.run_id)
        if run is None or result.status != "completed":
            continue
        settings = run_settings(run, result)
        if settings["solver"] == "direct":
            key = settings_key(settings)
            direct_refs[(result.repetition, key)] = result
            fallback_refs.setdefault(key, result)

    rows: list[dict[str, Any]] = []
    for result in results:
        run = run_by_id.get(result.run_id)
        if run is None:
            continue
        settings = run_settings(run, result)
        if settings["solver"] == "direct":
            continue

        metadata = run_metadata(result)
        key = settings_key(settings)
        direct_result = direct_refs.get((result.repetition, key)) or fallback_refs.get(key)
        row: dict[str, Any] = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "run_id": result.run_id,
            "direct_run_id": direct_result.run_id if direct_result else "",
            "status": result.status,
            "error": result.error,
            "git_commit": metadata.get("git_commit", ""),
            "config_sha256": metadata.get("config_sha256", ""),
            "direct_config_sha256": "",
            "solver": settings["solver"],
            "n_particles": settings["n_particles"],
            "tree_theta": settings["tree_theta"],
            "tree_leaf_capacity": settings["tree_leaf_capacity"],
            "fmm_expansion_order": settings["fmm_expansion_order"],
            "softening": settings["softening"],
            "force_rmse": math.nan,
            "force_mae": math.nan,
            "force_max_error": math.nan,
            "relative_force_rmse": math.nan,
            "runtime_direct": math.nan,
            "runtime_approx": run_seconds(result, summary_seconds),
            "speedup_vs_direct": math.nan,
            "config_path": result.config_path.as_posix(),
            "direct_config_path": "",
            "output_dir": result.output_dir.as_posix(),
            "direct_output_dir": "",
        }

        if direct_result is None:
            row["status"] = "missing_reference"
            row["error"] = "no completed direct run with matching solver settings"
            rows.append(row)
            continue

        direct_metadata = run_metadata(direct_result)
        direct_snapshot = first_snapshot_path(direct_result.output_dir)
        approx_snapshot = first_snapshot_path(result.output_dir)
        row["direct_config_sha256"] = direct_metadata.get("config_sha256", "")
        row["direct_config_path"] = direct_result.config_path.as_posix()
        row["direct_output_dir"] = direct_result.output_dir.as_posix()
        row["runtime_direct"] = run_seconds(direct_result, summary_seconds)

        if direct_snapshot is None or approx_snapshot is None:
            row["status"] = "missing_snapshot"
            row["error"] = "missing snapshot_000000 for direct or approximate run"
            rows.append(row)
            continue

        try:
            rmse, mae, max_error, relative_rmse = compare_force_snapshots(
                direct_snapshot,
                approx_snapshot,
            )
            row["force_rmse"] = rmse
            row["force_mae"] = mae
            row["force_max_error"] = max_error
            row["relative_force_rmse"] = relative_rmse
            if row["runtime_approx"] > 0.0:
                row["speedup_vs_direct"] = row["runtime_direct"] / row["runtime_approx"]
        except RuntimeError as exc:
            row["status"] = "failed_comparison"
            row["error"] = str(exc)
        rows.append(row)
    return rows


def make_per_step_diagnostics_rows(
    runs: list[sweep_runner.SweepRun],
    results: list[sweep_runner.SweepResult],
    summary_seconds: dict[str, float],
) -> list[dict[str, Any]]:
    run_by_id = {run.run_id: run for run in runs}
    rows: list[dict[str, Any]] = []
    for result in results:
        run = run_by_id.get(result.run_id)
        if run is None:
            continue
        metadata = run_metadata(result)
        settings = run_settings(run, result)
        diagnostics = read_csv_rows(result.output_dir / "diagnostics.csv")
        average_step_wall_time = (
            run_seconds(result, summary_seconds) / settings["steps"]
            if run_seconds(result, summary_seconds) > 0.0 and settings["steps"] > 0
            else math.nan
        )

        if not diagnostics:
            rows.append(
                {
                    "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    "run_id": result.run_id,
                    "status": "missing_diagnostics",
                    "git_commit": metadata.get("git_commit", ""),
                    "config_sha256": metadata.get("config_sha256", ""),
                    "solver": settings["solver"],
                    "n_particles": settings["n_particles"],
                    "step": math.nan,
                    "time": math.nan,
                    "kinetic_energy": math.nan,
                    "potential_energy": math.nan,
                    "total_energy": math.nan,
                    "linear_momentum_x": math.nan,
                    "linear_momentum_y": math.nan,
                    "linear_momentum_z": math.nan,
                    "angular_momentum_x": math.nan,
                    "angular_momentum_y": math.nan,
                    "angular_momentum_z": math.nan,
                    "step_wall_time": average_step_wall_time,
                    "config_path": result.config_path.as_posix(),
                    "output_dir": result.output_dir.as_posix(),
                }
            )
            continue

        for diagnostic in diagnostics:
            rows.append(
                {
                    "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    "run_id": result.run_id,
                    "status": result.status,
                    "git_commit": metadata.get("git_commit", ""),
                    "config_sha256": metadata.get("config_sha256", ""),
                    "solver": settings["solver"],
                    "n_particles": settings["n_particles"],
                    "step": safe_int(diagnostic.get("step")),
                    "time": safe_float(diagnostic.get("time")),
                    "kinetic_energy": safe_float(diagnostic.get("kinetic_energy")),
                    "potential_energy": safe_float(diagnostic.get("potential_energy")),
                    "total_energy": safe_float(diagnostic.get("total_energy")),
                    "linear_momentum_x": safe_float(diagnostic.get("momentum_x")),
                    "linear_momentum_y": safe_float(diagnostic.get("momentum_y")),
                    "linear_momentum_z": safe_float(diagnostic.get("momentum_z")),
                    "angular_momentum_x": safe_float(diagnostic.get("angular_momentum_x")),
                    "angular_momentum_y": safe_float(diagnostic.get("angular_momentum_y")),
                    "angular_momentum_z": safe_float(diagnostic.get("angular_momentum_z")),
                    "step_wall_time": average_step_wall_time,
                    "config_path": result.config_path.as_posix(),
                    "output_dir": result.output_dir.as_posix(),
                }
            )
    return rows


def output_path_for(base: Path, dataset_type: str, all_mode: bool) -> Path:
    if not all_mode:
        return base
    if base.suffix:
        return base.with_name(f"{base.stem}.{dataset_type}{base.suffix}")
    return base / f"{dataset_type}.csv"


def raw_path_for(clean_path: Path) -> Path:
    return clean_path.with_name(f"{clean_path.stem}.raw{clean_path.suffix}")


def summary_path_for(clean_path: Path) -> Path:
    return clean_path.with_suffix(clean_path.suffix + ".summary.md")


def manifest_path_for(clean_path: Path) -> Path:
    return clean_path.with_suffix(clean_path.suffix + ".manifest.json")


def write_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            pd.DataFrame(rows, columns=columns).to_parquet(path, index=False, engine="pyarrow")
        except ImportError as exc:
            raise RuntimeError(
                "Parquet dataset output requires pandas and pyarrow. Install project "
                "dependencies or use a .csv output path for local smoke tests."
            ) from exc
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def missing_counts(rows: list[dict[str, Any]], required_columns: list[str]) -> dict[str, int]:
    return {
        column: sum(1 for row in rows if is_missing(row.get(column))) for column in required_columns
    }


def clean_rows(dataset_type: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    schema = DATASET_SCHEMAS[dataset_type]
    required = schema["required"]
    counts = missing_counts(rows, required)
    cleaned = [
        row
        for row in rows
        if row.get("status") == "completed"
        and all(not is_missing(row.get(column)) for column in required)
    ]
    return cleaned, counts


def write_summary(
    path: Path,
    dataset_type: str,
    raw_rows: list[dict[str, Any]],
    cleaned_rows: list[dict[str, Any]],
    counts: dict[str, int],
    raw_path: Path,
    clean_path: Path,
) -> None:
    status_counts: dict[str, int] = {}
    for row in raw_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    missing_lines = [
        f"| `{column}` | {count} |" for column, count in counts.items() if count > 0
    ]
    if not missing_lines:
        missing_lines = ["| none | 0 |"]

    lines = [
        f"# {dataset_type.replace('_', ' ').title()} Dataset",
        "",
        f"- Schema version: `{DATASET_SCHEMA_VERSION}`",
        f"- Raw rows: {len(raw_rows)}",
        f"- Cleaned rows: {len(cleaned_rows)}",
        f"- Raw dataset: `{raw_path.as_posix()}`",
        f"- Cleaned dataset: `{clean_path.as_posix()}`",
        "",
        "## Status Counts",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ]
    lines.extend(f"| `{status}` | {count} |" for status, count in sorted(status_counts.items()))
    lines.extend(
        [
            "",
            "## Missing Required Values",
            "",
            "| Column | Missing rows |",
            "|---|---:|",
            *missing_lines,
            "",
            "## Columns",
            "",
            ", ".join(f"`{column}`" for column in DATASET_SCHEMAS[dataset_type]["columns"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    output_root: Path,
    artifact: DatasetArtifact,
) -> None:
    payload = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_type": artifact.dataset_type,
        "sweep": str(args.sweep),
        "sweep_output_root": output_root.as_posix(),
        "raw_dataset_path": artifact.raw_path.as_posix(),
        "clean_dataset_path": artifact.clean_path.as_posix(),
        "summary_path": artifact.summary_path.as_posix(),
        "raw_row_count": artifact.raw_rows,
        "clean_row_count": artifact.clean_rows,
        "missing_required_values": artifact.missing_counts,
        "columns": DATASET_SCHEMAS[artifact.dataset_type]["columns"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def materialize_dataset(
    dataset_type: str,
    raw_rows: list[dict[str, Any]],
    clean_path: Path,
    args: argparse.Namespace,
    output_root: Path,
) -> DatasetArtifact:
    schema = DATASET_SCHEMAS[dataset_type]
    raw_path = raw_path_for(clean_path)
    summary_path = summary_path_for(clean_path)
    manifest_path = manifest_path_for(clean_path)
    cleaned, counts = clean_rows(dataset_type, raw_rows)
    write_table(raw_path, raw_rows, schema["columns"])
    write_table(clean_path, cleaned, schema["columns"])
    write_summary(summary_path, dataset_type, raw_rows, cleaned, counts, raw_path, clean_path)
    artifact = DatasetArtifact(
        dataset_type=dataset_type,
        clean_path=clean_path,
        raw_path=raw_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        raw_rows=len(raw_rows),
        clean_rows=len(cleaned),
        missing_counts=counts,
    )
    write_manifest(manifest_path, args, output_root, artifact)
    return artifact


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
    parser.add_argument(
        "--dataset-type",
        choices=["solver_tuning", "force_error", "per_step_diagnostics", "all"],
        default="solver_tuning",
    )
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

    dataset_types = (
        ["solver_tuning", "force_error", "per_step_diagnostics"]
        if args.dataset_type == "all"
        else [args.dataset_type]
    )
    builders = {
        "solver_tuning": make_solver_tuning_rows,
        "force_error": make_force_error_rows,
        "per_step_diagnostics": make_per_step_diagnostics_rows,
    }

    artifacts: list[DatasetArtifact] = []
    all_mode = args.dataset_type == "all"
    for dataset_type in dataset_types:
        clean_path = output_path_for(args.output, dataset_type, all_mode)
        rows = builders[dataset_type](runs, results, summary_seconds)
        artifact = materialize_dataset(dataset_type, rows, clean_path, args, output_root)
        artifacts.append(artifact)
        print(f"Wrote {artifact.clean_path}")
        print(f"Wrote {artifact.raw_path}")
        print(f"Wrote {artifact.summary_path}")
        print(f"Wrote {artifact.manifest_path}")


if __name__ == "__main__":
    main()
