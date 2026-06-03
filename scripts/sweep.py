"""Run YAML-defined simulator parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.experiment_utils import (
    benchmark_env as experiment_benchmark_env,
    get_dotted,
    load_toml,
    read_diagnostics_summary as experiment_read_diagnostics,
    read_metadata as experiment_read_metadata,
    resolve_simulator_executable,
    run_simulator,
    set_dotted,
    sync_galaxy_particle_counts,
    write_toml,
)


SUMMARY_FIELD_ORDER = [
    "run_id",
    "repetition",
    "status",
    "exit_code",
    "seconds",
    "config_path",
    "output_dir",
    "log_path",
    "error",
]


@dataclass(frozen=True)
class SweepRun:
    run_id: str
    repetition: int
    parameters: dict[str, Any]
    config: dict[str, Any]
    config_path: Path
    output_dir: Path
    log_path: Path


@dataclass(frozen=True)
class SweepResult:
    run_id: str
    repetition: int
    parameters: dict[str, Any]
    config_path: Path
    output_dir: Path
    log_path: Path
    status: str
    exit_code: int | None
    seconds: float
    metadata: dict[str, Any]
    diagnostics: dict[str, float]
    error: str


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_sweep_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Sweep file must contain a YAML mapping: {path}")
        return loaded
    except ImportError:
        return load_simple_yaml(path)


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the simple mapping/list subset used by repository sweep configs."""
    result: dict[str, Any] = {}
    current_mapping: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            if current_mapping is None:
                raise ValueError(f"Indented YAML entry without a parent mapping: {raw_line}")
            key, separator, value = line.strip().partition(":")
            if not separator:
                raise ValueError(f"Invalid YAML mapping entry: {raw_line}")
            current_mapping[key.strip()] = parse_scalar(value.strip())
            continue

        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML entry: {raw_line}")
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = parse_scalar(value)
            current_mapping = None
        else:
            result[key] = {}
            current_mapping = result[key]
    return result


def sweep_values(parameters: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    keys = list(parameters)
    value_lists = []
    for key in keys:
        values = parameters[key]
        if not isinstance(values, list):
            values = [values]
        value_lists.append(values)

    rows: list[tuple[dict[str, Any], str]] = []
    for combination in itertools.product(*value_lists):
        parameter_values = dict(zip(keys, combination, strict=True))
        label_parts = []
        for key, value in parameter_values.items():
            safe_key = key.replace(".", "_")
            safe_value = str(value).replace(".", "p").replace("-", "m").replace("/", "_")
            label_parts.append(f"{safe_key}-{safe_value}")
        rows.append((parameter_values, "__".join(label_parts)))
    return rows


def resolve_executable(value: str | None) -> Path:
    return resolve_simulator_executable(value)


def apply_sweep_overrides(
    config: dict[str, Any],
    sweep: dict[str, Any],
    output_dir: Path,
    run_name: str,
) -> None:
    if "steps" in sweep:
        set_dotted(config, "simulation.steps", sweep["steps"])
    if "output_format" in sweep:
        set_dotted(config, "output.format", sweep["output_format"])
    set_dotted(config, "output.directory", output_dir.as_posix())
    current_name = get_dotted(config, "simulation.name", "sweep")
    set_dotted(config, "simulation.name", f"{current_name}_{run_name}")


def build_runs(sweep: dict[str, Any], grid_path: Path) -> tuple[Path, list[SweepRun]]:
    base_config_path = Path(str(sweep["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = (grid_path.parent / base_config_path).resolve()
        if not base_config_path.exists():
            base_config_path = (REPO_ROOT / str(sweep["base_config"])).resolve()
    base_config = load_toml(base_config_path)
    output_root = Path(str(sweep.get("output_root", "experiments/sweeps/default")))
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    repetitions = int(sweep.get("repetitions", 1))
    parameters = sweep.get("parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Sweep config must define a non-empty parameters mapping")

    runs: list[SweepRun] = []
    for case_index, (parameter_values, label) in enumerate(sweep_values(parameters), start=1):
        for repetition in range(1, repetitions + 1):
            run_id = f"case_{case_index:05d}_r{repetition:02d}"
            run_name = f"{run_id}_{label}"
            config = deepcopy(base_config)
            for key, value in parameter_values.items():
                set_dotted(config, key, value)
            output_dir = output_root / "runs" / run_id
            apply_sweep_overrides(config, sweep, output_dir, run_id)
            sync_galaxy_particle_counts(config)
            runs.append(
                SweepRun(
                    run_id=run_id,
                    repetition=repetition,
                    parameters=parameter_values,
                    config=config,
                    config_path=output_root / "configs" / f"{run_id}.toml",
                    output_dir=output_dir,
                    log_path=output_root / "logs" / f"{run_id}.log",
                )
            )
    return output_root, runs


def benchmark_env() -> dict[str, str]:
    return experiment_benchmark_env()


def read_metadata(output_dir: Path) -> dict[str, Any]:
    return experiment_read_metadata(output_dir)


def read_diagnostics(output_dir: Path) -> dict[str, float]:
    return experiment_read_diagnostics(output_dir)


def completed_result(run: SweepRun) -> SweepResult:
    return SweepResult(
        run_id=run.run_id,
        repetition=run.repetition,
        parameters=run.parameters,
        config_path=run.config_path,
        output_dir=run.output_dir,
        log_path=run.log_path,
        status="completed",
        exit_code=0,
        seconds=0.0,
        metadata=read_metadata(run.output_dir),
        diagnostics=read_diagnostics(run.output_dir),
        error="",
    )


def execute_run(executable: Path, run: SweepRun, resume: bool) -> SweepResult:
    if resume and (run.output_dir / "metadata.json").exists():
        return completed_result(run)

    write_toml(run.config_path, run.config)
    completed = run_simulator(
        executable,
        run.config_path,
        run.output_dir,
        cwd=REPO_ROOT,
        log_path=run.log_path,
        resume_marker=run.output_dir / "metadata.json" if resume else None,
    )
    status = "completed" if completed.exit_code == 0 else "failed"
    error = "" if completed.exit_code == 0 else f"exit code {completed.exit_code}"
    return SweepResult(
        run_id=run.run_id,
        repetition=run.repetition,
        parameters=run.parameters,
        config_path=run.config_path,
        output_dir=run.output_dir,
        log_path=run.log_path,
        status=status,
        exit_code=completed.exit_code,
        seconds=completed.seconds,
        metadata=completed.metadata,
        diagnostics=completed.diagnostics,
        error=error,
    )


def flatten_result(result: SweepResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": result.run_id,
        "repetition": result.repetition,
        "status": result.status,
        "exit_code": result.exit_code,
        "seconds": f"{result.seconds:.6f}",
        "config_path": result.config_path.as_posix(),
        "output_dir": result.output_dir.as_posix(),
        "log_path": result.log_path.as_posix(),
        "error": result.error,
    }
    for key, value in sorted(result.parameters.items()):
        row[f"param.{key}"] = value
    for key in (
        "git_commit",
        "git_branch",
        "git_dirty",
        "build_type",
        "compiler",
        "compiler_version",
        "config_sha256",
        "timestamp_utc",
        "hostname",
        "rank_count",
        "cuda_available",
        "cuda_device_name",
        "mpi_enabled",
    ):
        row[f"metadata.{key}"] = result.metadata.get(key, "")
    for key, value in result.diagnostics.items():
        row[f"diagnostics.{key}"] = value
    return row


def write_csv_summary(path: Path, results: list[SweepResult]) -> None:
    rows = [flatten_result(result) for result in results]
    fieldnames: list[str] = [field for field in SUMMARY_FIELD_ORDER if any(field in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet_summary(path: Path, results: list[SweepResult]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    rows = [flatten_result(result) for result in results]
    try:
        pd.DataFrame(rows).to_parquet(path, index=False, engine="pyarrow")
    except ImportError:
        return False
    return True


def write_sweep_metadata(path: Path, grid_path: Path, results: list[SweepResult], dry_run: bool) -> None:
    counts = {
        status: sum(1 for result in results if result.status == status)
        for status in {"planned", "completed", "failed"}
    }
    payload = {
        "grid_path": str(grid_path),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(),
        "dry_run": dry_run,
        "run_count": len(results),
        "status_counts": counts,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def planned_result(run: SweepRun) -> SweepResult:
    return SweepResult(
        run_id=run.run_id,
        repetition=run.repetition,
        parameters=run.parameters,
        config_path=run.config_path,
        output_dir=run.output_dir,
        log_path=run.log_path,
        status="planned",
        exit_code=None,
        seconds=0.0,
        metadata={},
        diagnostics={},
        error="",
    )


def run_sweep(args: argparse.Namespace) -> int:
    grid_path = args.grid.resolve()
    sweep = load_sweep_yaml(grid_path)
    executable = resolve_executable(args.executable or sweep.get("executable"))
    output_root, runs = build_runs(sweep, grid_path)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.limit is not None:
        runs = runs[: args.limit]

    if args.dry_run:
        for run in runs:
            write_toml(run.config_path, run.config)
        results = [planned_result(run) for run in runs]
    else:
        results = []
        max_workers = max(1, int(args.jobs or sweep.get("jobs", 1)))
        if max_workers == 1:
            for run in runs:
                result = execute_run(executable, run, args.resume)
                results.append(result)
                print(f"{result.status:>9} {run.run_id} {result.seconds:.3f}s")
                if result.status == "failed" and args.stop_on_failure:
                    break
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_run = {
                    executor.submit(execute_run, executable, run, args.resume): run for run in runs
                }
                for future in as_completed(future_to_run):
                    result = future.result()
                    results.append(result)
                    print(f"{result.status:>9} {result.run_id} {result.seconds:.3f}s")
                    if result.status == "failed" and args.stop_on_failure:
                        break
            results.sort(key=lambda result: result.run_id)

    write_csv_summary(output_root / "sweep_summary.csv", results)
    wrote_parquet = write_parquet_summary(output_root / "sweep_summary.parquet", results)
    write_sweep_metadata(output_root / "sweep_metadata.json", grid_path, results, args.dry_run)
    print(f"Wrote {output_root / 'sweep_summary.csv'}")
    if wrote_parquet:
        print(f"Wrote {output_root / 'sweep_summary.parquet'}")
    else:
        print("Skipped sweep_summary.parquet because pandas/pyarrow is unavailable")
    print(f"Wrote {output_root / 'sweep_metadata.json'}")
    return 1 if any(result.status == "failed" for result in results) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--executable", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N generated cases.")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run_sweep(parse_args()))


if __name__ == "__main__":
    main()
