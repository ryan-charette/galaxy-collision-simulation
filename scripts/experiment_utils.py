"""Shared helpers for simulator experiment scripts."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SimulatorRun:
    config_path: Path
    output_dir: Path
    seconds: float
    exit_code: int
    stdout: str
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float | None = None


def safe_float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    msys_runtime = Path("C:/msys64/ucrt64/bin")
    if msys_runtime.exists():
        env["PATH"] = f"{msys_runtime}{os.pathsep}{env.get('PATH', '')}"
    return env


def resolve_simulator_executable(value: str | Path | None) -> Path:
    executable = Path(value) if value else Path("build/fmm_galaxy_sim")
    if not executable.exists() and executable == Path("build/fmm_galaxy_sim"):
        for candidate in (Path("build/Release/fmm_galaxy_sim.exe"), Path("build/fmm_galaxy_sim.exe")):
            if candidate.exists():
                executable = candidate
                break
    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")
    return executable


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def write_toml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def emit_table(prefix: list[str], table: dict[str, Any]) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        child_items = [(key, value) for key, value in table.items() if isinstance(value, dict)]
        if prefix and scalar_items:
            lines.append(f"[{'.'.join(prefix)}]")
        for key, value in scalar_items:
            lines.append(f"{key} = {toml_value(value)}")
        if scalar_items or prefix:
            lines.append("")
        for key, child in child_items:
            emit_table([*prefix, key], child)

    for key, value in config.items():
        if isinstance(value, dict):
            emit_table([key], value)
        else:
            lines.append(f"{key} = {toml_value(value)}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        next_value = target.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a table")
        target = next_value
    target[parts[-1]] = value


def get_dotted(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    target: Any = config
    for part in dotted_key.split("."):
        if not isinstance(target, dict) or part not in target:
            return default
        target = target[part]
    return target


def sync_galaxy_particle_counts(config: dict[str, Any]) -> None:
    requested = get_dotted(config, "simulation.n_particles")
    galaxies = config.get("galaxy")
    if requested is None or not isinstance(galaxies, dict):
        return
    galaxy_items = [(name, galaxy) for name, galaxy in galaxies.items() if isinstance(galaxy, dict)]
    if not galaxy_items:
        return

    requested_count = int(requested)
    existing_counts = [max(int(galaxy.get("n_particles", 0)), 0) for _, galaxy in galaxy_items]
    existing_total = sum(existing_counts)
    if existing_total <= 0:
        base = requested_count // len(galaxy_items)
        counts = [base for _ in galaxy_items]
    else:
        raw_counts = [requested_count * count / existing_total for count in existing_counts]
        counts = [int(value) for value in raw_counts]
    remainder = requested_count - sum(counts)
    for index in range(remainder):
        counts[index % len(counts)] += 1

    for (_, galaxy), count in zip(galaxy_items, counts, strict=True):
        galaxy["n_particles"] = count


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_metadata(output_dir: Path) -> dict[str, Any]:
    return read_json_file(output_dir / "metadata.json")


def read_diagnostics_summary(output_dir: Path) -> dict[str, float]:
    path = output_dir / "diagnostics.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]

    def as_float(row: dict[str, str], key: str) -> float:
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return float("nan")

    initial_energy = as_float(first, "total_energy")
    final_energy = as_float(last, "total_energy")
    return {
        "initial_total_energy": initial_energy,
        "final_total_energy": final_energy,
        "energy_drift_abs": abs(final_energy - initial_energy),
        "final_momentum_norm": (
            as_float(last, "momentum_x") ** 2
            + as_float(last, "momentum_y") ** 2
            + as_float(last, "momentum_z") ** 2
        )
        ** 0.5,
        "final_angular_momentum_norm": (
            as_float(last, "angular_momentum_x") ** 2
            + as_float(last, "angular_momentum_y") ** 2
            + as_float(last, "angular_momentum_z") ** 2
        )
        ** 0.5,
    }


def _peak_memory_for_process(process: subprocess.Popen[str]) -> float | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None

    peak_memory_mb: float | None = None
    tracked = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            rss = tracked.memory_info().rss / (1024.0 * 1024.0)
            peak_memory_mb = max(peak_memory_mb or 0.0, rss)
        except psutil.Error:
            pass
        time.sleep(0.02)
    return peak_memory_mb


def run_simulator(
    executable: Path,
    config_path: Path,
    output_dir: Path,
    *,
    cwd: Path = REPO_ROOT,
    log_path: Path | None = None,
    resume_marker: Path | None = None,
    capture_memory: bool = False,
) -> SimulatorRun:
    if resume_marker is not None and resume_marker.exists():
        return SimulatorRun(
            config_path=config_path,
            output_dir=output_dir,
            seconds=0.0,
            exit_code=0,
            stdout="",
            metadata=read_metadata(output_dir),
            diagnostics=read_diagnostics_summary(output_dir),
        )

    command = [str(executable), "--config", str(config_path)]
    started = time.perf_counter()
    if capture_memory:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=benchmark_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        peak_memory_mb = _peak_memory_for_process(process)
        stdout, _ = process.communicate()
        exit_code = int(process.returncode or 0)
    else:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=benchmark_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stdout = completed.stdout
        exit_code = completed.returncode
        peak_memory_mb = None
    seconds = time.perf_counter() - started

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(shlex.quote(part) for part in command)}\n\n{stdout}",
            encoding="utf-8",
        )

    return SimulatorRun(
        config_path=config_path,
        output_dir=output_dir,
        seconds=seconds,
        exit_code=exit_code,
        stdout=stdout,
        metadata=read_metadata(output_dir),
        diagnostics=read_diagnostics_summary(output_dir),
        peak_memory_mb=peak_memory_mb,
    )


def write_two_galaxy_config(
    path: Path,
    *,
    name: str,
    solver: str,
    particles: int,
    steps: int,
    dt: float,
    snapshot_every: int,
    output: Path,
    output_format: str,
    theta: float,
    leaf_capacity: int,
    expansion_order: int,
    softening: float,
    seed: int,
    acceleration_dump: bool = False,
) -> None:
    half = particles // 2
    rest = particles - half
    acceleration_dump_line = "acceleration_dump = true\n" if acceleration_dump else ""
    config = f"""[simulation]
name = "{name}"
dim = 3
solver = "{solver}"
seed = {seed}
n_particles = {particles}
steps = {steps}
dt = {dt}
snapshot_every = {snapshot_every}
tree_theta = {theta}
tree_leaf_capacity = {leaf_capacity}
fmm_expansion_order = {expansion_order}

[physics]
G = 1.0
softening = {softening}

[galaxy.primary]
n_particles = {half}
mass = 1.0
radius = 0.85
position = [-0.72, -0.10, 0.06]
velocity = [0.34, 0.10, -0.015]
orientation = 0.25
group_id = 0
thickness = 0.045
inclination = 0.62

[galaxy.secondary]
n_particles = {rest}
mass = 1.0
radius = 0.85
position = [0.72, 0.10, -0.06]
velocity = [-0.34, -0.10, 0.015]
orientation = 3.42
group_id = 1
thickness = 0.045
inclination = -0.72

[output]
directory = "{output.as_posix()}"
format = "{output_format}"
{acceleration_dump_line}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config, encoding="utf-8")
