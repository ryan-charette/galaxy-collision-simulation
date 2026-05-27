"""Shared helpers for simulator experiment scripts."""

from __future__ import annotations

import os
from pathlib import Path


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

