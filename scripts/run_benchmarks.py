"""Run repeatable local benchmarks for the simulator executable."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCase:
    output_format: str
    solver: str
    particles: int
    steps: int
    replicate: int
    seconds: float
    git_commit: str
    config_sha256: str

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds

    @property
    def particle_steps_per_second(self) -> float:
        return (self.particles * self.steps) / self.seconds


def write_config(
    path: Path,
    solver: str,
    particles: int,
    steps: int,
    output: Path,
    output_format: str,
    theta: float,
    leaf_capacity: int,
    expansion_order: int,
) -> None:
    half = particles // 2
    rest = particles - half
    config = f"""[simulation]
name = "benchmark_{solver}_{particles}"
dim = 3
solver = "{solver}"
seed = 20260502
n_particles = {particles}
steps = {steps}
dt = 0.01
snapshot_every = {steps}
tree_theta = {theta}
tree_leaf_capacity = {leaf_capacity}
fmm_expansion_order = {expansion_order}

[physics]
G = 1.0
softening = 0.025

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
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config, encoding="utf-8")


def benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    msys_runtime = Path("C:/msys64/ucrt64/bin")
    if msys_runtime.exists():
        env["PATH"] = f"{msys_runtime}{os.pathsep}{env.get('PATH', '')}"
    return env


def run_case(
    executable: Path,
    solver: str,
    particles: int,
    steps: int,
    replicate: int,
    work_dir: Path,
    output_format: str,
    theta: float,
    leaf_capacity: int,
    expansion_order: int,
) -> BenchmarkCase:
    config_path = work_dir / "configs" / f"{output_format}_{solver}_{particles}_r{replicate}.toml"
    output_dir = work_dir / "outputs" / output_format / f"{solver}_{particles}_r{replicate}"
    write_config(
        config_path,
        solver,
        particles,
        steps,
        output_dir,
        output_format,
        theta,
        leaf_capacity,
        expansion_order,
    )

    command = [str(executable), "--config", str(config_path)]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=benchmark_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for solver={solver} particles={particles} replicate={replicate}\n"
            + completed.stdout
        )
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return BenchmarkCase(
        output_format,
        solver,
        particles,
        steps,
        replicate,
        seconds,
        metadata.get("git_commit", "unavailable"),
        metadata.get("config_sha256", "unavailable"),
    )


def write_csv(path: Path, results: list[BenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "solver",
                "output_format",
                "particles",
                "steps",
                "replicate",
                "seconds",
                "steps_per_second",
                "particle_steps_per_second",
                "git_commit",
                "config_sha256",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.solver,
                    result.output_format,
                    result.particles,
                    result.steps,
                    result.replicate,
                    f"{result.seconds:.6f}",
                    f"{result.steps_per_second:.6f}",
                    f"{result.particle_steps_per_second:.3f}",
                    result.git_commit,
                    result.config_sha256,
                ]
            )


def summarize(results: list[BenchmarkCase]) -> list[tuple[str, str, int, int, float, float, float, str]]:
    grouped: dict[tuple[str, str, int, int], list[BenchmarkCase]] = {}
    for result in results:
        grouped.setdefault((result.output_format, result.solver, result.particles, result.steps), []).append(result)

    rows = []
    solver_order = {
        "direct": 0,
        "cuda-direct": 1,
        "tree": 2,
        "cuda-tree": 3,
        "fmm": 4,
        "cuda-fmm": 5,
    }
    format_order = {"csv": 0, "parquet": 1, "none": 2}
    for (output_format, solver, particles, steps), cases in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][2],
            solver_order.get(item[0][1], 99),
            item[0][1],
            format_order.get(item[0][0], 99),
            item[0][0],
        ),
    ):
        seconds = [case.seconds for case in cases]
        median_seconds = statistics.median(seconds)
        steps_per_second = steps / median_seconds
        particle_steps_per_second = particles * steps_per_second
        commits = {case.git_commit for case in cases}
        git_commit = commits.pop() if len(commits) == 1 else "mixed"
        rows.append(
            (
                output_format,
                solver,
                particles,
                steps,
                median_seconds,
                steps_per_second,
                particle_steps_per_second,
                git_commit,
            )
        )
    return rows


def write_markdown(path: Path, results: list[BenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize(results)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Benchmark Results",
        "",
        f"Generated: {generated}",
        "",
        f"Platform: `{platform.platform()}`",
        "",
        "Build: Release executable. CUDA use depends on the selected solver and build configuration.",
        "",
        "| Output | Solver | Particles | Steps | Median wall time (s) | Steps/s | Particle-steps/s | Commit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for output_format, solver, particles, steps, seconds, steps_per_second, particle_steps_per_second, git_commit in rows:
        short_commit = git_commit[:12] if git_commit not in {"mixed", "unavailable"} else git_commit
        lines.append(
            f"| `{output_format}` | `{solver}` | {particles} | {steps} | {seconds:.3f} | "
            f"{steps_per_second:.2f} | {particle_steps_per_second:,.0f} | `{short_commit}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build-readme-gif/fmm_galaxy_sim.exe"))
    parser.add_argument("--work-dir", type=Path, default=Path("experiments/benchmarks/local_cpu"))
    parser.add_argument("--csv", type=Path, default=Path("docs/benchmarks/local_cpu_benchmark.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/benchmarks/local_cpu_benchmark.md"))
    parser.add_argument("--solvers", nargs="+", default=["direct", "tree", "fmm"])
    parser.add_argument("--particles", nargs="+", type=int, default=[250, 500, 1000])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-format", choices=["csv", "parquet", "none"], default="csv")
    parser.add_argument("--output-formats", nargs="+", choices=["csv", "parquet", "none"], default=None)
    parser.add_argument("--theta", type=float, default=0.58)
    parser.add_argument("--leaf-capacity", type=int, default=16)
    parser.add_argument("--expansion-order", type=int, choices=[0, 2, 4], default=4)
    args = parser.parse_args()

    if not args.executable.exists():
        raise FileNotFoundError(f"Executable not found: {args.executable}")

    results: list[BenchmarkCase] = []
    output_formats = args.output_formats if args.output_formats is not None else [args.output_format]
