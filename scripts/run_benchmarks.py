"""Run repeatable local CPU benchmarks for the simulator executable."""

from __future__ import annotations

import argparse
import csv
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
    solver: str
    particles: int
    steps: int
    replicate: int
    seconds: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds

    @property
    def particle_steps_per_second(self) -> float:
        return (self.particles * self.steps) / self.seconds


def write_config(path: Path, solver: str, particles: int, steps: int, output: Path) -> None:
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
tree_theta = 0.58
tree_leaf_capacity = 16
fmm_expansion_order = 4

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
format = "csv"
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
) -> BenchmarkCase:
    config_path = work_dir / "configs" / f"{solver}_{particles}_r{replicate}.toml"
    output_dir = work_dir / "outputs" / f"{solver}_{particles}_r{replicate}"
    write_config(config_path, solver, particles, steps, output_dir)

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
    return BenchmarkCase(solver, particles, steps, replicate, seconds)


def write_csv(path: Path, results: list[BenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "solver",
                "particles",
                "steps",
                "replicate",
                "seconds",
                "steps_per_second",
                "particle_steps_per_second",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.solver,
                    result.particles,
                    result.steps,
                    result.replicate,
                    f"{result.seconds:.6f}",
                    f"{result.steps_per_second:.6f}",
                    f"{result.particle_steps_per_second:.3f}",
                ]
            )


def summarize(results: list[BenchmarkCase]) -> list[tuple[str, int, int, float, float, float]]:
    grouped: dict[tuple[str, int, int], list[BenchmarkCase]] = {}
    for result in results:
        grouped.setdefault((result.solver, result.particles, result.steps), []).append(result)

    rows = []
    solver_order = {"direct": 0, "tree": 1, "fmm": 2}
    for (solver, particles, steps), cases in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], solver_order.get(item[0][0], 99), item[0][0]),
    ):
        seconds = [case.seconds for case in cases]
        median_seconds = statistics.median(seconds)
        steps_per_second = steps / median_seconds
        particle_steps_per_second = particles * steps_per_second
        rows.append((solver, particles, steps, median_seconds, steps_per_second, particle_steps_per_second))
    return rows


def write_markdown(path: Path, results: list[BenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize(results)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# CPU Benchmark Results",
        "",
        f"Generated: {generated}",
        "",
        f"Platform: `{platform.platform()}`",
        "",
        "Build: Release CPU executable, MPI disabled, CUDA disabled.",
        "",
        "| Solver | Particles | Steps | Median wall time (s) | Steps/s | Particle-steps/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for solver, particles, steps, seconds, steps_per_second, particle_steps_per_second in rows:
        lines.append(
            f"| `{solver}` | {particles} | {steps} | {seconds:.3f} | "
            f"{steps_per_second:.2f} | {particle_steps_per_second:,.0f} |"
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
    args = parser.parse_args()

    if not args.executable.exists():
        raise FileNotFoundError(f"Executable not found: {args.executable}")

    results: list[BenchmarkCase] = []
    for particles in args.particles:
        for solver in args.solvers:
            for replicate in range(1, args.repetitions + 1):
                result = run_case(args.executable, solver, particles, args.steps, replicate, args.work_dir)
                results.append(result)
                print(
                    f"{solver:>6} n={particles:<5} run={replicate} "
                    f"{result.seconds:.3f}s ({result.steps_per_second:.2f} steps/s)"
                )

    write_csv(args.csv, results)
    write_markdown(args.markdown, results)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
