"""Run repeatable local benchmarks for the simulator executable."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_utils import benchmark_env, write_two_galaxy_config


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
    build_type: str
    compiler: str
    cuda_available: bool | str
    cuda_device_name: str
    mpi_enabled: bool | str
    hostname: str

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds

    @property
    def particle_steps_per_second(self) -> float:
        return (self.particles * self.steps) / self.seconds


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
    write_two_galaxy_config(
        config_path,
        name=f"benchmark_{solver}_{particles}",
        solver=solver,
        particles=particles,
        steps=steps,
        dt=0.01,
        snapshot_every=steps,
        output=output_dir,
        output_format=output_format,
        theta=theta,
        leaf_capacity=leaf_capacity,
        expansion_order=expansion_order,
        softening=0.025,
        seed=20260502,
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
        metadata.get("build_type", "unknown"),
        metadata.get("compiler", "unknown"),
        metadata.get("cuda_available", ""),
        metadata.get("cuda_device_name", ""),
        metadata.get("mpi_enabled", ""),
        metadata.get("hostname", "unknown"),
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
                "build_type",
                "compiler",
                "cuda_available",
                "cuda_device_name",
                "mpi_enabled",
                "hostname",
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
                    result.build_type,
                    result.compiler,
                    result.cuda_available,
                    result.cuda_device_name,
                    result.mpi_enabled,
                    result.hostname,
                ]
            )


def summarize(results: list[BenchmarkCase]) -> list[tuple[str, str, int, int, float, float, float, str]]:
    grouped: dict[tuple[str, str, int, int], list[BenchmarkCase]] = {}
    for result in results:
        key = (result.output_format, result.solver, result.particles, result.steps)
        grouped.setdefault(key, []).append(result)

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
        "| Output | Solver | Particles | Steps | Median wall time (s) | "
        "Steps/s | Particle-steps/s | Commit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        (
            output_format,
            solver,
            particles,
            steps,
            seconds,
            steps_per_second,
            particle_steps_per_second,
            git_commit,
        ) = row
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
    parser.add_argument(
        "--crossover-suite",
        action="store_true",
        help="Use a wider N sweep with csv and none output for solver crossover analysis.",
    )
    args = parser.parse_args()

    if not args.executable.exists():
        raise FileNotFoundError(f"Executable not found: {args.executable}")

    if args.crossover_suite:
        args.output_formats = ["none", "csv"]
        args.particles = [128, 256, 512, 1024, 2048, 4096]
        args.solvers = ["direct", "tree", "fmm"]

    results: list[BenchmarkCase] = []
    output_formats = args.output_formats if args.output_formats is not None else [args.output_format]
    for output_format in output_formats:
        for particles in args.particles:
            for solver in args.solvers:
                for replicate in range(1, args.repetitions + 1):
                    result = run_case(
                        args.executable,
                        solver,
                        particles,
                        args.steps,
                        replicate,
                        args.work_dir,
                        output_format,
                        args.theta,
                        args.leaf_capacity,
                        args.expansion_order,
                    )
                    results.append(result)
                    print(
                        f"{output_format:>7} {solver:>6} n={particles:<5} run={replicate} "
                        f"{result.seconds:.3f}s ({result.steps_per_second:.2f} steps/s)"
                    )

    write_csv(args.csv, results)
    write_markdown(args.markdown, results)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
