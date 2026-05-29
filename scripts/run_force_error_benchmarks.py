"""Run force-error and drift benchmarks against the direct solver reference."""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.utils.snapshots import load_diagnostics, load_snapshot
from scripts.experiment_utils import (
    resolve_simulator_executable,
    run_simulator,
    safe_float_label,
    write_two_galaxy_config,
)


EPSILON = 1.0e-12


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    seconds: float
    peak_memory_mb: float | None
    metadata: dict


@dataclass(frozen=True)
class AccuracyResult:
    solver: str
    particles: int
    theta: float
    leaf_capacity: int
    expansion_order: int
    softening: float
    steps: int
    force_rmse: float
    max_force_error: float
    relative_force_error: float
    energy_drift: float
    momentum_drift: float
    angular_momentum_drift: float
    seconds: float
    runtime_per_step: float
    particle_steps_per_second: float
    peak_memory_mb: float | None
    git_commit: str
    config_sha256: str
    output_dir: Path


def _case_label(
    solver: str,
    particles: int,
    theta: float,
    leaf_capacity: int,
    expansion_order: int,
    softening: float,
) -> str:
    return (
        f"{solver}_n{particles}_theta{safe_float_label(theta)}_leaf{leaf_capacity}_"
        f"p{expansion_order}_soft{safe_float_label(softening)}"
    )


def write_config(
    path: Path,
    solver: str,
    particles: int,
    steps: int,
    snapshot_every: int,
    output: Path,
    theta: float,
    leaf_capacity: int,
    expansion_order: int,
    softening: float,
) -> None:
    write_two_galaxy_config(
        path,
        name=f"accuracy_{solver}_{particles}",
        solver=solver,
        particles=particles,
        steps=steps,
        dt=0.01,
        snapshot_every=snapshot_every,
        output=output,
        output_format="csv",
        theta=theta,
        leaf_capacity=leaf_capacity,
        expansion_order=expansion_order,
        softening=softening,
        seed=20260526,
    )


def run_simulation(executable: Path, config_path: Path, output_dir: Path) -> RunResult:
    completed = run_simulator(
        executable,
        config_path,
        output_dir,
        cwd=Path.cwd(),
        capture_memory=True,
    )
    if completed.exit_code != 0:
        raise RuntimeError(f"Simulation failed for {config_path}\n{completed.stdout}")
    return RunResult(output_dir, completed.seconds, completed.peak_memory_mb, completed.metadata)


def force_error(reference_dir: Path, candidate_dir: Path) -> tuple[float, float, float]:
    reference = load_snapshot(reference_dir / "snapshot_000000.csv")
    candidate = load_snapshot(candidate_dir / "snapshot_000000.csv")
    if len(reference.ids) != len(candidate.ids) or not np.array_equal(reference.ids, candidate.ids):
        raise RuntimeError(f"Snapshot particle IDs do not match: {reference.path} vs {candidate.path}")

    diff = candidate.accelerations - reference.accelerations
    diff_norm = np.linalg.norm(diff, axis=1)
    reference_norm = np.linalg.norm(reference.accelerations, axis=1)
    force_rmse = float(np.sqrt(np.mean(diff_norm * diff_norm)))
    max_force_error = float(np.max(diff_norm))
    reference_rmse = float(np.sqrt(np.mean(reference_norm * reference_norm)))
    relative_force_error = float(force_rmse / max(reference_rmse, EPSILON))
    return force_rmse, max_force_error, relative_force_error


def vector_drift(diagnostics: np.ndarray, columns: tuple[str, str, str]) -> float:
    vectors = np.column_stack([diagnostics[name] for name in columns])
    baseline = vectors[0]
    delta = np.linalg.norm(vectors - baseline, axis=1)
    return float(np.max(delta) / max(float(np.linalg.norm(baseline)), EPSILON))


def diagnostics_drift(output_dir: Path) -> tuple[float, float, float]:
    diagnostics = load_diagnostics(output_dir)
    if diagnostics.shape == ():
        diagnostics = np.array([diagnostics], dtype=diagnostics.dtype)
    initial_energy = float(diagnostics["total_energy"][0])
    energy_drift = float(
        np.max(np.abs(diagnostics["total_energy"] - initial_energy)) / max(abs(initial_energy), EPSILON)
    )
    momentum_drift = vector_drift(diagnostics, ("momentum_x", "momentum_y", "momentum_z"))
    angular_momentum_drift = vector_drift(
        diagnostics,
        ("angular_momentum_x", "angular_momentum_y", "angular_momentum_z"),
    )
    return energy_drift, momentum_drift, angular_momentum_drift


def run_suite(args: argparse.Namespace) -> list[AccuracyResult]:
    snapshot_every = max(1, args.steps // args.diagnostic_samples)
    references: dict[tuple[int, float], RunResult] = {}
    results: list[AccuracyResult] = []

    for particles in args.particles:
        for softening in args.softening:
            reference_key = (particles, softening)
            reference_output = (
                args.output / "runs" / f"direct_reference_n{particles}_soft{safe_float_label(softening)}"
            )
            reference_config = args.output / "configs" / f"{reference_output.name}.toml"
            write_config(
                reference_config,
                "direct",
                particles,
                args.steps,
                snapshot_every,
                reference_output,
                args.theta[0],
                args.leaf_capacity[0],
                args.expansion_order[0],
                softening,
            )
            references[reference_key] = run_simulation(args.executable, reference_config, reference_output)

            for solver in args.solvers:
                for theta in args.theta:
                    for leaf_capacity in args.leaf_capacity:
                        for expansion_order in args.expansion_order:
                            label = _case_label(
                                solver,
                                particles,
                                theta,
                                leaf_capacity,
                                expansion_order,
                                softening,
                            )
                            output_dir = args.output / "runs" / label
                            config_path = args.output / "configs" / f"{label}.toml"
                            write_config(
                                config_path,
                                solver,
                                particles,
                                args.steps,
                                snapshot_every,
                                output_dir,
                                theta,
                                leaf_capacity,
                                expansion_order,
                                softening,
                            )
                            run = run_simulation(args.executable, config_path, output_dir)
                            force_rmse, max_force_error, relative_force_error = force_error(
                                references[reference_key].output_dir,
                                output_dir,
                            )
                            energy_drift, momentum_drift, angular_momentum_drift = diagnostics_drift(
                                output_dir
                            )
                            results.append(
                                AccuracyResult(
                                    solver=solver,
                                    particles=particles,
                                    theta=theta,
                                    leaf_capacity=leaf_capacity,
                                    expansion_order=expansion_order,
                                    softening=softening,
                                    steps=args.steps,
                                    force_rmse=force_rmse,
                                    max_force_error=max_force_error,
                                    relative_force_error=relative_force_error,
                                    energy_drift=energy_drift,
                                    momentum_drift=momentum_drift,
                                    angular_momentum_drift=angular_momentum_drift,
                                    seconds=run.seconds,
                                    runtime_per_step=run.seconds / max(args.steps, 1),
                                    particle_steps_per_second=(particles * max(args.steps, 1)) / run.seconds,
                                    peak_memory_mb=run.peak_memory_mb,
                                    git_commit=run.metadata.get("git_commit", "unavailable"),
                                    config_sha256=run.metadata.get("config_sha256", "unavailable"),
                                    output_dir=output_dir,
                                )
                            )
                            print(
                                f"{solver:>4} n={particles:<5} theta={theta:g} leaf={leaf_capacity:<3} "
                                f"p={expansion_order:<1} rel_err={relative_force_error:.3e} "
                                f"{run.seconds:.3f}s"
                            )
    return results


def write_csv_summary(path: Path, results: Iterable[AccuracyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "solver",
        "particles",
        "theta",
        "leaf_capacity",
        "expansion_order",
        "softening",
        "steps",
        "force_rmse",
        "max_force_error",
        "relative_force_error",
        "energy_drift",
        "momentum_drift",
        "angular_momentum_drift",
        "seconds",
        "runtime_per_step",
        "particle_steps_per_second",
        "peak_memory_mb",
        "git_commit",
        "config_sha256",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "solver": result.solver,
                    "particles": result.particles,
                    "theta": result.theta,
                    "leaf_capacity": result.leaf_capacity,
                    "expansion_order": result.expansion_order,
                    "softening": result.softening,
                    "steps": result.steps,
                    "force_rmse": f"{result.force_rmse:.12e}",
                    "max_force_error": f"{result.max_force_error:.12e}",
                    "relative_force_error": f"{result.relative_force_error:.12e}",
                    "energy_drift": f"{result.energy_drift:.12e}",
                    "momentum_drift": f"{result.momentum_drift:.12e}",
                    "angular_momentum_drift": f"{result.angular_momentum_drift:.12e}",
                    "seconds": f"{result.seconds:.6f}",
                    "runtime_per_step": f"{result.runtime_per_step:.6f}",
                    "particle_steps_per_second": f"{result.particle_steps_per_second:.3f}",
                    "peak_memory_mb": "" if result.peak_memory_mb is None else f"{result.peak_memory_mb:.3f}",
                    "git_commit": result.git_commit,
                    "config_sha256": result.config_sha256,
                    "output_dir": result.output_dir.as_posix(),
                }
            )


def write_markdown_summary(path: Path, results: list[AccuracyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commits = sorted({result.git_commit for result in results})
    commit_summary = commits[0] if len(commits) == 1 else "mixed"
    if commit_summary not in {"mixed", "unavailable"}:
        commit_summary = commit_summary[:12]
    rows = sorted(results, key=lambda r: (r.particles, r.solver, r.theta, r.leaf_capacity, r.expansion_order))
    lines = [
        "# Force-Error Benchmark Results",
        "",
        f"Generated: {generated}",
        "",
        f"Platform: `{platform.platform()}`",
        "",
        f"Commit: `{commit_summary}`",
        "",
        "| Solver | N | Theta | Leaf | p | Softening | Rel force error | RMSE | "
        "Max error | Energy drift | Momentum drift | Particle-steps/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in rows:
        lines.append(
            f"| `{result.solver}` | {result.particles} | {result.theta:g} | {result.leaf_capacity} | "
            f"{result.expansion_order} | {result.softening:g} | {result.relative_force_error:.3e} | "
            f"{result.force_rmse:.3e} | {result.max_force_error:.3e} | {result.energy_drift:.3e} | "
            f"{result.momentum_drift:.3e} | {result.particle_steps_per_second:,.0f} |"
        )

    if results:
        fastest = max(results, key=lambda r: r.particle_steps_per_second)
        most_accurate = min(results, key=lambda r: r.relative_force_error)
        median_relative_error = statistics.median(result.relative_force_error for result in results)
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Fastest case: `{fastest.solver}` n={fastest.particles} theta={fastest.theta:g} "
                f"leaf={fastest.leaf_capacity} p={fastest.expansion_order} "
                f"({fastest.particle_steps_per_second:,.0f} particle-steps/s).",
                f"- Lowest relative force error: `{most_accurate.solver}` n={most_accurate.particles} "
                f"theta={most_accurate.theta:g} leaf={most_accurate.leaf_capacity} "
                f"p={most_accurate.expansion_order} ({most_accurate.relative_force_error:.3e}).",
                f"- Median relative force error: {median_relative_error:.3e}.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(output: Path, results: list[AccuracyResult]) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    colors = {"tree": "tab:blue", "fmm": "tab:orange", "cuda-tree": "tab:green", "cuda-fmm": "tab:red"}

    def _scatter(path: Path, x_name: str, y_name: str, xlabel: str, ylabel: str, log_x: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for solver in sorted({result.solver for result in results}):
            solver_results = [result for result in results if result.solver == solver]
            ax.scatter(
                [getattr(result, x_name) for result in solver_results],
                [max(float(getattr(result, y_name)), 1.0e-16) for result in solver_results],
                label=solver,
                alpha=0.78,
                color=colors.get(solver),
            )
        if log_x:
            ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.28)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    _scatter(
        output / "force_error_vs_n.png",
        "particles",
        "relative_force_error",
        "Particles",
        "Relative force error vs direct",
        log_x=True,
    )
    _scatter(
        output / "force_error_vs_theta.png",
        "theta",
        "relative_force_error",
        "Tree theta",
        "Relative force error vs direct",
    )
    _scatter(
        output / "energy_drift.png",
        "particles",
        "energy_drift",
        "Particles",
        "Energy drift",
        log_x=True,
    )
    _scatter(
        output / "momentum_drift.png",
        "particles",
        "momentum_drift",
        "Particles",
        "Momentum drift",
        log_x=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/fmm_galaxy_sim"))
    parser.add_argument("--output", type=Path, default=Path("experiments/accuracy"))
    parser.add_argument("--solvers", nargs="+", default=["tree", "fmm"])
    parser.add_argument("--particles", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--theta", nargs="+", type=float, default=[0.4, 0.6])
    parser.add_argument("--leaf-capacity", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--expansion-order", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--softening", nargs="+", type=float, default=[0.02])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--diagnostic-samples", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="Run a CI-scale one-case benchmark.")
    args = parser.parse_args()

    if args.smoke:
        args.particles = [32]
        args.theta = [0.6]
        args.leaf_capacity = [8]
        args.expansion_order = [0]
        args.softening = [0.02]
        args.steps = 2
        args.diagnostic_samples = 2

    args.executable = resolve_simulator_executable(args.executable)
    if args.diagnostic_samples <= 0:
        raise ValueError("--diagnostic-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = run_suite(args)
    write_csv_summary(args.output / "force_error_summary.csv", results)
    write_markdown_summary(args.output / "force_error_summary.md", results)
    plot_results(args.output, results)
    print(f"Wrote {args.output / 'force_error_summary.csv'}")
    print(f"Wrote {args.output / 'force_error_summary.md'}")
    print(f"Wrote plots to {args.output}")


if __name__ == "__main__":
    main()
