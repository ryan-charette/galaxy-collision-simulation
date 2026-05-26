"""Run force-error and drift benchmarks against the direct solver reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.utils.snapshots import load_diagnostics, load_snapshot


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


def _safe_float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _case_label(
    solver: str,
    particles: int,
    theta: float,
    leaf_capacity: int,
    expansion_order: int,
    softening: float,
) -> str:
    return (
        f"{solver}_n{particles}_theta{_safe_float_label(theta)}_leaf{leaf_capacity}_"
        f"p{expansion_order}_soft{_safe_float_label(softening)}"
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
    half = particles // 2
    rest = particles - half
    config = f"""[simulation]
name = "accuracy_{solver}_{particles}"
dim = 3
solver = "{solver}"
seed = 20260526
n_particles = {particles}
steps = {steps}
dt = 0.01
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


def run_simulation(executable: Path, config_path: Path, output_dir: Path) -> RunResult:
    command = [str(executable), "--config", str(config_path)]
    peak_memory_mb: float | None = None
    psutil = None
    try:
        import psutil as psutil_module  # type: ignore

        psutil = psutil_module
    except ImportError:
        psutil = None

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=benchmark_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if psutil is not None:
        tracked = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = tracked.memory_info().rss / (1024.0 * 1024.0)
                peak_memory_mb = max(peak_memory_mb or 0.0, rss)
            except psutil.Error:
                pass
            time.sleep(0.02)
    stdout, _ = process.communicate()
    seconds = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"Simulation failed for {config_path}\n{stdout}")

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return RunResult(output_dir, seconds, peak_memory_mb, metadata)


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
    relative_force_error = float(force_rmse / max(float(np.sqrt(np.mean(reference_norm * reference_norm))), EPSILON))
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
            reference_output = args.output / "runs" / f"direct_reference_n{particles}_soft{_safe_float_label(softening)}"
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
                            energy_drift, momentum_drift, angular_momentum_drift = diagnostics_drift(output_dir)
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
        "| Solver | N | Theta | Leaf | p | Softening | Rel force error | RMSE | Max error | Energy drift | Momentum drift | Particle-steps/s |",
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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plot_results_basic_png(output, results)
        return

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
    _scatter(output / "energy_drift.png", "particles", "energy_drift", "Particles", "Energy drift", log_x=True)
    _scatter(
        output / "momentum_drift.png",
        "particles",
        "momentum_drift",
        "Particles",
        "Momentum drift",
        log_x=True,
    )


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    rows = bytearray()
    stride = width * 3
    for y in range(height):
        rows.append(0)
        rows.extend(pixels[y * stride : (y + 1) * stride])

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(rows), level=9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(png)


def draw_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        pixels[offset : offset + 3] = bytes(color)


def draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        draw_pixel(pixels, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def draw_circle(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    radius_squared = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius_squared:
                draw_pixel(pixels, width, height, x, y, color)


def plot_basic_scatter(
    path: Path,
    results: list[AccuracyResult],
    x_name: str,
    y_name: str,
    log_x: bool = False,
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 70, 30, 30, 60
    pixels = bytearray([255] * width * height * 3)
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    axis_color = (40, 40, 40)
    grid_color = (220, 220, 220)
    colors = {
        "tree": (31, 119, 180),
        "fmm": (255, 127, 14),
        "cuda-tree": (44, 160, 44),
        "cuda-fmm": (214, 39, 40),
    }

    x_values = [float(getattr(result, x_name)) for result in results]
    y_values = [max(float(getattr(result, y_name)), 1.0e-16) for result in results]
    if log_x:
        x_values = [math.log2(max(value, 1.0e-16)) for value in x_values]
    y_values = [math.log10(value) for value in y_values]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    for fraction in (0.25, 0.5, 0.75):
        x = int(plot_left + fraction * (plot_right - plot_left))
        y = int(plot_bottom - fraction * (plot_bottom - plot_top))
        draw_line(pixels, width, height, x, plot_top, x, plot_bottom, grid_color)
        draw_line(pixels, width, height, plot_left, y, plot_right, y, grid_color)
    draw_line(pixels, width, height, plot_left, plot_bottom, plot_right, plot_bottom, axis_color)
    draw_line(pixels, width, height, plot_left, plot_bottom, plot_left, plot_top, axis_color)

    for result in results:
        x_value = float(getattr(result, x_name))
        if log_x:
            x_value = math.log2(max(x_value, 1.0e-16))
        y_value = math.log10(max(float(getattr(result, y_name)), 1.0e-16))
        x = int(plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left))
        y = int(plot_bottom - (y_value - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        draw_circle(pixels, width, height, x, y, 5, colors.get(result.solver, (90, 90, 90)))

    write_png(path, width, height, pixels)


def plot_results_basic_png(output: Path, results: list[AccuracyResult]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plot_basic_scatter(output / "force_error_vs_n.png", results, "particles", "relative_force_error", log_x=True)
    plot_basic_scatter(output / "force_error_vs_theta.png", results, "theta", "relative_force_error")
    plot_basic_scatter(output / "energy_drift.png", results, "particles", "energy_drift", log_x=True)
    plot_basic_scatter(output / "momentum_drift.png", results, "particles", "momentum_drift", log_x=True)


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

    if not args.executable.exists() and args.executable == Path("build/fmm_galaxy_sim"):
        for candidate in (Path("build/Release/fmm_galaxy_sim.exe"), Path("build/fmm_galaxy_sim.exe")):
            if candidate.exists():
                args.executable = candidate
                break

    if not args.executable.exists():
        raise FileNotFoundError(f"Executable not found: {args.executable}")
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
