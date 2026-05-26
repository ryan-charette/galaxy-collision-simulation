"""Analyze solver runtime and accuracy crossover points."""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RuntimePoint:
    solver: str
    particles: int
    output_format: str
    seconds: float
    particle_steps_per_second: float
    build_type: str
    compiler: str
    cuda_available: str
    cuda_device_name: str
    mpi_enabled: str
    hostname: str

    @property
    def hardware_key(self) -> str:
        cuda = self.cuda_device_name if self.cuda_device_name else f"cuda={self.cuda_available}"
        return f"{self.hostname} / {self.build_type} / {self.compiler} / {cuda} / mpi={self.mpi_enabled}"


@dataclass(frozen=True)
class AccuracyPoint:
    solver: str
    particles: int
    force_rmse: float
    max_force_error: float
    relative_force_error: float
    seconds: float
    particle_steps_per_second: float
    build_type: str
    compiler: str
    cuda_available: str
    cuda_device_name: str
    mpi_enabled: str
    hostname: str

    @property
    def hardware_key(self) -> str:
        cuda = self.cuda_device_name if self.cuda_device_name else f"cuda={self.cuda_available}"
        return f"{self.hostname} / {self.build_type} / {self.compiler} / {cuda} / mpi={self.mpi_enabled}"


def as_float(value: str | None, default: float = float("nan")) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    if value in {None, ""}:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_runtime_csv(path: Path) -> list[RuntimePoint]:
    points: list[RuntimePoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seconds = as_float(row.get("seconds"))
            particles = as_int(row.get("particles"))
            particle_steps = as_float(row.get("particle_steps_per_second"))
            if math.isnan(particle_steps) and seconds > 0:
                steps = as_float(row.get("steps"), 1.0)
                particle_steps = particles * steps / seconds
            points.append(
                RuntimePoint(
                    solver=row.get("solver", "unknown"),
                    particles=particles,
                    output_format=row.get("output_format", "unknown") or "unknown",
                    seconds=seconds,
                    particle_steps_per_second=particle_steps,
                    build_type=row.get("build_type", "unknown") or "unknown",
                    compiler=row.get("compiler", "unknown") or "unknown",
                    cuda_available=row.get("cuda_available", "") or "",
                    cuda_device_name=row.get("cuda_device_name", "") or "",
                    mpi_enabled=row.get("mpi_enabled", "") or "",
                    hostname=row.get("hostname", "unknown") or "unknown",
                )
            )
    return points


def read_accuracy_csv(path: Path) -> list[AccuracyPoint]:
    points: list[AccuracyPoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            points.append(
                AccuracyPoint(
                    solver=row.get("solver", "unknown"),
                    particles=as_int(row.get("particles")),
                    force_rmse=as_float(row.get("force_rmse")),
                    max_force_error=as_float(row.get("max_force_error")),
                    relative_force_error=as_float(row.get("relative_force_error")),
                    seconds=as_float(row.get("seconds")),
                    particle_steps_per_second=as_float(row.get("particle_steps_per_second")),
                    build_type=row.get("build_type", "unknown") or "unknown",
                    compiler=row.get("compiler", "unknown") or "unknown",
                    cuda_available=row.get("cuda_available", "") or "",
                    cuda_device_name=row.get("cuda_device_name", "") or "",
                    mpi_enabled=row.get("mpi_enabled", "") or "",
                    hostname=row.get("hostname", "unknown") or "unknown",
                )
            )
    return points


def median_runtime(points: Iterable[RuntimePoint]) -> list[RuntimePoint]:
    grouped: dict[tuple[str, str, int, str], list[RuntimePoint]] = {}
    for point in points:
        grouped.setdefault((point.hardware_key, point.output_format, point.particles, point.solver), []).append(point)
    medians: list[RuntimePoint] = []
    for (_hardware, _output, _particles, _solver), rows in grouped.items():
        base = rows[0]
        medians.append(
            RuntimePoint(
                solver=base.solver,
                particles=base.particles,
                output_format=base.output_format,
                seconds=statistics.median(row.seconds for row in rows),
                particle_steps_per_second=statistics.median(row.particle_steps_per_second for row in rows),
                build_type=base.build_type,
                compiler=base.compiler,
                cuda_available=base.cuda_available,
                cuda_device_name=base.cuda_device_name,
                mpi_enabled=base.mpi_enabled,
                hostname=base.hostname,
            )
        )
    return sorted(medians, key=lambda point: (point.hardware_key, point.output_format, point.particles, point.solver))


def load_points(runtime_csvs: list[Path], accuracy_csvs: list[Path]) -> tuple[list[RuntimePoint], list[AccuracyPoint]]:
    runtime_points: list[RuntimePoint] = []
    for path in runtime_csvs:
        runtime_points.extend(read_runtime_csv(path))
    accuracy_points: list[AccuracyPoint] = []
    for path in accuracy_csvs:
        accuracy_points.extend(read_accuracy_csv(path))
    return median_runtime(runtime_points), accuracy_points


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
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                draw_pixel(pixels, width, height, x, y, color)


def basic_plot(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    log_x: bool = True,
    log_y: bool = True,
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 70, 30, 30, 60
    pixels = bytearray([255] * width * height * 3)
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
    axis_color = (40, 40, 40)
    grid_color = (220, 220, 220)

    transformed: dict[str, list[tuple[float, float]]] = {}
    for name, values in series.items():
        rows = []
        for x, y in values:
            if x <= 0 or y <= 0:
                continue
            rows.append((math.log10(x) if log_x else x, math.log10(y) if log_y else y))
        if rows:
            transformed[name] = rows
    if not transformed:
        write_png(path, width, height, pixels)
        return
    xs = [x for rows in transformed.values() for x, _y in rows]
    ys = [y for rows in transformed.values() for _x, y in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
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

    for index, (_name, values) in enumerate(sorted(transformed.items())):
        color = colors[index % len(colors)]
        last: tuple[int, int] | None = None
        for x_value, y_value in sorted(values):
            x = int(plot_left + (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left))
            y = int(plot_bottom - (y_value - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
            if last is not None:
                draw_line(pixels, width, height, last[0], last[1], x, y, color)
            draw_circle(pixels, width, height, x, y, 5, color)
            last = (x, y)
    write_png(path, width, height, pixels)


def plot_with_matplotlib(
    output: Path,
    runtime_points: list[RuntimePoint],
    accuracy_points: list[AccuracyPoint],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    def runtime_series(y_attr: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for solver in sorted({point.solver for point in runtime_points}):
            rows = sorted([point for point in runtime_points if point.solver == solver], key=lambda p: p.particles)
            ax.plot([p.particles for p in rows], [getattr(p, y_attr) for p in rows], marker="o", label=solver)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Particles")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.28)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    runtime_series("seconds", "Runtime (s)", "runtime_vs_n.png")
    runtime_series("particle_steps_per_second", "Particle-steps/s", "particle_steps_vs_n.png")

    if accuracy_points:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for solver in sorted({point.solver for point in accuracy_points}):
            rows = [point for point in accuracy_points if point.solver == solver]
            ax.scatter([p.seconds for p in rows], [p.relative_force_error for p in rows], label=solver, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Relative force error")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "force_error_vs_runtime.png", dpi=180)
        plt.close(fig)
    return True


def plot_outputs(output: Path, runtime_points: list[RuntimePoint], accuracy_points: list[AccuracyPoint]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if plot_with_matplotlib(output, runtime_points, accuracy_points):
        return

    runtime_series: dict[str, list[tuple[float, float]]] = {}
    particle_series: dict[str, list[tuple[float, float]]] = {}
    for point in runtime_points:
        key = f"{point.solver} ({point.output_format})"
        runtime_series.setdefault(key, []).append((point.particles, point.seconds))
        particle_series.setdefault(key, []).append((point.particles, point.particle_steps_per_second))
    basic_plot(output / "runtime_vs_n.png", runtime_series)
    basic_plot(output / "particle_steps_vs_n.png", particle_series)
    error_series: dict[str, list[tuple[float, float]]] = {}
    for point in accuracy_points:
        error_series.setdefault(point.solver, []).append((point.seconds, point.relative_force_error))
    basic_plot(output / "force_error_vs_runtime.png", error_series)


def best_solver_by_n(runtime_points: list[RuntimePoint]) -> list[RuntimePoint]:
    grouped: dict[tuple[str, str, int], list[RuntimePoint]] = {}
    for point in runtime_points:
        grouped.setdefault((point.hardware_key, point.output_format, point.particles), []).append(point)
    return [min(rows, key=lambda point: point.seconds) for rows in grouped.values()]


def target_accuracy_rows(accuracy_points: list[AccuracyPoint], thresholds: list[float]) -> list[tuple[float, int, str, float, float]]:
    rows: list[tuple[float, int, str, float, float]] = []
    for threshold in thresholds:
        for particles in sorted({point.particles for point in accuracy_points}):
            candidates = [
                point
                for point in accuracy_points
                if point.particles == particles and point.force_rmse <= threshold
            ]
            if candidates:
                best = min(candidates, key=lambda point: point.seconds)
                rows.append((threshold, particles, best.solver, best.force_rmse, best.seconds))
    return rows


def cuda_crossover_rows(runtime_points: list[RuntimePoint]) -> list[tuple[str, str, str]]:
    pairs = [("direct", "cuda-direct"), ("tree", "cuda-tree"), ("fmm", "cuda-fmm")]
    rows: list[tuple[str, str, str]] = []
    for cpu_solver, cuda_solver in pairs:
        found = "not observed"
        for particles in sorted({point.particles for point in runtime_points}):
            cpu = [point for point in runtime_points if point.solver == cpu_solver and point.particles == particles]
            cuda = [point for point in runtime_points if point.solver == cuda_solver and point.particles == particles]
            if cpu and cuda and min(point.seconds for point in cuda) < min(point.seconds for point in cpu):
                found = str(particles)
                break
        rows.append((cpu_solver, cuda_solver, found))
    return rows


def write_best_solver_csv(path: Path, rows: list[RuntimePoint]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hardware", "output_format", "particles", "solver", "seconds", "particle_steps_per_second"])
        for point in sorted(rows, key=lambda p: (p.hardware_key, p.output_format, p.particles)):
            writer.writerow(
                [
                    point.hardware_key,
                    point.output_format,
                    point.particles,
                    point.solver,
                    f"{point.seconds:.6f}",
                    f"{point.particle_steps_per_second:.3f}",
                ]
            )


def write_target_accuracy_csv(path: Path, rows: list[tuple[float, int, str, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_force_rmse", "particles", "solver", "force_rmse", "seconds"])
        for threshold, particles, solver, force_rmse, seconds in rows:
            writer.writerow([threshold, particles, solver, f"{force_rmse:.12e}", f"{seconds:.6f}"])


def write_markdown(
    path: Path,
    runtime_points: list[RuntimePoint],
    accuracy_points: list[AccuracyPoint],
    thresholds: list[float],
) -> None:
    best_rows = best_solver_by_n(runtime_points)
    target_rows = target_accuracy_rows(accuracy_points, thresholds)
    cuda_rows = cuda_crossover_rows(runtime_points)
    lines = [
        "# Solver Crossover Analysis",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Platform: `{platform.platform()}`",
        "",
        "## Fastest Solver By N",
        "",
        "| Hardware/build | Output | N | Fastest solver | Runtime (s) | Particle-steps/s |",
        "|---|---|---:|---|---:|---:|",
    ]
    for point in sorted(best_rows, key=lambda p: (p.hardware_key, p.output_format, p.particles)):
        lines.append(
            f"| {point.hardware_key} | `{point.output_format}` | {point.particles} | `{point.solver}` | "
            f"{point.seconds:.3f} | {point.particle_steps_per_second:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Target Accuracy",
            "",
            "| Target force RMSE | N | Fastest qualifying solver | Observed RMSE | Runtime (s) |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    if target_rows:
        for threshold, particles, solver, force_rmse, seconds in target_rows:
            lines.append(f"| {threshold:.3e} | {particles} | `{solver}` | {force_rmse:.3e} | {seconds:.3f} |")
    else:
        lines.append("| n/a | n/a | No qualifying force-error rows found | n/a | n/a |")

    lines.extend(
        [
            "",
            "## CPU vs CUDA Crossover",
            "",
            "| CPU solver | CUDA solver | First tested N where CUDA is faster |",
            "|---|---|---:|",
        ]
    )
    for cpu_solver, cuda_solver, particles in cuda_rows:
        lines.append(f"| `{cpu_solver}` | `{cuda_solver}` | {particles} |")

    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `runtime_vs_n.png`",
            "- `particle_steps_vs_n.png`",
            "- `force_error_vs_runtime.png`",
            "- `best_solver_by_n.csv`",
            "- `target_accuracy_summary.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_thresholds(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-csv",
        nargs="+",
        type=Path,
        default=[Path("docs/benchmarks/local_cpu_benchmark.csv")],
    )
    parser.add_argument(
        "--accuracy-csv",
        nargs="+",
        type=Path,
        default=[Path("experiments/accuracy/force_error_summary.csv")],
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/crossover"))
    parser.add_argument("--target-rmse", nargs="+", default=["1e-2", "1e-3", "1e-4"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_csvs = [path for path in args.runtime_csv if path.exists()]
    accuracy_csvs = [path for path in args.accuracy_csv if path.exists()]
    if not runtime_csvs:
        raise FileNotFoundError("No runtime CSV inputs were found")
    runtime_points, accuracy_points = load_points(runtime_csvs, accuracy_csvs)
    if not runtime_points:
        raise RuntimeError("No runtime benchmark rows were loaded")

    args.output.mkdir(parents=True, exist_ok=True)
    plot_outputs(args.output, runtime_points, accuracy_points)
    best_rows = best_solver_by_n(runtime_points)
    target_rows = target_accuracy_rows(accuracy_points, parse_thresholds(args.target_rmse))
    write_best_solver_csv(args.output / "best_solver_by_n.csv", best_rows)
    write_target_accuracy_csv(args.output / "target_accuracy_summary.csv", target_rows)
    write_markdown(
        args.output / "solver_crossover_summary.md",
        runtime_points,
        accuracy_points,
        parse_thresholds(args.target_rmse),
    )
    print(f"Wrote {args.output / 'solver_crossover_summary.md'}")
    print(f"Wrote plots and CSV summaries to {args.output}")


if __name__ == "__main__":
    main()
