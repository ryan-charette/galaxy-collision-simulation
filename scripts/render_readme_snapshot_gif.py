"""Render the README GIF from simulator CSV snapshots.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Snapshot:
    step: int
    time: float
    rows: list[tuple[int, float, float, float, float]]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rotate_x(point: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = point
    ca, sa = math.cos(angle), math.sin(angle)
    return x, y * ca - z * sa, y * sa + z * ca


def rotate_z(point: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    x, y, z = point
    ca, sa = math.cos(angle), math.sin(angle)
    return x * ca - y * sa, x * sa + y * ca, z


def load_snapshot(path: Path) -> Snapshot:
    time = 0.0
    rows: list[tuple[int, float, float, float, float]] = []
    with path.open(newline="") as handle:
        first = handle.readline().strip()
        if first.startswith("# time="):
            time = float(first.split("=", 1)[1])
        else:
            handle.seek(0)
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    int(row["group_id"]),
                    float(row["mass"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                )
            )
    step = int(path.stem.split("_")[-1])
    return Snapshot(step=step, time=time, rows=rows)


def load_snapshots(directory: Path) -> list[Snapshot]:
    paths = sorted(directory.glob("snapshot_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No snapshot_*.csv files found in {directory}")
    return [load_snapshot(path) for path in paths]


def bounds(snapshots: list[Snapshot]) -> tuple[float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for snapshot in snapshots:
        for _group, _mass, x, y, z in snapshot.rows:
            xs.append(x)
            ys.append(y)
            zs.append(z)
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cz = 0.5 * (min(zs) + max(zs))
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    return cx, cy, cz, span


def draw_background(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "#101827")
    draw = ImageDraw.Draw(image, "RGBA")
    horizon = int(height * 0.76)
    for y in range(height):
        tint = int(28 * y / max(1, height - 1))
        draw.line([(0, y), (width, y)], fill=(15, 23, 42 + tint // 4, 255))
    for i in range(9):
        y = horizon - i * 24
        draw.line([(42, y), (width - 42, y)], fill=(148, 163, 184, max(9, 36 - i * 3)))
    for i in range(11):
        x = 55 + i * (width - 110) / 10.0
        draw.line([(x, horizon - 190), (width * 0.5, horizon + 18)], fill=(148, 163, 184, 15))
    return image


def project(
    point: tuple[float, float, float],
    center: tuple[float, float, float],
    span: float,
    frame: int,
    frame_count: int,
    width: int,
    height: int,
) -> tuple[float, float, float]:
    x = point[0] - center[0]
    y = point[1] - center[1]
    z = point[2] - center[2]
    t = frame / max(1, frame_count - 1)
    p = rotate_z((x, y, z), math.radians(-35.0 + 145.0 * t))
    p = rotate_x(p, math.radians(25.0 + 8.0 * math.sin(math.tau * t)))
    px, py, pz = p
    camera_distance = 5.8 * span
    scale = 0.82 * min(width, height) * camera_distance / (span * (camera_distance - pz))
    return width * 0.5 + px * scale, height * 0.55 - py * scale, pz / span


def render_frame(
    snapshot: Snapshot,
    frame: int,
    frame_count: int,
    center: tuple[float, float, float],
    span: float,
    width: int,
    height: int,
) -> Image.Image:
    image = draw_background(width, height)
    draw = ImageDraw.Draw(image, "RGBA")

    projected = []
    for group, mass, x, y, z in snapshot.rows:
        sx, sy, depth = project((x, y, z), center, span, frame, frame_count, width, height)
        if -10.0 <= sx <= width + 10.0 and -10.0 <= sy <= height + 10.0:
            projected.append((depth, sx, sy, group, mass))
    projected.sort(key=lambda item: item[0])

    for depth, sx, sy, group, mass in projected:
        base = (88, 190, 255) if group == 0 else (255, 185, 82)
        brightness = clamp(0.76 + 0.22 * depth, 0.48, 1.12)
        color = tuple(int(clamp(channel * brightness, 0, 255)) for channel in base)
        radius = 1.15 + 4.8 * math.sqrt(max(mass, 0.0))
        if depth > 0.12:
            radius += 0.25
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=(*color, 205))

    font = ImageFont.load_default()
    label = f"FMM p=4  |  n=1000  |  step {snapshot.step:03d}  |  t={snapshot.time:.3f}"
    draw.rectangle((14, 12, 318, 32), fill=(15, 23, 42, 172))
    draw.text((22, 18), label, fill=(226, 232, 240, 235), font=font)
    return image


def render_gif(input_dir: Path, output: Path, width: int, height: int, fps: int) -> None:
    snapshots = load_snapshots(input_dir)
    center_x, center_y, center_z, span = bounds(snapshots)
    center = (center_x, center_y, center_z)
    frames = [
        render_frame(snapshot, idx, len(snapshots), center, span, width, height)
        for idx, snapshot in enumerate(snapshots)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(1000 / fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/validation/readme_1000_body_collision"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/galaxy_collision_3d_1000.gif"))
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    render_gif(args.input, args.output, args.width, args.height, args.fps)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
