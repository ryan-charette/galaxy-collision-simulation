from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from python.utils.snapshots import Snapshot, list_snapshot_files, load_snapshot


PALETTE = np.array(
    [
        (88, 190, 255),
        (255, 185, 82),
        (167, 243, 208),
        (244, 114, 182),
        (250, 204, 21),
        (192, 132, 252),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SceneBounds:
    center: np.ndarray
    span: float


@dataclass(frozen=True)
class ProjectedSnapshot:
    snapshot: Snapshot
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    group_id: np.ndarray
    masses: np.ndarray
    total_particles: int


def _select_frame_paths(paths: list[Path], stride: int, max_frames: int) -> list[Path]:
    if stride <= 0:
        raise ValueError("--frame-stride must be positive")
    selected = paths[::stride]
    if max_frames > 0 and len(selected) > max_frames:
        indices = np.linspace(0, len(selected) - 1, max_frames)
        selected = [selected[int(round(index))] for index in indices]
    return selected


def _sample_ids(snapshot: Snapshot, max_particles: int, seed: int) -> set[int] | None:
    if max_particles <= 0 or len(snapshot.ids) <= max_particles:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(snapshot.ids), size=max_particles, replace=False)
    return {int(value) for value in snapshot.ids[np.sort(indices)]}


def _apply_sample(snapshot: Snapshot, sample_ids: set[int] | None) -> Snapshot:
    if sample_ids is None:
        return snapshot
    mask = np.isin(snapshot.ids, list(sample_ids))
    if not np.any(mask):
        return snapshot
    return Snapshot(
        step=snapshot.step,
        time=snapshot.time,
        ids=snapshot.ids[mask],
        positions=snapshot.positions[mask],
        velocities=snapshot.velocities[mask],
        accelerations=snapshot.accelerations[mask],
        masses=snapshot.masses[mask],
        group_id=snapshot.group_id[mask],
        path=snapshot.path,
    )


def _compute_scene_bounds(paths: list[Path], margin: float) -> SceneBounds:
    mins: np.ndarray | None = None
    maxs: np.ndarray | None = None
    for path in paths:
        positions = load_snapshot(path).positions
        if len(positions) == 0:
            continue
        frame_min = positions.min(axis=0)
        frame_max = positions.max(axis=0)
        mins = frame_min if mins is None else np.minimum(mins, frame_min)
        maxs = frame_max if maxs is None else np.maximum(maxs, frame_max)
    if mins is None or maxs is None:
        raise ValueError("Selected snapshots do not contain any particles")
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    return SceneBounds(center=center, span=max(span * (1.0 + margin), 1.0e-12))


def _rotate(points: np.ndarray, azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    cos_az, sin_az = math.cos(azimuth), math.sin(azimuth)
    cos_el, sin_el = math.cos(elevation), math.sin(elevation)

    x = points[:, 0] * cos_az - points[:, 1] * sin_az
    y = points[:, 0] * sin_az + points[:, 1] * cos_az
    z = points[:, 2]

    y2 = y * cos_el - z * sin_el
    z2 = y * sin_el + z * cos_el
    return np.column_stack([x, y2, z2])


def _project_snapshot(
    snapshot: Snapshot,
    total_particles: int,
    bounds: SceneBounds,
    frame_index: int,
    frame_count: int,
    args: argparse.Namespace,
) -> ProjectedSnapshot:
    t = frame_index / max(1, frame_count - 1)
    azimuth = args.azimuth_start + args.azimuth_span * t
    centered = snapshot.positions - bounds.center
    rotated = _rotate(centered, azimuth, args.elevation)

    scale = args.zoom * min(args.width, args.height) / bounds.span
    x = args.width * 0.5 + rotated[:, 0] * scale
    y = args.height * 0.54 - rotated[:, 1] * scale
    depth = rotated[:, 2] / bounds.span
    return ProjectedSnapshot(
        snapshot=snapshot,
        x=x,
        y=y,
        depth=depth,
        group_id=snapshot.group_id,
        masses=snapshot.masses,
        total_particles=total_particles,
    )


def _background(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "#101827")
    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(height):
        tint = int(30 * y / max(1, height - 1))
        draw.line([(0, y), (width, y)], fill=(15, 23, 42 + tint // 4, 255))
    return image


def _valid_mask(frame: ProjectedSnapshot, width: int, height: int) -> np.ndarray:
    return (frame.x >= 0) & (frame.x < width) & (frame.y >= 0) & (frame.y < height)


def _render_density(frame: ProjectedSnapshot, args: argparse.Namespace) -> Image.Image:
    background = np.asarray(_background(args.width, args.height), dtype=float)
    valid = _valid_mask(frame, args.width, args.height)
    if not np.any(valid):
        return Image.fromarray(background.astype(np.uint8), mode="RGB")

    rgb = np.zeros((args.height, args.width, 3), dtype=float)
    alpha = np.zeros((args.height, args.width), dtype=float)
    groups = sorted(int(value) for value in np.unique(frame.group_id[valid]))
    for group in groups:
        group_mask = valid & (frame.group_id == group)
        hist, _, _ = np.histogram2d(
            frame.y[group_mask],
            frame.x[group_mask],
            bins=(args.height, args.width),
            range=((0, args.height), (0, args.width)),
            weights=frame.masses[group_mask],
        )
        density = np.log1p(hist)
        positive = density[density > 0]
        if len(positive):
            scale = np.percentile(positive, args.density_percentile)
            intensity = np.clip(density / max(float(scale), 1.0e-12), 0.0, 1.0)
            color = PALETTE[group % len(PALETTE)]
            rgb += intensity[..., None] * color
            alpha = np.maximum(alpha, intensity)

    blended = background * (1.0 - 0.92 * alpha[..., None]) + np.clip(rgb, 0.0, 255.0) * (
        0.92 * alpha[..., None]
    )
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _render_scatter(frame: ProjectedSnapshot, args: argparse.Namespace) -> Image.Image:
    image = _background(args.width, args.height)
    draw = ImageDraw.Draw(image, "RGBA")
    valid = _valid_mask(frame, args.width, args.height)
    order = np.argsort(frame.depth[valid])
    valid_indices = np.flatnonzero(valid)[order]
    positive_masses = frame.masses[frame.masses > 0]
    mass_scale = float(np.median(positive_masses)) if len(positive_masses) else 1.0

    for index in valid_indices:
        group = int(frame.group_id[index])
        color = PALETTE[group % len(PALETTE)]
        brightness = np.clip(0.78 + 0.25 * frame.depth[index], 0.45, 1.15)
        radius = args.point_radius * math.sqrt(max(float(frame.masses[index]) / mass_scale, 0.05))
        x = float(frame.x[index])
        y = float(frame.y[index])
        fill = tuple(int(np.clip(channel * brightness, 0, 255)) for channel in color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*fill, args.point_alpha))
    return image


def _label_text(template: str, frame: ProjectedSnapshot) -> str:
    return template.format(
        step=frame.snapshot.step,
        time=frame.snapshot.time,
        n=frame.total_particles,
        rendered=len(frame.snapshot.positions),
        path=frame.snapshot.path.name if frame.snapshot.path else "",
    )


def _draw_label(image: Image.Image, label: str) -> None:
    if not label:
        return
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = 14
    y = 12
    draw.rectangle((x, y, x + width + 16, y + height + 12), fill=(15, 23, 42, 176))
    draw.text((x + 8, y + 6), label, fill=(226, 232, 240, 238), font=font)


def _render_frame(
    path: Path,
    sample_ids: set[int] | None,
    bounds: SceneBounds,
    frame_index: int,
    frame_count: int,
    args: argparse.Namespace,
) -> Image.Image:
    loaded = load_snapshot(path)
    total_particles = len(loaded.positions)
    snapshot = _apply_sample(loaded, sample_ids)
    projected = _project_snapshot(snapshot, total_particles, bounds, frame_index, frame_count, args)
    image = _render_density(projected, args) if args.mode == "density" else _render_scatter(projected, args)
    if not args.no_label:
        _draw_label(image, _label_text(args.label, projected))
    return image


def render_gif(args: argparse.Namespace) -> None:
    paths = _select_frame_paths(list_snapshot_files(args.input), args.frame_stride, args.max_frames)
    if not paths:
        raise FileNotFoundError(f"No snapshot_*.csv or snapshot_*.parquet files found in {args.input}")

    first_snapshot = load_snapshot(paths[0])
    sample_ids = _sample_ids(first_snapshot, args.max_particles, args.sample_seed)
    bounds = _compute_scene_bounds(paths[:: max(args.bounds_stride, 1)], args.bounds_margin)

    frames = [
        _render_frame(path, sample_ids, bounds, index, len(paths), args)
        for index, path in enumerate(paths)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(1000 / args.fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {args.output} ({len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a scalable README-style GIF from simulator snapshots."
    )
    parser.add_argument("--input", type=Path, default=Path("experiments/validation/smoke_test"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/galaxy_collision.gif"))
    parser.add_argument("--mode", choices=["density", "scatter"], default="density")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=180, help="Use 0 to render every selected frame.")
    parser.add_argument("--max-particles", type=int, default=0, help="Use 0 to render every particle.")
    parser.add_argument("--sample-seed", type=int, default=20260603)
    parser.add_argument("--bounds-stride", type=int, default=1)
    parser.add_argument("--bounds-margin", type=float, default=0.12)
    parser.add_argument("--zoom", type=float, default=0.88)
    parser.add_argument("--azimuth-start", type=float, default=-35.0)
    parser.add_argument("--azimuth-span", type=float, default=145.0)
    parser.add_argument("--elevation", type=float, default=28.0)
    parser.add_argument("--density-percentile", type=float, default=99.5)
    parser.add_argument("--point-radius", type=float, default=1.2)
    parser.add_argument("--point-alpha", type=int, default=205)
    parser.add_argument(
        "--label",
        default="n={n:,} | rendered={rendered:,} | step {step:06d} | t={time:.3f}",
        help="Python format string with n, rendered, step, time, and path fields.",
    )
    parser.add_argument("--no-label", action="store_true")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.bounds_stride <= 0:
        raise ValueError("--bounds-stride must be positive")
    if args.density_percentile <= 0 or args.density_percentile > 100:
        raise ValueError("--density-percentile must be in (0, 100]")

    render_gif(args)


if __name__ == "__main__":
    main()
