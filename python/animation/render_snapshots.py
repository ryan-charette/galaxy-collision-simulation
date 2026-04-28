from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from python.utils.snapshots import iter_group_masks, load_snapshots


def _rotation_matrix(azimuth: float, elevation: float) -> np.ndarray:
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ]
    )
    return rx @ rz


def _project(positions: np.ndarray, projection: str, azimuth: float, elevation: float) -> np.ndarray:
    if projection == "xy":
        return positions[:, [0, 1]]
    if projection == "xz":
        return positions[:, [0, 2]]
    if projection == "yz":
        return positions[:, [1, 2]]
    rotated = positions @ _rotation_matrix(azimuth, elevation).T
    return rotated[:, [0, 1]]


def _bounds(snapshots, projection: str, azimuth: float, elevation: float) -> tuple[float, float, float, float]:
    positions = np.vstack([_project(snapshot.positions, projection, azimuth, elevation) for snapshot in snapshots])
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    padding = 0.08 * span if span > 0 else 1.0
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span + padding
    return cx - half, cx + half, cy - half, cy + half


def _bounds3d(snapshots) -> tuple[float, float, float, float, float, float]:
    positions = np.vstack([snapshot.positions for snapshot in snapshots])
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(maxs - mins)
    half = 0.55 * span if span > 0 else 1.0
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
        center[2] - half,
        center[2] + half,
    )


def _save_animation(fig, ani, output: Path, fps: int, dpi: int) -> None:
    if output.suffix.lower() == ".gif":
        ani.save(output, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        ani.save(output, fps=fps, dpi=dpi)
    plt.close(fig)


def _render_scatter(snapshots, args) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    xmin, xmax, ymin, ymax = _bounds(snapshots, args.projection, args.azimuth, args.elevation)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(args.projection[0] if args.projection != "camera" else "camera x")
    ax.set_ylabel(args.projection[1] if args.projection != "camera" else "camera y")

    first = snapshots[0]
    scatters = {}
    first_projected = _project(first.positions, args.projection, args.azimuth, args.elevation)
    for group, mask in iter_group_masks(first.group_id):
        scatters[group] = ax.scatter(
            first_projected[mask, 0],
            first_projected[mask, 1],
            s=args.point_size,
            alpha=0.85,
            linewidths=0,
            label=f"group {group}",
        )
    title = ax.set_title("")
    ax.legend(loc="upper right", markerscale=4)

    def update(frame: int):
        snapshot = snapshots[frame]
        azimuth = args.azimuth + (360.0 * frame / max(1, len(snapshots) - 1) if args.camera_orbit else 0.0)
        projected = _project(snapshot.positions, args.projection, azimuth, args.elevation)
        for group, mask in iter_group_masks(snapshot.group_id):
            scatters[group].set_offsets(projected[mask])
        title.set_text(f"step {snapshot.step}   t={snapshot.time:.3f}")
        return (*scatters.values(), title)

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / args.fps, blit=False)
    _save_animation(fig, ani, Path(args.output), args.fps, args.dpi)


def _render_density(snapshots, args) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    extent = _bounds(snapshots, args.projection, args.azimuth, args.elevation)
    image = ax.imshow(
        np.zeros((args.density_bins, args.density_bins)),
        extent=extent,
        origin="lower",
        cmap=args.cmap,
        interpolation="bilinear",
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    title = ax.set_title("")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="log density")

    def update(frame: int):
        snapshot = snapshots[frame]
        azimuth = args.azimuth + (360.0 * frame / max(1, len(snapshots) - 1) if args.camera_orbit else 0.0)
        projected = _project(snapshot.positions, args.projection, azimuth, args.elevation)
        hist, _, _ = np.histogram2d(
            projected[:, 0],
            projected[:, 1],
            bins=args.density_bins,
            range=((extent[0], extent[1]), (extent[2], extent[3])),
            weights=snapshot.masses,
        )
        image.set_data(np.log1p(hist.T))
        image.set_clim(0.0, max(float(np.log1p(hist).max()), 1.0e-12))
        title.set_text(f"step {snapshot.step}   t={snapshot.time:.3f}")
        return (image, title)

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / args.fps, blit=False)
    _save_animation(fig, ani, Path(args.output), args.fps, args.dpi)


def _render_scatter3d(snapshots, args) -> None:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    xmin, xmax, ymin, ymax, zmin, zmax = _bounds3d(snapshots)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    first = snapshots[0]
    scatters = {}
    for group, mask in iter_group_masks(first.group_id):
        scatter = ax.scatter(
            first.positions[mask, 0],
            first.positions[mask, 1],
            first.positions[mask, 2],
            s=args.point_size,
            alpha=0.75,
            linewidths=0,
            label=f"group {group}",
        )
        scatters[group] = scatter
    title = ax.set_title("")
    ax.legend(loc="upper right", markerscale=4)

    def update(frame: int):
        snapshot = snapshots[frame]
        for group, mask in iter_group_masks(snapshot.group_id):
            pts = snapshot.positions[mask]
            scatters[group]._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        azimuth = args.azimuth + (360.0 * frame / max(1, len(snapshots) - 1) if args.camera_orbit else 0.0)
        ax.view_init(elev=args.elevation, azim=azimuth)
        title.set_text(f"step {snapshot.step}   t={snapshot.time:.3f}")
        return (*scatters.values(), title)

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / args.fps, blit=False)
    _save_animation(fig, ani, Path(args.output), args.fps, args.dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render simulator snapshots to MP4 or GIF.")
    parser.add_argument("--input", default="experiments/validation/smoke_test")
    parser.add_argument("--output", default="galaxy_collision.mp4")
    parser.add_argument("--mode", choices=["scatter", "density", "scatter3d"], default="scatter")
    parser.add_argument("--projection", choices=["xy", "xz", "yz", "camera"], default="camera")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--density-bins", type=int, default=360)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--azimuth", type=float, default=35.0)
    parser.add_argument("--elevation", type=float, default=22.0)
    parser.add_argument("--camera-orbit", action="store_true")
    args = parser.parse_args()

    snapshots = load_snapshots(args.input, stride=args.stride)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {args.input}")

    if args.mode == "density":
        _render_density(snapshots, args)
    elif args.mode == "scatter3d":
        _render_scatter3d(snapshots, args)
    else:
        _render_scatter(snapshots, args)


if __name__ == "__main__":
    main()
