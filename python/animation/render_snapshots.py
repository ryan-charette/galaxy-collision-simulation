from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from python.utils.snapshots import iter_group_masks, load_snapshots


def _bounds(snapshots) -> tuple[float, float, float, float]:
    positions = np.vstack([snapshot.positions for snapshot in snapshots])
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    padding = 0.08 * span if span > 0 else 1.0
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span + padding
    return cx - half, cx + half, cy - half, cy + half


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CSV snapshots to an MP4 or GIF animation.")
    parser.add_argument("--input", default="experiments/validation/smoke_test")
    parser.add_argument("--output", default="galaxy_collision.mp4")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    snapshots = load_snapshots(args.input, stride=args.stride)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {args.input}")

    fig, ax = plt.subplots(figsize=(7, 7))
    xmin, xmax, ymin, ymax = _bounds(snapshots)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    first = snapshots[0]
    scatters = {}
    for group, mask in iter_group_masks(first.group_id):
        scatters[group] = ax.scatter(
            first.positions[mask, 0],
            first.positions[mask, 1],
            s=2,
            alpha=0.85,
            linewidths=0,
            label=f"group {group}",
        )
    title = ax.set_title("")
    ax.legend(loc="upper right", markerscale=4)

    def update(frame: int):
        snapshot = snapshots[frame]
        for group, mask in iter_group_masks(snapshot.group_id):
            scatters[group].set_offsets(snapshot.positions[mask])
        title.set_text(f"step {snapshot.step}   t={snapshot.time:.3f}")
        return (*scatters.values(), title)

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / args.fps, blit=False)

    output = Path(args.output)
    if output.suffix.lower() == ".gif":
        ani.save(output, writer=animation.PillowWriter(fps=args.fps), dpi=args.dpi)
    else:
        ani.save(output, fps=args.fps, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
