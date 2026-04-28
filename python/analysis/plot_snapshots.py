from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from python.utils.snapshots import (
    iter_group_masks,
    load_diagnostics,
    load_latest_snapshot,
    load_snapshot,
)


def _scatter_projection(ax, snapshot, axes: tuple[int, int], labels: tuple[str, str]) -> None:
    for group, mask in iter_group_masks(snapshot.group_id):
        ax.scatter(
            snapshot.positions[mask, axes[0]],
            snapshot.positions[mask, axes[1]],
            s=2,
            alpha=0.8,
            linewidths=0,
            label=f"group {group}",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


def _plot_snapshot(snapshot_path: Path | None, input_directory: Path, output: Path) -> None:
    snapshot = load_snapshot(snapshot_path) if snapshot_path else load_latest_snapshot(input_directory)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _scatter_projection(axes[0], snapshot, (0, 1), ("x", "y"))
    _scatter_projection(axes[1], snapshot, (0, 2), ("x", "z"))
    _scatter_projection(axes[2], snapshot, (1, 2), ("y", "z"))
    axes[0].legend(loc="upper right", markerscale=4)
    fig.suptitle(f"step {snapshot.step}   t={snapshot.time:.3f}")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_diagnostics(input_directory: Path, output: Path) -> None:
    diagnostics_path = input_directory / "diagnostics.csv"
    if not diagnostics_path.exists():
        return

    diagnostics = load_diagnostics(diagnostics_path)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(diagnostics["time"], diagnostics["kinetic_energy"], label="kinetic")
    axes[0].plot(diagnostics["time"], diagnostics["potential_energy"], label="potential")
    axes[0].plot(diagnostics["time"], diagnostics["total_energy"], label="total")
    axes[0].set_ylabel("energy")
    axes[0].legend(loc="best")

    axes[1].plot(diagnostics["time"], diagnostics["momentum_x"], label="px")
    axes[1].plot(diagnostics["time"], diagnostics["momentum_y"], label="py")
    if "momentum_z" in diagnostics.dtype.names:
        axes[1].plot(diagnostics["time"], diagnostics["momentum_z"], label="pz")
    axes[1].set_ylabel("momentum")
    axes[1].legend(loc="best")

    if "angular_momentum_x" in diagnostics.dtype.names:
        axes[2].plot(diagnostics["time"], diagnostics["angular_momentum_x"], label="Lx")
        axes[2].plot(diagnostics["time"], diagnostics["angular_momentum_y"], label="Ly")
        axes[2].plot(diagnostics["time"], diagnostics["angular_momentum_z"], label="Lz")
    else:
        axes[2].plot(diagnostics["time"], diagnostics["angular_momentum"], label="Lz")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("angular momentum")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simulator snapshots and diagnostics.")
    parser.add_argument("--input", default="experiments/validation/smoke_test")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--output", default="snapshot.png")
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    input_directory = Path(args.input)
    output = Path(args.output)
    snapshot_path = Path(args.snapshot) if args.snapshot else None

    _plot_snapshot(snapshot_path, input_directory, output)

    if not args.no_diagnostics:
        diagnostics_output = (
            Path(args.diagnostics_output)
            if args.diagnostics_output
            else output.with_name(f"{output.stem}_diagnostics{output.suffix}")
        )
        _plot_diagnostics(input_directory, diagnostics_output)


if __name__ == "__main__":
    main()
