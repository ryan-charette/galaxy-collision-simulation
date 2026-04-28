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


def _plot_snapshot(snapshot_path: Path | None, input_directory: Path, output: Path) -> None:
    snapshot = load_snapshot(snapshot_path) if snapshot_path else load_latest_snapshot(input_directory)

    fig, ax = plt.subplots(figsize=(7, 7))
    for group, mask in iter_group_masks(snapshot.group_id):
        ax.scatter(
            snapshot.positions[mask, 0],
            snapshot.positions[mask, 1],
            s=2,
            alpha=0.8,
            linewidths=0,
            label=f"group {group}",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"step {snapshot.step}   t={snapshot.time:.3f}")
    ax.legend(loc="upper right", markerscale=4)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_diagnostics(input_directory: Path, output: Path) -> None:
    diagnostics_path = input_directory / "diagnostics.csv"
    if not diagnostics_path.exists():
        return

    diagnostics = load_diagnostics(diagnostics_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(diagnostics["time"], diagnostics["kinetic_energy"], label="kinetic")
    axes[0].plot(diagnostics["time"], diagnostics["potential_energy"], label="potential")
    axes[0].plot(diagnostics["time"], diagnostics["total_energy"], label="total")
    axes[0].set_ylabel("energy")
    axes[0].legend(loc="best")

    axes[1].plot(diagnostics["time"], diagnostics["momentum_x"], label="px")
    axes[1].plot(diagnostics["time"], diagnostics["momentum_y"], label="py")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("momentum")
    axes[1].legend(loc="best")

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
