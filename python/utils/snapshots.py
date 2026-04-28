"""Snapshot loading utilities for simulator CSV output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Snapshot:
    step: int
    time: float
    ids: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    masses: np.ndarray
    group_id: np.ndarray
    path: Path | None = None


def _read_time(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if first_line.startswith("# time="):
        return float(first_line.split("=", 1)[1])
    return 0.0


def _step_from_name(path: Path) -> int:
    stem = path.stem
    if stem.startswith("snapshot_"):
        return int(stem.split("_", 1)[1])
    return 0


def list_snapshot_files(directory: str | Path) -> list[Path]:
    """Return snapshot CSV files sorted by step number."""
    directory = Path(directory)
    return sorted(directory.glob("snapshot_*.csv"), key=_step_from_name)


def load_snapshot(path: str | Path) -> Snapshot:
    """Load one C++ CSV snapshot."""
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True, comments="#")
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    names = data.dtype.names or ()
    z = data["z"] if "z" in names else np.zeros_like(data["x"])
    vz = data["vz"] if "vz" in names else np.zeros_like(data["vx"])
    az = data["az"] if "az" in names else np.zeros_like(data["ax"])

    positions = np.column_stack([data["x"], data["y"], z])
    velocities = np.column_stack([data["vx"], data["vy"], vz])
    accelerations = np.column_stack([data["ax"], data["ay"], az])

    return Snapshot(
        step=_step_from_name(path),
        time=_read_time(path),
        ids=np.asarray(data["id"], dtype=np.int64),
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        masses=np.asarray(data["mass"], dtype=float),
        group_id=np.asarray(data["group_id"], dtype=np.int64),
        path=path,
    )


def load_latest_snapshot(directory: str | Path) -> Snapshot:
    """Load the highest-step snapshot in a directory."""
    files = list_snapshot_files(directory)
    if not files:
        raise FileNotFoundError(f"No snapshot_*.csv files found in {Path(directory)}")
    return load_snapshot(files[-1])


def load_snapshots(directory: str | Path, stride: int = 1) -> list[Snapshot]:
    """Load all snapshots from a directory, optionally thinning by stride."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    return [load_snapshot(path) for path in list_snapshot_files(directory)[::stride]]


def load_diagnostics(path_or_directory: str | Path) -> np.ndarray:
    """Load diagnostics.csv from either a file path or snapshot directory."""
    path = Path(path_or_directory)
    if path.is_dir():
        path = path / "diagnostics.csv"
    return np.genfromtxt(path, delimiter=",", names=True)


def iter_group_masks(group_id: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    """Yield stable group masks for plotting."""
    for group in sorted(int(value) for value in np.unique(group_id)):
        yield group, group_id == group


def load_placeholder_snapshot(path: str | Path) -> Snapshot:
    """Backward-compatible synthetic snapshot for older notebooks."""
    _ = Path(path)
    theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    r = 0.2 + 0.8 * np.sqrt(np.linspace(0.0, 1.0, 512))
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
    zeros = np.zeros_like(positions)
    return Snapshot(
        step=0,
        time=0.0,
        ids=np.arange(len(theta)),
        positions=positions,
        velocities=zeros,
        accelerations=zeros,
        masses=np.ones(len(theta)) / len(theta),
        group_id=np.zeros(len(theta), dtype=np.int32),
        path=None,
    )
