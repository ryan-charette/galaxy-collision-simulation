"""Snapshot loading utilities for simulator CSV and Parquet output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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
    if path.suffix == ".parquet":
        frame = _read_parquet_frame(path)
        if "time" in frame.columns and not frame.empty:
            return float(frame["time"].iloc[0])
        return 0.0
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if first_line.startswith("# time="):
        return float(first_line.split("=", 1)[1])
    return 0.0


def _step_from_name(path: Path) -> int:
    stem = path.stem
    if stem.startswith("snapshot_"):
        step_text = stem.split("_", 1)[1]
        if step_text.isdigit():
            return int(step_text)
    raise ValueError(f"Invalid snapshot filename: {path.name}")


def _is_snapshot_file(path: Path) -> bool:
    try:
        _step_from_name(path)
    except ValueError:
        return False
    return True


def list_snapshot_files(directory: str | Path) -> list[Path]:
    """Return snapshot CSV and Parquet files sorted by step number."""
    directory = Path(directory)
    files = [*directory.glob("snapshot_*.csv"), *directory.glob("snapshot_*.parquet")]
    files = [path for path in files if _is_snapshot_file(path)]
    return sorted(files, key=_step_from_name)


def _read_parquet_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            "Parquet snapshot loading requires pyarrow. Install project dependencies "
            "or run `pip install pyarrow`."
        ) from exc


def _snapshot_from_frame(path: Path, frame: pd.DataFrame, time: float) -> Snapshot:
    z = frame["z"] if "z" in frame.columns else np.zeros_like(frame["x"])
    vz = frame["vz"] if "vz" in frame.columns else np.zeros_like(frame["vx"])
    az = frame["az"] if "az" in frame.columns else np.zeros_like(frame["ax"])

    positions = np.column_stack([frame["x"], frame["y"], z])
    velocities = np.column_stack([frame["vx"], frame["vy"], vz])
    accelerations = np.column_stack([frame["ax"], frame["ay"], az])

    return Snapshot(
        step=_step_from_name(path),
        time=time,
        ids=np.asarray(frame["id"], dtype=np.int64),
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        masses=np.asarray(frame["mass"], dtype=float),
        group_id=np.asarray(frame["group_id"], dtype=np.int64),
        path=path,
    )


def load_snapshot(path: str | Path) -> Snapshot:
    """Load one C++ CSV or Parquet snapshot."""
    path = Path(path)
    if path.suffix == ".parquet":
        frame = _read_parquet_frame(path)
        time = float(frame["time"].iloc[0]) if "time" in frame.columns and not frame.empty else 0.0
        return _snapshot_from_frame(path, frame, time)

    with path.open("r", encoding="utf-8") as handle:
        skip_header = 1 if handle.readline().startswith("# time=") else 0
    data = np.genfromtxt(path, delimiter=",", names=True, skip_header=skip_header)
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    names = data.dtype.names or ()
    z = data["z"] if "z" in names else np.zeros_like(data["x"])
    vz = data["vz"] if "vz" in names else np.zeros_like(data["vx"])
    az = data["az"] if "az" in names else np.zeros_like(data["ax"])

    frame = pd.DataFrame(
        {
            "id": data["id"],
            "group_id": data["group_id"],
            "mass": data["mass"],
            "x": data["x"],
            "y": data["y"],
            "z": z,
            "vx": data["vx"],
            "vy": data["vy"],
            "vz": vz,
            "ax": data["ax"],
            "ay": data["ay"],
            "az": az,
        }
    )
    return _snapshot_from_frame(path, frame, _read_time(path))


def load_latest_snapshot(directory: str | Path) -> Snapshot:
    """Load the highest-step snapshot in a directory."""
    files = list_snapshot_files(directory)
    if not files:
        raise FileNotFoundError(f"No snapshot_*.csv or snapshot_*.parquet files found in {Path(directory)}")
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
