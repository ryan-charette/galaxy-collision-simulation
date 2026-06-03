"""Parquet conversion helpers for simulator snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SNAPSHOT_COLUMNS = [
    "id",
    "group_id",
    "mass",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
]


def csv_snapshot_to_parquet(csv_path: str | Path, parquet_path: str | Path, step: int, time: float) -> None:
    """Convert a simulator CSV snapshot into an Apache Parquet snapshot."""
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    frame = pd.read_csv(csv_path, comment="#")
    frame = frame[SNAPSHOT_COLUMNS]
    frame.insert(0, "time", float(time))
    frame.insert(0, "step", int(step))
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(parquet_path, index=False, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install project dependencies or run "
            "`pip install pyarrow` in the Python environment used by FMM_GALAXY_PYTHON."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one simulator CSV snapshot to Parquet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--time", type=float, required=True)
    args = parser.parse_args()
    csv_snapshot_to_parquet(args.input, args.output, args.step, args.time)


if __name__ == "__main__":
    main()
