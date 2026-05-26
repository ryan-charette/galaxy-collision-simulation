"""Dataset loading, validation, and model artifact helpers."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from python.ml import DATASET_SCHEMA_VERSION


@dataclass(frozen=True)
class Split:
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except ImportError as exc:
            raise RuntimeError("Parquet datasets require pyarrow. Use CSV or install dependencies.") from exc
    else:
        frame = pd.read_csv(path)

    if "dataset_schema_version" in frame.columns:
        versions = {str(value) for value in frame["dataset_schema_version"].dropna().unique()}
        if versions and versions != {DATASET_SCHEMA_VERSION}:
            raise ValueError(f"Unsupported dataset schema version(s): {sorted(versions)}")
    return frame


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def completed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        return frame.copy()
    return frame[frame["status"] == "completed"].copy()


def finite_target_rows(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    require_columns(frame, targets)
    filtered = frame.copy()
    for target in targets:
        values = pd.to_numeric(filtered[target], errors="coerce")
        filtered = filtered[np.isfinite(values)]
    return filtered.copy()


def train_test_split_indices(count: int, test_fraction: float, seed: int) -> Split:
    if count <= 0:
        raise ValueError("Cannot split an empty dataset")
    if count == 1:
        return Split(train_indices=np.array([0]), test_indices=np.array([0]))

    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    test_count = max(1, int(round(count * test_fraction)))
    test_count = min(test_count, count - 1)
    test_indices = np.sort(indices[:test_count])
    train_indices = np.sort(indices[test_count:])
    return Split(train_indices=train_indices, test_indices=test_indices)


def numeric_matrix(frame: pd.DataFrame, targets: list[str]) -> np.ndarray:
    require_columns(frame, targets)
    columns = [pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float) for target in targets]
    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Target matrix contains missing or non-finite values")
    return matrix


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_model_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    version = bundle.get("dataset_schema_version")
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model dataset schema version: {version}")
    return bundle
