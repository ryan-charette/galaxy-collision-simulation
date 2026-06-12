"""Feature extraction and lightweight regression models."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd


RUNTIME_NUMERIC_FEATURES = [
    "n_particles",
    "steps",
    "dt",
    "softening",
    "tree_theta",
    "tree_leaf_capacity",
    "fmm_expansion_order",
    "initial_density_summary",
    "initial_velocity_summary",
    "initial_bounding_box",
    "estimated_tree_depth",
]
RUNTIME_CATEGORICAL_FEATURES = ["solver", "hardware_type", "output_format"]

FORCE_NUMERIC_FEATURES = [
    "n_particles",
    "softening",
    "tree_theta",
    "tree_leaf_capacity",
    "fmm_expansion_order",
    "runtime_approx",
    "runtime_direct",
    "initial_density_summary",
    "initial_velocity_summary",
    "initial_bounding_box",
    "estimated_tree_depth",
]
FORCE_CATEGORICAL_FEATURES = ["solver"]


def estimate_tree_depth(n_particles: int, leaf_capacity: int) -> int:
    """Estimate octree depth from particle count and leaf capacity."""
    if n_particles <= leaf_capacity:
        return 1
    ratio = max(n_particles / max(leaf_capacity, 1), 1.0)
    return max(1, int(math.ceil(math.log(ratio, 8.0))) + 1)


def runtime_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return available numeric and categorical columns for runtime prediction."""
    numeric = [column for column in RUNTIME_NUMERIC_FEATURES if column in frame.columns]
    categorical = [column for column in RUNTIME_CATEGORICAL_FEATURES if column in frame.columns]
    return numeric, categorical


def force_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return available numeric and categorical columns for force-error prediction."""
    numeric = [column for column in FORCE_NUMERIC_FEATURES if column in frame.columns]
    categorical = [column for column in FORCE_CATEGORICAL_FEATURES if column in frame.columns]
    return numeric, categorical


@dataclass
class FeatureTransformer:
    """Median-impute numeric columns and one-hot encode categorical columns."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> "FeatureTransformer":
        """Fit numeric medians and categorical vocabularies."""
        self.numeric_medians = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            self.numeric_medians[column] = median

        self.categories = {}
        for column in self.categorical_columns:
            values = frame[column].fillna("__missing__").astype(str)
            self.categories[column] = sorted(values.unique().tolist())

        self.feature_names = [
            *self.numeric_columns,
            *[
                f"{column}={category}"
                for column in self.categorical_columns
                for category in self.categories[column]
            ],
        ]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe into a numeric feature matrix."""
        numeric_parts = []
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.numeric_medians[column])
            numeric_parts.append(values.to_numpy(dtype=float))

        categorical_parts = []
        for column in self.categorical_columns:
            values = frame[column].fillna("__missing__").astype(str)
            for category in self.categories[column]:
                categorical_parts.append((values == category).astype(float).to_numpy())

        if numeric_parts or categorical_parts:
            return np.column_stack([*numeric_parts, *categorical_parts])
        return np.zeros((len(frame), 0), dtype=float)


@dataclass
class StandardScaler:
    """Columnwise standard scaler with dependency-free serialization."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, matrix: np.ndarray) -> "StandardScaler":
        """Fit means and standard deviations."""
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale == 0.0] = 1.0
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Scale a feature matrix using fitted statistics."""
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler is not fitted")
        return (matrix - self.mean) / self.scale


@dataclass
class NumpyLinearRegressor:
    """Small ridge-regularized linear regressor implemented with NumPy."""

    ridge_alpha: float = 1.0e-6
    coefficients: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "NumpyLinearRegressor":
        """Fit linear coefficients for one or more targets."""
        if targets.ndim == 1:
            targets = targets[:, None]
        design = np.column_stack([np.ones(features.shape[0]), features])
        penalty = self.ridge_alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict targets from a feature matrix."""
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted")
        design = np.column_stack([np.ones(features.shape[0]), features])
        return design @ self.coefficients


@dataclass
class MeanRegressor:
    """Baseline regressor that always predicts the training target mean."""

    mean: np.ndarray | None = None

    def fit(self, targets: np.ndarray) -> "MeanRegressor":
        """Fit the target mean."""
        if targets.ndim == 1:
            targets = targets[:, None]
        self.mean = targets.mean(axis=0)
        return self

    def predict(self, count: int) -> np.ndarray:
        """Return `count` copies of the fitted mean."""
        if self.mean is None:
            raise RuntimeError("Baseline is not fitted")
        return np.tile(self.mean, (count, 1))


def make_regressor(model_type: str, random_state: int) -> Any:
    """Construct a supported regression model by name."""
    if model_type == "linear":
        return NumpyLinearRegressor()

    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except ImportError as exc:
        raise RuntimeError(
            f"Model type '{model_type}' requires scikit-learn. Install project dependencies "
            "or use --model linear."
        ) from exc

    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=random_state, min_samples_leaf=2)
    if model_type == "gradient_boosting":
        return MultiOutputRegressor(GradientBoostingRegressor(random_state=random_state))
    raise ValueError(f"Unknown model type: {model_type}")


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Run feature transformation, scaling, and prediction for a saved model bundle."""
    transformer: FeatureTransformer = bundle["feature_transformer"]
    scaler: StandardScaler = bundle["feature_scaler"]
    features = scaler.transform(transformer.transform(frame))
    predictions = bundle["model"].predict(features)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    return predictions
