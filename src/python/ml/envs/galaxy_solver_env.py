"""Gymnasium-compatible contextual bandit environment for solver tuning."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from python.ml.datasets import load_model_bundle
from python.ml.features import estimate_tree_depth
from python.ml.recommend_config import add_force_predictions, direct_runtime_estimate, predict_cost
from scripts.experiment_utils import run_simulator
from scripts import sweep as sweep_runner

try:  # Gymnasium is optional; the local smoke path uses the fallback classes.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when Gymnasium is unavailable
    gym = None

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def sample(self) -> int:
            return int(np.random.default_rng().integers(0, self.n))

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _Env:
        pass

    spaces = _Spaces()


@dataclass(frozen=True)
class SolverAction:
    """One solver/configuration action available to the tuning environment."""

    solver: str
    tree_theta: float
    tree_leaf_capacity: int
    fmm_expansion_order: int
    output_format: str = "csv"
    snapshot_every: int = 1


@dataclass
class RewardWeights:
    """Weights used to combine runtime, accuracy, and conservation penalties."""

    runtime: float = 1.0
    force_error: float = 100.0
    energy_drift: float = 10.0
    momentum_drift: float = 10.0
    invalid_config: float = 100.0


OBSERVATION_KEYS = [
    "n_particles",
    "current_step",
    "current_time",
    "solver_index",
    "tree_theta",
    "tree_leaf_capacity",
    "fmm_expansion_order",
    "recent_step_time",
    "energy_drift_so_far",
    "momentum_drift_so_far",
    "bounding_box_size",
    "density_summary",
    "velocity_summary",
]

SOLVER_TO_INDEX = {
    "direct": 0,
    "tree": 1,
    "fmm": 2,
    "cuda-direct": 3,
    "cuda-tree": 4,
    "cuda-fmm": 5,
}


def default_actions(hardware: str = "cpu") -> list[SolverAction]:
    """Return the default discrete solver-action grid for CPU or CUDA hardware."""
    solvers = ["direct", "tree", "fmm"]
    if hardware == "cuda":
        solvers.extend(["cuda-tree", "cuda-fmm"])
    actions: list[SolverAction] = []
    for solver in solvers:
        if solver == "direct":
            actions.append(SolverAction(solver, 0.6, 16, 0))
            continue
        for theta in (0.5, 0.6, 0.7, 1.0):
            for leaf_capacity in (8, 16, 32):
                expansion_orders = (0,) if "tree" in solver else (0, 2, 4)
                for expansion_order in expansion_orders:
                    actions.append(SolverAction(solver, theta, leaf_capacity, expansion_order))
    return actions


def observation_from_context(context: dict[str, Any]) -> np.ndarray:
    """Convert a solver-tuning context dictionary into an observation vector."""
    return np.asarray([float(context.get(key, 0.0)) for key in OBSERVATION_KEYS], dtype=np.float32)


def default_contexts(
    n_particles: list[int],
    steps: int,
    dt: float,
    softening: float,
    hardware: str,
    output_format: str,
) -> list[dict[str, Any]]:
    """Create deterministic starter contexts for solver-tuning experiments."""
    contexts = []
    for count in n_particles:
        bounding_box = 2.0
        contexts.append(
            {
                "n_particles": count,
                "steps": steps,
                "dt": dt,
                "softening": softening,
                "hardware_type": hardware,
                "output_format": output_format,
                "current_step": 0,
                "current_time": 0.0,
                "solver": "tree",
                "solver_index": SOLVER_TO_INDEX["tree"],
                "tree_theta": 0.6,
                "tree_leaf_capacity": 16,
                "fmm_expansion_order": 0,
                "recent_step_time": 0.0,
                "energy_drift_so_far": 0.0,
                "momentum_drift_so_far": 0.0,
                "bounding_box_size": bounding_box,
                "density_summary": count / (bounding_box**3),
                "velocity_summary": 0.2,
                "estimated_tree_depth": estimate_tree_depth(count, 16),
            }
        )
    return contexts


def action_frame(context: dict[str, Any], action: SolverAction) -> pd.DataFrame:
    """Represent one context/action pair as model-feature dataframe rows."""
    row = {
        "solver": action.solver,
        "n_particles": int(context["n_particles"]),
        "steps": int(context["steps"]),
        "dt": float(context["dt"]),
        "softening": float(context["softening"]),
        "tree_theta": action.tree_theta,
        "tree_leaf_capacity": action.tree_leaf_capacity,
        "fmm_expansion_order": action.fmm_expansion_order,
        "hardware_type": context.get("hardware_type", "cpu"),
        "output_format": action.output_format,
        "initial_density_summary": context.get("density_summary", 0.0),
        "initial_velocity_summary": context.get("velocity_summary", 0.0),
        "initial_bounding_box": context.get("bounding_box_size", 0.0),
        "estimated_tree_depth": estimate_tree_depth(
            int(context["n_particles"]),
            action.tree_leaf_capacity,
        ),
    }
    return pd.DataFrame([row])


def diagnostic_drifts(output_dir: Path) -> tuple[float, float]:
    """Compute final relative energy and momentum drift from a diagnostics file."""
    rows = []
    path = output_dir / "diagnostics.csv"
    if not path.exists():
        return math.nan, math.nan
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return math.nan, math.nan
    first, last = rows[0], rows[-1]
    initial_energy = float(first["total_energy"])
    final_energy = float(last["total_energy"])
    initial_momentum = np.array(
        [float(first["momentum_x"]), float(first["momentum_y"]), float(first["momentum_z"])],
        dtype=float,
    )
    final_momentum = np.array(
        [float(last["momentum_x"]), float(last["momentum_y"]), float(last["momentum_z"])],
        dtype=float,
    )
    energy = abs(final_energy - initial_energy) / max(abs(initial_energy), 1.0e-12)
    momentum = float(
        np.linalg.norm(final_momentum - initial_momentum)
        / max(np.linalg.norm(initial_momentum), 1.0e-12)
    )
    return energy, momentum


class GalaxySolverEnv(gym.Env if gym is not None else _Env):
    """One-step solver tuning environment.

    In `cheap` mode, rewards are computed from supervised model predictions or
    deterministic heuristics. In `real` mode, one short simulator episode is
    executed for the selected action.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        contexts: list[dict[str, Any]] | None = None,
        actions: list[SolverAction] | None = None,
        mode: str = "cheap",
        cost_model_path: str | Path | None = None,
        force_model_path: str | Path | None = None,
        base_config: str | Path = "configs/smoke_test.toml",
        executable: str | Path | None = None,
        output_root: str | Path = "experiments/rl_env",
        reward_weights: RewardWeights | None = None,
        seed: int | None = None,
    ) -> None:
        self.mode = mode
        self.contexts = contexts or default_contexts([256, 512, 1024], 20, 0.01, 0.02, "cpu", "csv")
        self.actions = actions or default_actions(str(self.contexts[0].get("hardware_type", "cpu")))
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(seed)
        self.current_context: dict[str, Any] | None = None
        self.current_observation: np.ndarray | None = None
        self.cost_model = (
            load_model_bundle(cost_model_path)
            if cost_model_path and Path(cost_model_path).exists()
            else None
        )
        self.force_model = (
            load_model_bundle(force_model_path)
            if force_model_path and Path(force_model_path).exists()
            else None
        )
        self.base_config = Path(base_config)
        if self.mode == "real":
            self.executable = sweep_runner.resolve_executable(str(executable) if executable else None)
        else:
            self.executable = Path(executable) if executable else Path("build/fmm_galaxy_sim")
        self.output_root = Path(output_root)
        self.episode_index = 0
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(OBSERVATION_KEYS),),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a one-step episode and return the initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "context" in options:
            context = dict(options["context"])
        else:
            context = dict(self.contexts[int(self.rng.integers(0, len(self.contexts)))])
        self.current_context = context
        self.current_observation = observation_from_context(context)
        return self.current_observation.copy(), {"context": context}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Evaluate one solver action and return a terminated Gymnasium step tuple."""
        if self.current_context is None or self.current_observation is None:
            self.reset()
        assert self.current_context is not None
        assert self.current_observation is not None
        action_spec = self.actions[int(action)]
        metrics = self._evaluate_action(self.current_context, action_spec)
        reward = self._reward(action_spec, metrics)
        info = {"action": action_spec.__dict__, "metrics": metrics}
        return self.current_observation.copy(), reward, True, False, info

    def _evaluate_action(self, context: dict[str, Any], action: SolverAction) -> dict[str, float]:
        if self.mode == "real":
            return self._evaluate_real(context, action)
        return self._evaluate_cheap(context, action)

    def _evaluate_cheap(self, context: dict[str, Any], action: SolverAction) -> dict[str, float]:
        frame = predict_cost(self.cost_model, action_frame(context, action))
        direct_frame = predict_cost(
            self.cost_model,
            action_frame(context, SolverAction("direct", 0.6, 16, 0)),
        )
        direct_runtime = direct_runtime_estimate(direct_frame)
        frame = add_force_predictions(self.force_model, frame, direct_runtime)
        row = frame.iloc[0]
        median_step_time = float(row.get("predicted_median_step_time", 1.0))
        runtime = median_step_time * float(context["steps"])
        relative_force_error = float(row.get("predicted_relative_force_rmse", math.nan))
        if not math.isfinite(relative_force_error):
            theta_factor = max(action.tree_theta, 0.1)
            order_factor = 1.0 / (1.0 + max(action.fmm_expansion_order, 0))
            relative_force_error = 0.0 if action.solver == "direct" else 0.01 * theta_factor * order_factor
        energy_drift = float(row.get("predicted_energy_drift_final", math.nan))
        momentum_drift = float(row.get("predicted_momentum_drift_final", math.nan))
        return {
            "runtime_cost": runtime,
            "force_error": relative_force_error,
            "energy_drift": 0.0 if not math.isfinite(energy_drift) else abs(energy_drift),
            "momentum_drift": 0.0 if not math.isfinite(momentum_drift) else abs(momentum_drift),
        }

    def _evaluate_real(self, context: dict[str, Any], action: SolverAction) -> dict[str, float]:
        config = sweep_runner.load_toml(self.base_config)
        config = deepcopy(config)
        output_dir = self.output_root / f"episode_{self.episode_index:05d}_{action.solver}"
        self.episode_index += 1
        sweep_runner.set_dotted(config, "simulation.solver", action.solver)
        sweep_runner.set_dotted(config, "simulation.n_particles", int(context["n_particles"]))
        sweep_runner.set_dotted(config, "simulation.steps", int(context["steps"]))
        sweep_runner.set_dotted(config, "simulation.dt", float(context["dt"]))
        sweep_runner.set_dotted(config, "simulation.tree_theta", action.tree_theta)
        sweep_runner.set_dotted(config, "simulation.tree_leaf_capacity", action.tree_leaf_capacity)
        sweep_runner.set_dotted(config, "simulation.fmm_expansion_order", action.fmm_expansion_order)
        sweep_runner.set_dotted(config, "simulation.snapshot_every", action.snapshot_every)
        sweep_runner.set_dotted(config, "output.format", action.output_format)
        sweep_runner.set_dotted(config, "output.directory", output_dir.as_posix())
        sweep_runner.sync_galaxy_particle_counts(config)
        config_path = self.output_root / "configs" / f"{output_dir.name}.toml"
        sweep_runner.write_toml(config_path, config)
        completed = run_simulator(
            self.executable,
            config_path,
            output_dir,
            cwd=Path.cwd(),
        )
        runtime = completed.seconds
        if completed.exit_code != 0:
            return {
                "runtime_cost": runtime,
                "force_error": 1.0,
                "energy_drift": 1.0,
                "momentum_drift": 1.0,
                "invalid_config": 1.0,
            }
        energy, momentum = diagnostic_drifts(output_dir)
        return {
            "runtime_cost": runtime,
            "force_error": math.nan,
            "energy_drift": energy,
            "momentum_drift": momentum,
        }

    def _reward(self, action: SolverAction, metrics: dict[str, float]) -> float:
        invalid = 0.0
        hardware = str((self.current_context or {}).get("hardware_type", "cpu"))
        if hardware == "cpu" and action.solver.startswith("cuda"):
            invalid = 1.0
        if action.solver not in SOLVER_TO_INDEX:
            invalid = 1.0
        force_error = (
            0.0 if math.isnan(metrics.get("force_error", math.nan)) else metrics.get("force_error", 0.0)
        )
        energy = (
            0.0 if math.isnan(metrics.get("energy_drift", math.nan)) else metrics.get("energy_drift", 0.0)
        )
        momentum = (
            0.0
            if math.isnan(metrics.get("momentum_drift", math.nan))
            else metrics.get("momentum_drift", 0.0)
        )
        return -(
            self.reward_weights.runtime * metrics.get("runtime_cost", 0.0)
            + self.reward_weights.force_error * force_error
            + self.reward_weights.energy_drift * energy
            + self.reward_weights.momentum_drift * momentum
            + self.reward_weights.invalid_config * invalid
        )

    def action_table(self) -> list[dict[str, Any]]:
        return [action.__dict__ for action in self.actions]

    def save_action_table(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.action_table(), indent=2) + "\n", encoding="utf-8")
