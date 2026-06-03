"""Baseline and learned policies for solver-tuning environments."""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from python.ml import DATASET_SCHEMA_VERSION
from python.ml.envs.galaxy_solver_env import GalaxySolverEnv, OBSERVATION_KEYS, SolverAction


POLICY_SCHEMA_VERSION = "0.1.0"


class SolverPolicy(Protocol):
    name: str

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        env: GalaxySolverEnv,
        rng: np.random.Generator | None = None,
    ) -> int:
        ...


def _action_score(
    action: SolverAction,
    solver: str,
    tree_theta: float | None,
    tree_leaf_capacity: int | None,
    fmm_expansion_order: int | None,
) -> float:
    if action.solver != solver:
        return float("inf")
    score = 0.0
    if tree_theta is not None:
        score += abs(action.tree_theta - tree_theta)
    if tree_leaf_capacity is not None:
        score += abs(action.tree_leaf_capacity - tree_leaf_capacity) / max(tree_leaf_capacity, 1)
    if fmm_expansion_order is not None:
        score += abs(action.fmm_expansion_order - fmm_expansion_order)
    return score


def find_action_index(
    actions: list[SolverAction],
    solver: str,
    tree_theta: float | None = None,
    tree_leaf_capacity: int | None = None,
    fmm_expansion_order: int | None = None,
) -> int:
    scores = [
        _action_score(action, solver, tree_theta, tree_leaf_capacity, fmm_expansion_order)
        for action in actions
    ]
    best_index = int(np.argmin(scores))
    if not np.isfinite(scores[best_index]):
        return 0
    return best_index


@dataclass
class FixedActionPolicy:
    name: str
    solver: str
    tree_theta: float | None = None
    tree_leaf_capacity: int | None = None
    fmm_expansion_order: int | None = None

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        env: GalaxySolverEnv,
        rng: np.random.Generator | None = None,
    ) -> int:
        return find_action_index(
            env.actions,
            self.solver,
            self.tree_theta,
            self.tree_leaf_capacity,
            self.fmm_expansion_order,
        )


@dataclass
class FastestSupervisedPolicy:
    """Choose the highest predicted cheap-mode reward across the action table."""

    name: str = "fastest_supervised"

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        env: GalaxySolverEnv,
        rng: np.random.Generator | None = None,
    ) -> int:
        context = dict(info.get("context") or env.current_context or {})
        rewards = []
        for action in env.actions:
            metrics = env._evaluate_cheap(context, action)
            rewards.append(env._reward(action, metrics))
        return int(np.argmax(rewards))


@dataclass
class LinearContextualBanditPolicy:
    """Independent ridge-regression reward model per discrete action."""

    name: str
    a_matrices: np.ndarray
    b_vectors: np.ndarray

    @classmethod
    def create(
        cls,
        action_count: int,
        feature_count: int,
        ridge_alpha: float,
        name: str = "linear_contextual_bandit",
    ) -> "LinearContextualBanditPolicy":
        matrices = np.tile(np.eye(feature_count) * ridge_alpha, (action_count, 1, 1))
        vectors = np.zeros((action_count, feature_count), dtype=float)
        return cls(name=name, a_matrices=matrices, b_vectors=vectors)

    @property
    def action_count(self) -> int:
        return int(self.a_matrices.shape[0])

    @staticmethod
    def features(observation: np.ndarray) -> np.ndarray:
        return np.concatenate(([1.0], np.asarray(observation, dtype=float)))

    def action_values(self, observation: np.ndarray) -> np.ndarray:
        x = self.features(observation)
        values = []
        for action_index in range(self.action_count):
            theta = np.linalg.solve(self.a_matrices[action_index], self.b_vectors[action_index])
            values.append(float(x @ theta))
        return np.asarray(values, dtype=float)

    def select_action(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        env: GalaxySolverEnv,
        rng: np.random.Generator | None = None,
        epsilon: float = 0.0,
    ) -> int:
        if self.action_count != len(env.actions):
            raise ValueError(
                f"Policy has {self.action_count} actions but environment has {len(env.actions)}"
            )
        rng = rng or np.random.default_rng()
        if epsilon > 0.0 and float(rng.random()) < epsilon:
            return int(rng.integers(0, self.action_count))
        return int(np.argmax(self.action_values(observation)))

    def update(self, observation: np.ndarray, action_index: int, reward: float) -> None:
        x = self.features(observation)
        self.a_matrices[action_index] += np.outer(x, x)
        self.b_vectors[action_index] += reward * x

    def to_bundle(self, actions: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "policy_kind": "linear_contextual_bandit",
            "name": self.name,
            "observation_keys": OBSERVATION_KEYS,
            "actions": actions,
            "a_matrices": self.a_matrices,
            "b_vectors": self.b_vectors,
            "metadata": metadata,
        }

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> "LinearContextualBanditPolicy":
        if bundle.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset schema version: {bundle.get('dataset_schema_version')}")
        if bundle.get("policy_schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported policy schema version: {bundle.get('policy_schema_version')}")
        return cls(
            name=str(bundle.get("name", "linear_contextual_bandit")),
            a_matrices=np.asarray(bundle["a_matrices"], dtype=float),
            b_vectors=np.asarray(bundle["b_vectors"], dtype=float),
        )


def fixed_baseline_policies() -> list[SolverPolicy]:
    return [
        FixedActionPolicy("always_direct", "direct"),
        FixedActionPolicy("always_tree_theta_0.6", "tree", tree_theta=0.6, tree_leaf_capacity=16),
        FixedActionPolicy(
            "always_fmm_p4",
            "fmm",
            tree_theta=0.6,
            tree_leaf_capacity=16,
            fmm_expansion_order=4,
        ),
    ]


def all_baseline_policies() -> list[SolverPolicy]:
    return [*fixed_baseline_policies(), FastestSupervisedPolicy()]


def save_policy_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_policy_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema version: {bundle.get('dataset_schema_version')}")
    if bundle.get("policy_schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported policy schema version: {bundle.get('policy_schema_version')}")
    return bundle


def load_policy(path: str | Path) -> LinearContextualBanditPolicy:
    return LinearContextualBanditPolicy.from_bundle(load_policy_bundle(path))


def validate_policy_actions(bundle: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    saved_actions = bundle.get("actions")
    if saved_actions != actions:
        raise ValueError(
            "Policy action table does not match the environment. "
            "Use the same hardware/action grid used during training."
        )


def evaluate_policies(
    env: GalaxySolverEnv,
    policies: list[SolverPolicy],
    contexts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    contexts = contexts or env.contexts
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for context_index, context in enumerate(contexts):
            observation, info = env.reset(options={"context": context})
            action_index = policy.select_action(observation, info, env, rng=env.rng)
            _, reward, terminated, truncated, step_info = env.step(action_index)
            action = step_info["action"]
            metrics = step_info["metrics"]
            row: dict[str, Any] = {
                "policy": policy.name,
                "context_index": context_index,
                "terminated": terminated,
                "truncated": truncated,
                "reward": reward,
                "action_index": action_index,
                "solver": action["solver"],
                "tree_theta": action["tree_theta"],
                "tree_leaf_capacity": action["tree_leaf_capacity"],
                "fmm_expansion_order": action["fmm_expansion_order"],
                "n_particles": context.get("n_particles"),
                "steps": context.get("steps"),
                "dt": context.get("dt"),
                "softening": context.get("softening"),
                "hardware_type": context.get("hardware_type"),
            }
            row.update(metrics)
            rows.append(row)
    return rows


def summarize_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for policy, policy_rows in sorted(by_policy.items()):
        rewards = np.asarray([float(row["reward"]) for row in policy_rows], dtype=float)
        runtimes = np.asarray([float(row.get("runtime_cost", np.nan)) for row in policy_rows], dtype=float)
        force_errors = np.asarray([float(row.get("force_error", np.nan)) for row in policy_rows], dtype=float)
        solvers = sorted({str(row["solver"]) for row in policy_rows})
        summaries.append(
            {
                "policy": policy,
                "episodes": len(policy_rows),
                "mean_reward": float(np.nanmean(rewards)),
                "mean_runtime_cost": float(np.nanmean(runtimes)),
                "mean_force_error": float(np.nanmean(force_errors)),
                "selected_solvers": ", ".join(solvers),
            }
        )
    summaries.sort(key=lambda row: row["mean_reward"], reverse=True)
    return summaries


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def policy_markdown_report(
    title: str,
    metadata: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> str:
    lines = [f"# {title}", "", "## Metadata", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Policy Comparison",
            "",
            "| Policy | Episodes | Mean reward | Mean runtime cost | Mean force error | Selected solvers |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| `{row['policy']}` | {row['episodes']} | {row['mean_reward']:.6g} | "
            f"{row['mean_runtime_cost']:.6g} | {row['mean_force_error']:.6g} | "
            f"{row['selected_solvers']} |"
        )
    lines.append("")
    return "\n".join(lines)
