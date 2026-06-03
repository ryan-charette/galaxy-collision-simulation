"""Train a first contextual-bandit policy for solver tuning."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from python.ml.envs.galaxy_solver_env import (
    GalaxySolverEnv,
    OBSERVATION_KEYS,
    RewardWeights,
    default_actions,
    default_contexts,
)
from python.ml.rl.baselines import (
    LinearContextualBanditPolicy,
    all_baseline_policies,
    evaluate_policies,
    policy_markdown_report,
    save_policy_bundle,
    summarize_policy_rows,
    write_rows_csv,
    write_summary_json,
)


def make_contexts(args: argparse.Namespace) -> list[dict[str, Any]]:
    return default_contexts(
        n_particles=args.n_particles,
        steps=args.steps,
        dt=args.dt,
        softening=args.softening,
        hardware=args.hardware,
        output_format=args.output_format,
    )


def make_reward_weights(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(
        runtime=args.reward_runtime,
        force_error=args.reward_force_error,
        energy_drift=args.reward_energy_drift,
        momentum_drift=args.reward_momentum_drift,
        invalid_config=args.reward_invalid_config,
    )


def make_env(args: argparse.Namespace, contexts: list[dict[str, Any]]) -> GalaxySolverEnv:
    return GalaxySolverEnv(
        contexts=contexts,
        actions=default_actions(args.hardware),
        mode=args.mode,
        cost_model_path=args.cost_model,
        force_model_path=args.force_model,
        base_config=args.base_config,
        executable=args.executable,
        output_root=args.real_output_root,
        reward_weights=make_reward_weights(args),
        seed=args.seed,
    )


def train(args: argparse.Namespace) -> None:
    contexts = make_contexts(args)
    env = make_env(args, contexts)
    feature_count = len(OBSERVATION_KEYS) + 1
    policy = LinearContextualBanditPolicy.create(
        action_count=len(env.actions),
        feature_count=feature_count,
        ridge_alpha=args.ridge_alpha,
    )
    history: list[dict[str, Any]] = []
    warm_start = args.warm_start or (args.mode == "cheap" and not args.skip_warm_start)

    if warm_start:
        for context_index, context in enumerate(contexts):
            for action_index in range(len(env.actions)):
                observation, info = env.reset(options={"context": context})
                _, reward, terminated, truncated, step_info = env.step(action_index)
                policy.update(observation, action_index, reward)
                action = step_info["action"]
                metrics = step_info["metrics"]
                row: dict[str, Any] = {
                    "episode": -1,
                    "phase": "warm_start",
                    "context_index": context_index,
                    "epsilon": 0.0,
                    "terminated": terminated,
                    "truncated": truncated,
                    "reward": reward,
                    "action_index": action_index,
                    "solver": action["solver"],
                    "tree_theta": action["tree_theta"],
                    "tree_leaf_capacity": action["tree_leaf_capacity"],
                    "fmm_expansion_order": action["fmm_expansion_order"],
                    "n_particles": info["context"].get("n_particles"),
                }
                row.update(metrics)
                history.append(row)

    for episode in range(args.episodes):
        observation, info = env.reset()
        progress = episode / max(args.episodes - 1, 1)
        epsilon = max(args.min_epsilon, args.epsilon * (1.0 - progress))
        action_index = policy.select_action(observation, info, env, rng=env.rng, epsilon=epsilon)
        _, reward, terminated, truncated, step_info = env.step(action_index)
        policy.update(observation, action_index, reward)
        action = step_info["action"]
        metrics = step_info["metrics"]
        row: dict[str, Any] = {
            "episode": episode,
            "phase": "train",
            "context_index": "",
            "epsilon": epsilon,
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "action_index": action_index,
            "solver": action["solver"],
            "tree_theta": action["tree_theta"],
            "tree_leaf_capacity": action["tree_leaf_capacity"],
            "fmm_expansion_order": action["fmm_expansion_order"],
            "n_particles": info["context"].get("n_particles"),
        }
        row.update(metrics)
        history.append(row)

    metadata = {
        "policy_schema_version": "0.1.0",
        "policy_kind": "linear_contextual_bandit",
        "mode": args.mode,
        "episodes": args.episodes,
        "warm_start": warm_start,
        "ridge_alpha": args.ridge_alpha,
        "epsilon": args.epsilon,
        "min_epsilon": args.min_epsilon,
        "hardware": args.hardware,
        "n_particles": args.n_particles,
        "steps": args.steps,
        "dt": args.dt,
        "softening": args.softening,
        "output_format": args.output_format,
        "cost_model": str(args.cost_model) if args.cost_model else "",
        "force_model": str(args.force_model) if args.force_model else "",
        "trained_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    bundle = policy.to_bundle(env.action_table(), metadata)
    save_policy_bundle(args.output, bundle)

    history_path = args.history or args.output.with_suffix(args.output.suffix + ".history.csv")
    write_rows_csv(history_path, history)

    learned_policy = policy
    comparison_rows = evaluate_policies(env, [learned_policy, *all_baseline_policies()], contexts)
    comparison_summary = summarize_policy_rows(comparison_rows)
    comparison_path = args.comparison_csv or args.output.with_suffix(args.output.suffix + ".comparison.csv")
    write_rows_csv(comparison_path, comparison_rows)

    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.md")
    report = policy_markdown_report(
        "Contextual Bandit Solver Policy",
        metadata,
        comparison_summary,
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report, encoding="utf-8")

    write_summary_json(
        args.output.with_suffix(args.output.suffix + ".metadata.json"),
        {
            "metadata": metadata,
            "action_count": len(env.actions),
            "observation_keys": OBSERVATION_KEYS,
            "comparison_summary": comparison_summary,
        },
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {history_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--comparison-csv", type=Path, default=None)
    parser.add_argument("--mode", choices=["cheap", "real"], default="cheap")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Evaluate every action once per context before training.",
    )
    parser.add_argument(
        "--skip-warm-start",
        action="store_true",
        help="Disable default cheap-mode action warm start.",
    )
    parser.add_argument("--n-particles", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--softening", type=float, default=0.02)
    parser.add_argument("--hardware", choices=["cpu", "cuda", "mpi"], default="cpu")
    parser.add_argument("--output-format", default="csv")
    parser.add_argument(
        "--cost-model",
        type=Path,
        default=Path("experiments/ml_models/solver_cost_model.pkl"),
    )
    parser.add_argument(
        "--force-model",
        type=Path,
        default=Path("experiments/ml_models/force_error_model.pkl"),
    )
    parser.add_argument("--base-config", type=Path, default=Path("configs/smoke_test.toml"))
    parser.add_argument("--executable", type=Path, default=None)
    parser.add_argument("--real-output-root", type=Path, default=Path("experiments/rl_env"))
    parser.add_argument("--reward-runtime", type=float, default=1.0)
    parser.add_argument("--reward-force-error", type=float, default=100.0)
    parser.add_argument("--reward-energy-drift", type=float, default=10.0)
    parser.add_argument("--reward-momentum-drift", type=float, default=10.0)
    parser.add_argument("--reward-invalid-config", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
