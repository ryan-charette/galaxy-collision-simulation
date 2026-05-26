"""Evaluate learned solver-tuning policies against fixed baselines."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from python.ml.envs.galaxy_solver_env import (
    GalaxySolverEnv,
    OBSERVATION_KEYS,
    RewardWeights,
    default_actions,
    default_contexts,
)
from python.ml.rl.baselines import (
    all_baseline_policies,
    evaluate_policies,
    LinearContextualBanditPolicy,
    load_policy_bundle,
    policy_markdown_report,
    summarize_policy_rows,
    validate_policy_actions,
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
        reward_weights=RewardWeights(
            runtime=args.reward_runtime,
            force_error=args.reward_force_error,
            energy_drift=args.reward_energy_drift,
            momentum_drift=args.reward_momentum_drift,
            invalid_config=args.reward_invalid_config,
        ),
        seed=args.seed,
    )


def evaluate(args: argparse.Namespace) -> None:
    contexts = make_contexts(args)
    env = make_env(args, contexts)
    policy_bundle = load_policy_bundle(args.policy)
    validate_policy_actions(policy_bundle, env.action_table())
    learned_policy = LinearContextualBanditPolicy.from_bundle(policy_bundle)
    rows = evaluate_policies(env, [learned_policy, *all_baseline_policies()], contexts)
    summaries = summarize_policy_rows(rows)

    csv_path = args.csv_output or args.output.with_suffix(args.output.suffix + ".csv")
    write_rows_csv(csv_path, rows)

    metadata = {
        "policy": str(args.policy),
        "mode": args.mode,
        "hardware": args.hardware,
        "n_particles": args.n_particles,
        "steps": args.steps,
        "dt": args.dt,
        "softening": args.softening,
        "output_format": args.output_format,
        "cost_model": str(args.cost_model) if args.cost_model else "",
        "force_model": str(args.force_model) if args.force_model else "",
        "evaluated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    report = policy_markdown_report("Solver Policy Evaluation", metadata, summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    write_summary_json(
        args.output.with_suffix(args.output.suffix + ".summary.json"),
        {
            "metadata": metadata,
            "observation_keys": OBSERVATION_KEYS,
            "action_count": len(env.actions),
            "summary": summaries,
        },
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--mode", choices=["cheap", "real"], default="cheap")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--real-output-root", type=Path, default=Path("experiments/rl_env_eval"))
    parser.add_argument("--reward-runtime", type=float, default=1.0)
    parser.add_argument("--reward-force-error", type=float, default=100.0)
    parser.add_argument("--reward-energy-drift", type=float, default=10.0)
    parser.add_argument("--reward-momentum-drift", type=float, default=10.0)
    parser.add_argument("--reward-invalid-config", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
