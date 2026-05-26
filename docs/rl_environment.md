# Adaptive Solver Tuning Environment

Phase 3 adds a Gymnasium-compatible one-step environment for adaptive solver
tuning. The environment is a contextual bandit first: the agent observes the
simulation context, chooses one solver configuration, and receives a reward from
runtime, force-error, and drift terms.

The implementation lives in:

```text
python/ml/envs/galaxy_solver_env.py
python/ml/rl/baselines.py
python/ml/rl/train_policy.py
python/ml/rl/evaluate_policy.py
```

## Modes

- `cheap`: evaluates actions through the supervised runtime and force-error
  models when available, with deterministic heuristics as a fallback.
- `real`: writes a temporary TOML config and launches one short simulator run
  for each selected action. This mode records runtime and diagnostics drift; it
  does not compute direct-reference force error by itself.

## Reward

The default reward is:

```text
-(runtime_cost
  + 100 * force_error
  + 10 * energy_drift
  + 10 * momentum_drift
  + 100 * invalid_config)
```

Each coefficient is configurable from the training and evaluation CLIs.

## Fixed Baselines

The evaluator compares learned policies against:

- `always_direct`
- `always_tree_theta_0.6`
- `always_fmm_p4`
- `fastest_supervised`

`fastest_supervised` scores every action in cheap mode and selects the highest
predicted reward.

## Training

```bash
python -m python.ml.rl.train_policy \
  --episodes 200 \
  --cost-model experiments/ml_models/solver_cost_model.pkl \
  --force-model experiments/ml_models/force_error_model.pkl \
  --output experiments/ml_policies/solver_bandit_policy.pkl
```

The training command writes:

- the policy pickle bundle,
- an episode history CSV,
- a learned-vs-baseline comparison CSV,
- a Markdown report,
- metadata with observation keys and action-table size.

## Evaluation

```bash
python -m python.ml.rl.evaluate_policy \
  --policy experiments/ml_policies/solver_bandit_policy.pkl \
  --cost-model experiments/ml_models/solver_cost_model.pkl \
  --force-model experiments/ml_models/force_error_model.pkl \
  --output experiments/ml_policies/solver_bandit_eval.md
```

The report ranks the learned policy and fixed baselines by mean reward and
includes mean runtime cost, mean force error, and selected solver families.

