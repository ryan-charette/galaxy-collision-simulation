# Adaptive Solver Tuning Environment

The adaptive solver-tuning workflow provides a Gymnasium-compatible one-step
environment. It starts as a contextual bandit: the agent observes the simulation
context, chooses one solver configuration, and receives a reward based on
runtime, force error, and drift terms.

The implementation lives in:

```text
src/python/ml/envs/galaxy_solver_env.py
src/python/ml/rl/baselines.py
src/python/ml/rl/train_policy.py
src/python/ml/rl/evaluate_policy.py
```

## Modes

`cheap`
: Evaluates actions through supervised runtime and force-error models when
  available, with deterministic heuristics as a fallback.

`real`
: Writes a temporary TOML config and launches one short simulator run for each
  selected action. This mode records runtime and diagnostics drift. It does not
  compute direct-reference force error by itself.

## Observation

Observations describe the run and recent solver state:

- Particle count.
- Current solver family.
- Tree/FMM parameters.
- Hardware type.
- Recent runtime and drift summaries.
- Density, velocity, and bounding-box summaries when available.

## Action Space

Actions choose a solver configuration from a finite table. The table includes
direct, tree, FMM, and CUDA-named solver variants where applicable, along with
representative values for `tree_theta`, `tree_leaf_capacity`, and
`fmm_expansion_order`.

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

- The policy pickle bundle.
- An episode history CSV.
- A learned-vs-baseline comparison CSV.
- A Markdown report.
- Metadata with observation keys and action-table size.

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
