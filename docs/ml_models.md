# Supervised Solver Models

Supervised solver modeling trains baseline regressors from generated ML
datasets. The models predict runtime, throughput, drift, and force-error metrics
from simulation settings and hardware descriptors.

Model artifacts are Python pickle bundles with stable keys:

- `dataset_schema_version`
- Model kind and model type.
- Target columns.
- Fitted feature transformer.
- Fitted model.
- Mean-baseline metrics.
- Training metadata.

The NumPy linear baseline works with only the core Python dependencies. Random
forest and gradient-boosted tree models require `scikit-learn`.

## Runtime and Drift Model

Train the solver cost model:

```bash
python -m python.ml.train_solver_cost_model \
  --data experiments/ml_datasets/solver_tuning.parquet \
  --output experiments/ml_models/solver_cost_model.pkl
```

Default targets:

```text
median_step_time
particle_steps_per_second
energy_drift_final
momentum_drift_final
```

Use `--model random_forest` or `--model gradient_boosting` after installing the
project dependencies.

## Force Error Model

Train the force-error model:

```bash
python -m python.ml.train_force_error_model \
  --data experiments/ml_datasets/force_error.parquet \
  --output experiments/ml_models/force_error_model.pkl
```

Default targets:

```text
force_rmse
relative_force_rmse
```

## Evaluation

Evaluate a trained model against a dataset:

```bash
python -m python.ml.evaluate_models \
  --model experiments/ml_models/solver_cost_model.pkl \
  --data experiments/ml_datasets/solver_tuning.parquet \
  --output experiments/ml_models/solver_cost_model.eval.md
```

Reports include MAE, RMSE, R2, mean-baseline RMSE, whether the model beats the
mean baseline per target, and solver-selection accuracy when solver groups are
available.

## Recommendation

Generate a simulator config snippet from trained models:

```bash
python -m python.ml.recommend_config \
  --n-particles 100000 \
  --target-force-rmse 1e-3 \
  --hardware cpu
```

The command prints a valid TOML `[simulation]` snippet. Pass `--report` to save
the model recommendation and fixed hand-tuned baseline predictions as JSON.
