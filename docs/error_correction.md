# Learned Approximation Error Correction

The learned error-correction workflow trains a model to predict the
per-particle acceleration residual:

```text
acceleration_direct - acceleration_approx
```

The corrected one-step acceleration is:

```text
acceleration_approx + predicted_residual
```

This does not replace the direct, tree, FMM, CUDA, or MPI solvers. It is an
evaluation layer for deciding whether learned residuals are accurate enough to
justify deeper integration work.

## Acceleration Dumps

Set `output.acceleration_dump = true` to write lightweight CSV force dumps:

```toml
[output]
directory = "experiments/error_correction/example"
format = "none"
acceleration_dump = true
```

The simulator writes `accelerations_000000.csv`, `accelerations_000001.csv`,
and so on. These files contain particle IDs, group IDs, positions, velocities,
masses, and accelerations. They can be enabled even when snapshot output is
disabled with `format = "none"`.

## Dataset Generation

Generate a residual dataset:

```bash
python scripts/generate_residual_dataset.py \
  --output experiments/ml_datasets/accel_residuals.csv
```

The generator runs a direct reference and approximate solver cases on identical
initial conditions, pairs the step-0 acceleration dumps by particle ID, and
writes one row per particle.

Smoke profile:

```bash
python scripts/generate_residual_dataset.py \
  --smoke \
  --output experiments/ml_datasets/smoke_accel_residuals.csv
```

## Training

```bash
python -m python.ml.train_accel_residual_model \
  --data experiments/ml_datasets/accel_residuals.csv \
  --output experiments/ml_models/accel_residual_model.pkl
```

The default residual model is a dependency-free KNN regressor. `--model linear`
uses the NumPy ridge baseline, while `random_forest` and `gradient_boosting`
require `scikit-learn`.

Training splits by `config_id`, so evaluation rows come from solver settings not
used for fitting.

## Evaluation

```bash
python -m python.ml.evaluate_accel_residual_model \
  --model experiments/ml_models/accel_residual_model.pkl \
  --data experiments/ml_datasets/accel_residuals.csv \
  --heldout-from-model \
  --stability-steps 5 \
  --output experiments/ml_models/accel_residual_eval.md
```

The report compares approximate one-step acceleration error against corrected
one-step error, records prediction throughput, and only runs the short
constant-acceleration stability sanity check when correction improves held-out
RMSE. Full integration-level correction should remain gated until this one-step
report is consistently positive on unseen solver configs.
