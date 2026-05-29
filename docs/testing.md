# Testing and Validation Plan

## Unit tests

Initial C++ unit tests should cover:

- vector arithmetic,
- particle initialization through generated galaxies,
- pairwise force symmetry,
- finite acceleration under softening,
- leapfrog update consistency,
- octree Barnes-Hut force agreement against direct summation,
- `p=4` FMM force agreement against direct summation,
- CUDA direct solver agreement with the CPU direct solver or CPU fallback,
- MPI ownership range decomposition,
- diagnostics sanity,
- config parsing,
- snapshot writing.

These are covered by `cpp/tests/smoke_tests.cpp` and registered with CTest as `smoke_tests`.
The smoke executable is split into focused source files so failures point to the
relevant subsystem:

- `cpp/tests/math_direct_tests.cpp`
- `cpp/tests/tree_fmm_accuracy_tests.cpp`
- `cpp/tests/cuda_fallback_tests.cpp`
- `cpp/tests/config_snapshot_tests.cpp`
- `cpp/tests/smoke_tests.cpp` as the shared entrypoint

## Physics sanity tests

### Two-body orbit

Expected behavior:

- approximately closed orbit for small timestep,
- bounded energy drift,
- total momentum conservation.

### Isolated disk

Expected behavior:

- disk remains coherent for a reasonable number of dynamical times,
- inner particles are more timestep-sensitive,
- no immediate numerical explosion under default softening.

### Head-on collision

Expected behavior:

- symmetric morphology for equal-mass identical disks,
- conservation diagnostics remain interpretable,
- close encounter remains stable due to softening.

## Solver validation

Compare direct sum and approximate solver accelerations.

Metrics:

```text
relative_error_i = ||a_approx_i - a_direct_i|| / max(||a_direct_i||, tiny)
mean_relative_error
median_relative_error
p95_relative_error
max_relative_error
```

The standard force-error suite automates this comparison:

```bash
python scripts/run_force_error_benchmarks.py --smoke
```

The smoke profile is intended for CI-scale validation. The full sweep writes
CSV, Markdown, and plots under `experiments/accuracy/`, including force RMSE,
max force error, relative force error, energy drift, momentum drift, angular
momentum drift, runtime per step, particle-steps per second, and optional peak
memory when `psutil` is installed.

Generic config sweeps can be validated without launching simulations:

```bash
python scripts/sweep.py --grid configs/sweeps/theta_leaf_order.yaml --dry-run --limit 2
```

For a small execution smoke test, pass a small grid and `--limit`, then inspect
`sweep_summary.csv` for completed and failed runs.

ML dataset generation can be smoke-tested from the standard solver-tuning sweep:

```bash
python scripts/generate_ml_dataset.py --sweep configs/sweeps/ml_solver_dataset.yaml --output experiments/ml_datasets/smoke_solver_tuning.csv --limit 2
```

Use a `.parquet` output path for the production artifact when `pandas` and
`pyarrow` are installed.

To validate all ML dataset types, run a slightly larger smoke subset so
the force-error table has both direct and approximate solver rows:

```bash
python scripts/generate_ml_dataset.py --sweep configs/sweeps/ml_solver_dataset.yaml --output experiments/ml_datasets/smoke_all --dataset-type all --limit 6
```

Supervised ML training can be smoke-tested with the generated CSV artifacts:

```bash
python -m python.ml.train_solver_cost_model --data experiments/ml_datasets/smoke_all/solver_tuning.csv --output experiments/ml_models/smoke_solver_cost_model.pkl
python -m python.ml.train_force_error_model --data experiments/ml_datasets/smoke_all/force_error.csv --output experiments/ml_models/smoke_force_error_model.pkl
python -m python.ml.evaluate_models --model experiments/ml_models/smoke_solver_cost_model.pkl --data experiments/ml_datasets/smoke_all/solver_tuning.csv --output experiments/ml_models/smoke_solver_cost_model.eval.md
python -m python.ml.recommend_config --n-particles 100000 --target-force-rmse 1e-3 --hardware cpu --cost-model experiments/ml_models/smoke_solver_cost_model.pkl --force-model experiments/ml_models/smoke_force_error_model.pkl
```

The adaptive solver-tuning environment and contextual-bandit policy can be
smoke-tested in cheap mode without launching new simulations:

```bash
python -m python.ml.rl.train_policy --episodes 30 --n-particles 256 512 --cost-model experiments/ml_models/smoke_solver_cost_model.pkl --force-model experiments/ml_models/smoke_force_error_model.pkl --output experiments/ml_policies/smoke_bandit_policy.pkl
python -m python.ml.rl.evaluate_policy --policy experiments/ml_policies/smoke_bandit_policy.pkl --n-particles 256 512 --cost-model experiments/ml_models/smoke_solver_cost_model.pkl --force-model experiments/ml_models/smoke_force_error_model.pkl --output experiments/ml_policies/smoke_bandit_eval.md
```

Learned acceleration-residual correction can be smoke-tested with direct and
approximate step-0 acceleration dumps:

```bash
python scripts/generate_residual_dataset.py --smoke --output experiments/ml_datasets/smoke_accel_residuals.csv
python -m python.ml.train_accel_residual_model --data experiments/ml_datasets/smoke_accel_residuals.csv --output experiments/ml_models/smoke_accel_residual_model.pkl
python -m python.ml.evaluate_accel_residual_model --model experiments/ml_models/smoke_accel_residual_model.pkl --data experiments/ml_datasets/smoke_accel_residuals.csv --heldout-from-model --stability-steps 3 --output experiments/ml_models/smoke_accel_residual_eval.md
```

Solver crossover summaries can be regenerated from benchmark artifacts:

```bash
python -m python.analysis.solver_crossover --runtime-csv docs/benchmarks/local_cpu_benchmark.csv
```

When a force-error summary is available, pass it with `--accuracy-csv` to include
force-error-vs-runtime and target-accuracy tables.

## Parallel validation

Compare serial and MPI outputs using identical seeds/configs.

Expected:

- small floating-point differences are acceptable,
- statistical diagnostics should match,
- aggregate mass, momentum, and total particle count should match exactly or within strict tolerance.

## CUDA validation

Compare CPU and GPU kernels for:

- accelerations,
- integrated positions,
- integrated velocities,
- diagnostics.

Use tolerance-based tests rather than exact equality.
