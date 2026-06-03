# Testing and Validation

The validation strategy combines C++ smoke/unit tests, Python tests, benchmark
smoke runs, and optional CUDA hardware checks. Direct summation is the numerical
reference for approximate solver validation whenever the particle count is small
enough for `O(N^2)` work.

## C++ Tests

The C++ test executable is registered with CTest as `smoke_tests`.

```bash
cmake --build build --config Release --target fmm_galaxy_tests
ctest --test-dir build -C Release --output-on-failure
```

The smoke executable is split into focused source files:

- `src/cpp/tests/math_direct_tests.cpp`
- `src/cpp/tests/tree_fmm_accuracy_tests.cpp`
- `src/cpp/tests/cuda_fallback_tests.cpp`
- `src/cpp/tests/config_snapshot_tests.cpp`
- `src/cpp/tests/smoke_tests.cpp`

Coverage includes vector arithmetic, generated galaxies, pairwise force
symmetry, finite softened accelerations, leapfrog consistency, direct/tree/FMM
agreement, CUDA fallback parity, MPI ownership ranges, diagnostics, config
parsing, provenance, and snapshot writing.

## Python Tests

Python tests live under `src/python/tests`.

```bash
pytest
```

The project config sets `src` on `pythonpath`, enables strict pytest config and
marker handling, and reports useful skip/failure summaries.

## Coverage

CI generates separate Python and C++ coverage reports and uploads both to
Codecov:

- `coverage/python.xml` from `pytest-cov`.
- `coverage/cpp.xml` from `gcovr` after a GCC coverage build.

Generate Python coverage locally with:

```bash
nox -s coverage
```

Generate C++ coverage locally on a GCC-compatible toolchain with:

```bash
cmake -S . -B build-coverage \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_MPI=OFF \
  -DENABLE_CUDA=OFF \
  -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage"

cmake --build build-coverage --target fmm_galaxy_tests --parallel
ctest --test-dir build-coverage --output-on-failure
gcovr --root . --filter src/cpp --exclude src/cpp/tests \
  --xml-pretty --output coverage/cpp.xml build-coverage
```

## Physics Sanity Cases

### Two-Body Orbit

Expected behavior:

- Approximately closed orbit for small timesteps.
- Bounded energy drift.
- Total momentum conservation.

### Isolated Disk

Expected behavior:

- The disk remains coherent for a reasonable number of dynamical times.
- Inner particles are more timestep-sensitive.
- Default softening avoids immediate numerical instability.

### Head-On Collision

Expected behavior:

- Equal-mass identical disks produce symmetric morphology.
- Conservation diagnostics remain interpretable.
- Close encounters remain stable under the configured softening.

## Solver Validation

Approximate solver accelerations are compared against direct summation:

```text
relative_error_i = ||a_approx_i - a_direct_i|| / max(||a_direct_i||, tiny)
mean_relative_error
median_relative_error
p95_relative_error
max_relative_error
```

Run the CI-scale force-error benchmark:

```bash
python scripts/run_force_error_benchmarks.py --smoke
```

The full suite writes CSV, Markdown, and plots under `experiments/accuracy/`.
Metrics include force RMSE, maximum force error, relative force error, energy
drift, momentum drift, angular momentum drift, runtime per step,
particle-steps per second, and optional peak memory when `psutil` is installed.

## Sweep Smoke Tests

Generic config sweeps can be validated without launching simulations:

```bash
python scripts/sweep.py --grid configs/sweeps/theta_leaf_order.yaml --dry-run --limit 2
```

For a small execution smoke test, pass a small grid and `--limit`, then inspect
`sweep_summary.csv` for completed and failed runs.

## ML Workflow Smoke Tests

Generate a small solver-tuning dataset:

```bash
python scripts/generate_ml_dataset.py \
  --sweep configs/sweeps/ml_solver_dataset.yaml \
  --output experiments/ml_datasets/smoke_solver_tuning.csv \
  --limit 2
```

Generate all dataset types with a slightly larger subset:

```bash
python scripts/generate_ml_dataset.py \
  --sweep configs/sweeps/ml_solver_dataset.yaml \
  --output experiments/ml_datasets/smoke_all \
  --dataset-type all \
  --limit 6
```

Train and evaluate supervised models:

```bash
python -m python.ml.train_solver_cost_model \
  --data experiments/ml_datasets/smoke_all/solver_tuning.csv \
  --output experiments/ml_models/smoke_solver_cost_model.pkl

python -m python.ml.train_force_error_model \
  --data experiments/ml_datasets/smoke_all/force_error.csv \
  --output experiments/ml_models/smoke_force_error_model.pkl

python -m python.ml.evaluate_models \
  --model experiments/ml_models/smoke_solver_cost_model.pkl \
  --data experiments/ml_datasets/smoke_all/solver_tuning.csv \
  --output experiments/ml_models/smoke_solver_cost_model.eval.md
```

Train and evaluate the cheap-mode contextual-bandit policy:

```bash
python -m python.ml.rl.train_policy \
  --episodes 30 \
  --n-particles 256 512 \
  --cost-model experiments/ml_models/smoke_solver_cost_model.pkl \
  --force-model experiments/ml_models/smoke_force_error_model.pkl \
  --output experiments/ml_policies/smoke_bandit_policy.pkl

python -m python.ml.rl.evaluate_policy \
  --policy experiments/ml_policies/smoke_bandit_policy.pkl \
  --n-particles 256 512 \
  --cost-model experiments/ml_models/smoke_solver_cost_model.pkl \
  --force-model experiments/ml_models/smoke_force_error_model.pkl \
  --output experiments/ml_policies/smoke_bandit_eval.md
```

Generate and evaluate residual-correction data:

```bash
python scripts/generate_residual_dataset.py \
  --smoke \
  --output experiments/ml_datasets/smoke_accel_residuals.csv

python -m python.ml.train_accel_residual_model \
  --data experiments/ml_datasets/smoke_accel_residuals.csv \
  --output experiments/ml_models/smoke_accel_residual_model.pkl

python -m python.ml.evaluate_accel_residual_model \
  --model experiments/ml_models/smoke_accel_residual_model.pkl \
  --data experiments/ml_datasets/smoke_accel_residuals.csv \
  --heldout-from-model \
  --stability-steps 3 \
  --output experiments/ml_models/smoke_accel_residual_eval.md
```

## Solver Crossover Reports

Solver crossover summaries can be regenerated from benchmark artifacts:

```bash
python -m python.analysis.solver_crossover \
  --runtime-csv docs/benchmarks/local_cpu_benchmark.csv
```

When a force-error summary is available, pass it with `--accuracy-csv` to
include force-error-vs-runtime and target-accuracy tables.

## Parallel Validation

Compare serial and MPI outputs using identical seeds/configs.

Expected behavior:

- Small floating-point differences are acceptable.
- Statistical diagnostics should match.
- Aggregate mass, momentum, and total particle count should match exactly or
  within strict tolerance.

## CUDA Validation

Compare CPU and GPU kernels for:

- Accelerations.
- Integrated positions.
- Integrated velocities.
- Diagnostics.

CUDA-enabled builds should be validated on a machine with a CUDA compiler and
device. CPU-only builds validate fallback behavior but do not compile or execute
the `.cu` kernels.
