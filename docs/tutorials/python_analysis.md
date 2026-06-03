# Python Analysis

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

The Python package uses a `src/` layout. Installing the project editable makes
commands such as `python -m python.analysis.plot_snapshots` work from the source
checkout. If you prefer not to install the package, set `PYTHONPATH=src` before
running module commands.

## Plot and Render Snapshots

Plot the latest snapshot and diagnostics:

```bash
python -m python.analysis.plot_snapshots --input experiments/validation/smoke_test --output smoke_snapshot.png
```

Render an animation:

```bash
python -m python.animation.render_snapshots --input experiments/validation/smoke_test --mode scatter3d --camera-orbit --output smoke_collision.mp4
```

Render a density projection:

```bash
python -m python.animation.render_snapshots --input experiments/validation/smoke_test --mode density --projection camera --output smoke_density.mp4
```

Render a static density projection:

```bash
python -m python.analysis.plot_snapshots --input experiments/validation/smoke_test --density-output smoke_density.png --no-diagnostics
```

Create a self-contained interactive browser viewer:

```bash
python -m python.animation.interactive_viewer --input experiments/validation/smoke_test --output viewer.html
```

## Regenerate README Artifacts

Regenerate the README collision GIF from the dedicated 1000-body config:

```bash
./build/fmm_galaxy_sim --config configs/readme_1000_body_collision.toml
python scripts/render_readme_snapshot_gif.py --input experiments/validation/readme_1000_body_collision --output docs/assets/galaxy_collision_3d_1000.gif
python -m python.analysis.plot_snapshots --input experiments/validation/readme_1000_body_collision --snapshot experiments/validation/readme_1000_body_collision/snapshot_000149.csv --output docs/assets/readme_snapshot_step149.png --density-output docs/assets/readme_density_step149.png --no-diagnostics
python scripts/run_benchmarks.py --executable build-readme-gif/fmm_galaxy_sim.exe --particles 250 500 1000 --steps 20 --repetitions 3
python scripts/run_benchmarks.py --executable build/fmm_galaxy_sim --solvers cuda-tree cuda-fmm --particles 10000 50000 100000 --steps 10 --repetitions 3 --output-format none --expansion-order 0
```

## Benchmarks and Sweeps

Compare output formats on the same benchmark cases:

```bash
python scripts/run_benchmarks.py --executable build/fmm_galaxy_sim --solvers direct --particles 10000 --steps 10 --output-formats csv parquet
```

Generate the standard direct-reference force-error suite:

```bash
python scripts/run_force_error_benchmarks.py --executable build/fmm_galaxy_sim
```

For CI-scale validation, use the smoke profile:

```bash
python scripts/run_force_error_benchmarks.py --executable build/fmm_galaxy_sim --smoke
```

The suite writes `experiments/accuracy/force_error_summary.csv`,
`force_error_summary.md`, `force_error_vs_n.png`, `force_error_vs_theta.png`,
`energy_drift.png`, and `momentum_drift.png`. It compares step-0 accelerations
against direct summation and reports drift from each solver's diagnostics over a
short integration window.

Launch a generic YAML-defined parameter sweep:

```bash
python scripts/sweep.py --grid configs/sweeps/theta_leaf_order.yaml
```

The sweep runner generates per-run TOML configs, raw logs, simulator output
directories, `sweep_summary.csv`, optional `sweep_summary.parquet`, and
`sweep_metadata.json`. Use `--dry-run` to only materialize planned configs,
`--resume` to skip completed runs with metadata, and `--jobs N` for local
parallel execution.

## ML Dataset and Model Workflows

Generate an ML-ready solver-tuning dataset from a sweep:

```bash
python scripts/generate_ml_dataset.py \
  --sweep configs/sweeps/ml_solver_dataset.yaml \
  --output experiments/ml_datasets/solver_tuning.parquet
```

The dataset generator writes one row per run with solver settings, provenance,
hardware/build metadata, wall-clock timing, particle-steps/s, and
energy/momentum drift fields. Use `--limit N` for a smoke subset and `--resume`
to reuse completed run directories.

Pass `--dataset-type force_error`, `--dataset-type per_step_diagnostics`, or
`--dataset-type all` to materialize the other ML dataset types from the same
sweep outputs. Stable schemas are documented in `docs/ml_datasets.md`.

Train baseline supervised solver models from the ML datasets:

```bash
python -m python.ml.train_solver_cost_model --data experiments/ml_datasets/solver_tuning.parquet --output experiments/ml_models/solver_cost_model.pkl
python -m python.ml.train_force_error_model --data experiments/ml_datasets/force_error.parquet --output experiments/ml_models/force_error_model.pkl
python -m python.ml.recommend_config --n-particles 100000 --target-force-rmse 1e-3 --hardware cpu
```

Model training and recommendation workflows are documented in
`docs/ml_models.md`.

Train and evaluate the first adaptive solver-tuning policy:

```bash
python -m python.ml.rl.train_policy --episodes 200 --cost-model experiments/ml_models/solver_cost_model.pkl --force-model experiments/ml_models/force_error_model.pkl --output experiments/ml_policies/solver_bandit_policy.pkl
python -m python.ml.rl.evaluate_policy --policy experiments/ml_policies/solver_bandit_policy.pkl --cost-model experiments/ml_models/solver_cost_model.pkl --force-model experiments/ml_models/force_error_model.pkl --output experiments/ml_policies/solver_bandit_eval.md
```

The adaptive solver-tuning environment starts as a contextual bandit and
supports both supervised-model `cheap` mode and simulator-launching `real` mode.
Details are in `docs/rl_environment.md`.

Generate, train, and evaluate a learned acceleration-residual correction model:

```bash
python scripts/generate_residual_dataset.py --output experiments/ml_datasets/accel_residuals.csv
python -m python.ml.train_accel_residual_model --data experiments/ml_datasets/accel_residuals.csv --output experiments/ml_models/accel_residual_model.pkl
python -m python.ml.evaluate_accel_residual_model --model experiments/ml_models/accel_residual_model.pkl --data experiments/ml_datasets/accel_residuals.csv --heldout-from-model --stability-steps 5 --output experiments/ml_models/accel_residual_eval.md
```

This workflow predicts direct-minus-approximate acceleration residuals and
reports whether corrected one-step forces improve on held-out solver configs.
Details are in `docs/error_correction.md`.

## Solver Crossover Analysis

Generate solver crossover plots and tables from runtime and accuracy benchmark
CSVs:

```bash
python -m python.analysis.solver_crossover \
  --runtime-csv docs/benchmarks/local_cpu_benchmark.csv \
  --accuracy-csv experiments/accuracy/force_error_summary.csv
```

For fresh runtime inputs, `scripts/run_benchmarks.py --crossover-suite` runs a
wider particle-count sweep with both snapshot output disabled and CSV output
enabled. The crossover analysis writes `runtime_vs_n.png`,
`particle_steps_vs_n.png`, `force_error_vs_runtime.png`,
`best_solver_by_n.csv`, `target_accuracy_summary.csv`, and
`solver_crossover_summary.md`.
