# ML Dataset Schemas

All generated ML datasets include `dataset_schema_version = "0.1.0"` and are
written by `scripts/generate_ml_dataset.py`.

The generator writes two tables for each dataset type:

- `*.raw.*`: every generated row, including failed or incomplete runs.
- `*.*`: cleaned rows with completed status and no missing required values.

Each dataset also receives `*.manifest.json` and `*.summary.md` sidecar files.

## Solver Tuning

One row per simulator run or segment.

Required columns:

```text
dataset_schema_version
run_id
git_commit
config_sha256
hardware_type
solver
n_particles
steps
dt
softening
tree_theta
tree_leaf_capacity
fmm_expansion_order
output_format
median_step_time
total_wall_time
particle_steps_per_second
energy_drift_final
momentum_drift_final
max_energy_drift
max_momentum_drift
```

## Force Error

One row per approximate solver/config comparison against a matching direct run.

Required columns:

```text
dataset_schema_version
run_id
direct_run_id
git_commit
config_sha256
direct_config_sha256
solver
n_particles
tree_theta
tree_leaf_capacity
fmm_expansion_order
softening
force_rmse
force_mae
force_max_error
relative_force_rmse
runtime_direct
runtime_approx
speedup_vs_direct
```

## Per-Step Diagnostics

One row per diagnostics entry. To capture every simulation step, use a sweep
where `simulation.snapshot_every = 1`.

Required columns:

```text
dataset_schema_version
run_id
git_commit
config_sha256
solver
n_particles
step
time
kinetic_energy
potential_energy
total_energy
linear_momentum_x
linear_momentum_y
linear_momentum_z
angular_momentum_x
angular_momentum_y
angular_momentum_z
step_wall_time
```

`step_wall_time` is currently the run-level average wall time per configured
step, because the simulator does not yet emit per-step timing samples.
