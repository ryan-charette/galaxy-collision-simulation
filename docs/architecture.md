# Architecture

The repository is organized around a compiled C++ simulation engine and Python
tools for experiments, analysis, visualization, and machine-learning workflows.

```text
                +------------------------------+
                |        Config files          |
                |  TOML simulation settings    |
                +--------------+---------------+
                               |
                               v
+------------------------------+------------------------------+
|                     C++ Simulation Engine                   |
|                                                             |
|  +------------+   +------------+   +----------------------+  |
|  | Initial    |   | Integrator |   | Diagnostics          |  |
|  | Conditions |-->| Leapfrog   |-->| energy/momentum      |  |
|  +------------+   +------------+   +----------------------+  |
|          |                |                    |             |
|          v                v                    v             |
|  +-------------------------------------------------------+  |
|  | Force Solvers                                          |  |
|  | direct, tree, FMM, CUDA variants, MPI wrapper          |  |
|  +-------------------------------------------------------+  |
|          |                |                    |             |
|          v                v                    v             |
|  +------------+   +------------+   +----------------------+  |
|  | MPI        |   | CUDA       |   | Snapshot I/O         |  |
|  | ranks      |   | kernels    |   | CSV/Parquet + JSON   |  |
|  +------------+   +------------+   +----------------------+  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Python Analysis + Rendering                |
|                                                             |
|  load snapshots -> diagnostics -> plots -> density render   |
|                         -> MP4/GIF animation                |
+-------------------------------------------------------------+
```

## C++ Modules

`src/cpp/core`
: Core simulation types and orchestration: vectors, particles, configuration,
  initial conditions, provenance, diagnostics, integration, and the simulation
  runner.

`src/cpp/direct`
: Direct softened-gravity force solver. This is the correctness baseline for
  tests and force-error benchmarks.

`src/cpp/fmm`
: Shared tree geometry, Barnes-Hut treecode, multipole support, and FMM solver
  passes.

`src/cpp/mpi`
: Rank ownership and synchronization helpers. MPI builds distribute owned
  target ranges while keeping reproducible all-rank state synchronization.

`src/cpp/cuda`
: Optional GPU force paths and CPU fallback implementations for CUDA-named
  solvers.

`src/cpp/io`
: CSV/Parquet snapshots, diagnostics output, JSON metadata helpers, and Parquet
  conversion support.

`src/cpp/tests`
: C++ smoke and subsystem tests registered with CTest.

## Python Modules

`src/python/utils`
: Snapshot, diagnostics, config, table, and report helpers.

`src/python/analysis`
: Static plots, force-error summaries, benchmark analysis, and solver crossover
  reports.

`src/python/animation`
: MP4/GIF rendering utilities for simulation snapshots.

`src/python/ml`
: Dataset schemas, supervised model training/evaluation, recommendation tools,
  residual error-correction models, and adaptive solver-tuning environments.

## Scripts

The `scripts/` directory contains command-line entry points for repeatable
developer and experiment workflows:

- `run_benchmarks.py`
- `run_force_error_benchmarks.py`
- `sweep.py`
- `generate_ml_dataset.py`
- `generate_residual_dataset.py`
- README artifact rendering helpers

These scripts use shared Python runtime helpers so simulator discovery, config
generation, metadata loading, and log handling are consistent across workflows.

## Build Modes

Supported build combinations are:

```text
Serial CPU:      direct + tree/FMM
MPI CPU:         distributed direct/tree/FMM wrappers
Serial CUDA:     GPU-accelerated kernels on one device
MPI + CUDA:      MPI orchestration with CUDA-capable solver paths
```

If MPI or CUDA are requested but unavailable, CMake emits a warning and builds
the available CPU fallback paths.

## Data Flow

1. A TOML config defines the experiment.
2. The C++ engine generates initial particle state.
3. The selected solver computes accelerations.
4. The leapfrog integrator advances positions and velocities.
5. The rank responsible for output writes metadata, diagnostics, and snapshots.
6. Python tools load those artifacts for benchmarks, plots, datasets, and
   animations.
