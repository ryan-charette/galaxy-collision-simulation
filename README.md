# Distributed Fast Multipole Galaxy Collision Simulator

A compact 2D gravitational N-body simulator for galaxy collision experiments. The project currently includes a working C++ simulation engine, CSV snapshot output, diagnostics, and Python plotting/animation tools.

## Implemented MVP

- Softened Newtonian gravity in nondimensional units
- Direct `O(N^2)` force solver for correctness baselines
- Barnes-Hut quadtree solver on the FMM implementation path
- Kick-drift-kick leapfrog integrator
- Reproducible disk-galaxy initial conditions from TOML-like configs
- CSV snapshots plus metadata and energy/momentum diagnostics
- Python snapshot loader, static plotting, and MP4/GIF animation scripts
- CTest smoke tests covering vectors, forces, integration, config parsing, tree accuracy, diagnostics, and snapshot writing
- Optional MPI/CUDA detection remains in the build for future distributed/GPU milestones

## Repository Layout

```text
cpp/core/       core particles, config, integrator, diagnostics, CLI
cpp/direct/     direct softened-gravity solver
cpp/fmm/        Barnes-Hut quadtree solver and future FMM home
cpp/io/         CSV snapshot and diagnostics writer
cpp/tests/      C++ smoke/unit tests
python/utils/   snapshot and diagnostics loaders
python/analysis/static plots
python/animation/MP4/GIF rendering
configs/        simulation configs
experiments/    output destinations and experiment notes
docs/           design, architecture, roadmap, testing plan
scripts/        build and smoke-test helpers
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=OFF -DENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On Windows PowerShell, if CMake is installed:

```powershell
.\scripts\build.ps1
.\scripts\run_smoke_test.ps1
```

## Run a Simulation

```bash
./build/fmm_galaxy_sim --config configs/smoke_test.toml
```

The default smoke config writes:

```text
experiments/validation/smoke_test/
  metadata.json
  diagnostics.csv
  snapshot_000000.csv
  snapshot_000010.csv
  ...
```

Use the tree solver by setting:

```toml
[simulation]
solver = "tree"
tree_theta = 0.6
tree_leaf_capacity = 16
```

## Python Analysis

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Plot the latest snapshot and diagnostics:

```bash
python -m python.analysis.plot_snapshots --input experiments/validation/smoke_test --output smoke_snapshot.png
```

Render an animation:

```bash
python -m python.animation.render_snapshots --input experiments/validation/smoke_test --output smoke_collision.mp4
```

## Current Scope

This is a complete single-node MVP. MPI distribution, CUDA kernels, and true high-order FMM passes are still roadmap items, but the project now has the working physics, I/O, validation, and visualization backbone needed to build those pieces honestly.
