# Distributed Fast Multipole Galaxy Collision Simulator

A compact 3D gravitational N-body simulator for galaxy collision experiments. The project includes a working C++ simulation engine, CSV snapshot output, diagnostics, and Python plotting/animation tools.

## Implemented MVP

- Softened Newtonian gravity in nondimensional units
- Direct `O(N^2)` force solver for correctness baselines
- Barnes-Hut octree solver with optional `p=4` far-field correction
- FMM solver with P2M/M2M aggregation, `p=4` Cartesian moments, M2L-style cell interaction lists, and P2P near-field leaves
- MPI rank ownership with all-rank particle synchronization
- Optional CUDA direct/P2P force and leapfrog kernels with CPU fallback
- Kick-drift-kick leapfrog integrator
- Reproducible disk-galaxy initial conditions from TOML-like configs
- CSV snapshots plus metadata and energy/momentum diagnostics
- Python snapshot loader, static plotting, and MP4/GIF animation scripts
- CTest smoke tests covering vectors, forces, integration, FMM accuracy, CUDA fallback, MPI ownership, config parsing, diagnostics, and snapshot writing

## Repository Layout

```text
cpp/core/                core particles, config, integrator, diagnostics, CLI
cpp/direct/              direct softened-gravity solver
cpp/fmm/                 Barnes-Hut treecode and p=4 FMM solver
cpp/mpi/                 rank ownership and particle synchronization helpers
cpp/cuda/                optional CUDA direct/P2P kernels and CPU fallback
cpp/io/                  CSV snapshot and diagnostics writer
cpp/tests/               C++ smoke/unit tests
python/utils/            snapshot and diagnostics loaders
python/analysis/static   plots
python/animation/MP4/GIF rendering
configs/                 simulation configs
experiments/             output destinations and experiment notes
docs/                    design, architecture, roadmap, testing plan
scripts/                 build and smoke-test helpers
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If MPI or CUDA are not installed, CMake falls back to the serial CPU build.

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

Choose a solver in the config:

```toml
[simulation]
solver = "fmm"          # direct, tree, fmm, cuda-direct
dim = 3
tree_theta = 0.6
tree_leaf_capacity = 16
fmm_expansion_order = 4 # 0 = monopole, 2 = quadrupole, 4 = fourth-order Cartesian
```

Run with MPI when available:

```bash
mpirun -np 4 ./build/fmm_galaxy_sim --config configs/smoke_test.toml
```

Run the CUDA direct/P2P kernel when a CUDA device is available:

```toml
[simulation]
solver = "cuda-direct"
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
python -m python.animation.render_snapshots --input experiments/validation/smoke_test --mode scatter3d --camera-orbit --output smoke_collision.mp4
```

Render a density projection:

```bash
python -m python.animation.render_snapshots --input experiments/validation/smoke_test --mode density --projection camera --output smoke_density.mp4
```

Create a self-contained interactive browser viewer:

```bash
python -m python.animation.interactive_viewer --input experiments/validation/smoke_test --output viewer.html
```

## Current Scope

This is now a distributed/GPU-capable 3D MVP. The FMM supports monopole (`p=0`), quadrupole (`p=2`), and fourth-order Cartesian (`p=4`) moments; orders above `p=4` are intentionally out of scope for this project.
