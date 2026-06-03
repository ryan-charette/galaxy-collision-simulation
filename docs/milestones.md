# Milestones

This page summarizes the implemented project phases and the remaining
experiment-oriented work. It is a status record, not a promise of benchmark
results.

## Milestone 0: Project Setup and Design

Status: complete.

Delivered:

- Repository structure.
- CMake build system.
- MPI/CUDA detection with CPU fallback.
- C++ smoke-test executable.
- Python environment and tooling.
- Design documentation.
- Experiment config templates.

## Milestone 1: Direct-Sum Baseline

Status: implemented.

Goal: implement correct `O(N^2)` softened gravitational force calculation.

Delivered:

- Particle state container.
- Softened pairwise force kernel.
- Leapfrog integrator.
- Snapshot output.
- Two-body and disk sanity checks.

## Milestone 2: Single-Node FMM

Status: implemented.

Goal: implement octree-based FMM on one CPU process.

Delivered:

- Octree construction.
- P2M, M2M, M2L-style, L2L, L2P, and P2P passes.
- Monopole, quadrupole, and `p=4` Cartesian moments.
- Direct-vs-FMM accuracy checks.
- Runtime/error plotting support.

## Milestone 3: Galaxy Initial Conditions

Status: implemented.

Goal: generate collision-ready disk galaxies.

Delivered:

- Reproducible 3D exponential disk galaxy generator.
- Configurable mass, radius, position, velocity, orientation, inclination,
  thickness, and group ID.
- Collision parameter configs.
- Stable isolated-disk and galaxy-collision examples.

## Milestone 4: Snapshot I/O and Python Analysis

Status: implemented.

Goal: make outputs easy to inspect, plot, and render.

Delivered:

- CSV snapshot schema.
- Apache Parquet snapshot conversion.
- JSON metadata and diagnostics CSV.
- Python loaders.
- Diagnostic plots.
- Scatter and density animation helpers.

## Milestone 5: MPI Distributed CPU Solvers

Status: implemented for particle-count decomposition.

Goal: run solver workflows across multiple MPI ranks.

Delivered:

- Distributed particle ownership by contiguous range.
- Full particle-state synchronization with `MPI_Allgatherv`.
- Rank-0 snapshot and diagnostics output.
- MPI-aware provenance metadata.

## Milestone 6: CUDA Acceleration

Status: implemented with CPU fallback.

Goal: accelerate measured bottlenecks where CUDA is available.

Delivered:

- CUDA direct/P2P acceleration support.
- GPU evaluation paths for CPU-built tree/FMM interaction data.
- Public solver names `cuda-direct`, `cuda-tree`, and `cuda-fmm`.
- Device buffer reuse across steps.
- Pinned host staging.
- Expansion-order specializations for `0`, `2`, and `4`.
- CPU fallback behavior when CUDA is unavailable.

## Milestone 7: Full Galaxy Collision Experiments

Status: supported by configuration and benchmark tooling.

Experiment directions:

- Timestep stability sweeps.
- Impact-parameter sweeps.
- Relative-velocity sweeps.
- Mass-ratio sweeps.
- Orientation sweeps.

## Milestone 8: Animation Pipeline

Status: implemented for README-scale and analysis-scale outputs.

Delivered:

- Scatter and density renderers.
- GIF and MP4 export helpers.
- README artifact rendering scripts.

## Milestone 9: Benchmarking and Evaluation

Status: implemented as repeatable scripts and documented artifacts.

Delivered:

- Runtime benchmarks.
- Force-error benchmarks.
- Solver crossover analysis.
- CSV/Parquet output comparison support.
- Metadata-stamped benchmark artifacts.
