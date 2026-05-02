# Milestones

## Milestone 0: Project setup and design

Status: complete in this scaffold.

Deliverables:

- repository structure,
- CMake build system,
- MPI/CUDA detection,
- smoke-test executable,
- Python environment,
- design documentation,
- experiment config template.

## Milestone 1: Direct-sum baseline

Goal: implement correct `O(N^2)` gravitational force calculation.

Status: implemented in the MVP.

Deliverables:

- particle state container,
- softened pairwise force kernel,
- leapfrog integrator,
- snapshot output,
- two-body and disk sanity tests.

## Milestone 2: Single-node FMM

Goal: implement octree-based FMM on one CPU process.

Status: implemented as a 3D octree FMM with monopole, quadrupole, and `p=4` Cartesian moments. The code now has explicit P2M/M2M aggregation, M2L-style far-cell interaction lists, P2P near-field leaves, target-range evaluation, and direct-vs-FMM smoke tests.

Deliverables:

- octree construction,
- P2M/M2M/M2L/L2L/L2P/P2P passes,
- direct-vs-FMM accuracy tests,
- runtime/error plots.

## Milestone 3: Galaxy initial conditions

Goal: generate collision-ready disk galaxies.

Status: implemented for reproducible 3D exponential disk galaxies with configurable mass, radius, position, velocity, orientation, inclination, thickness, and group ID.

Deliverables:

- disk galaxy generator,
- collision parameter configs,
- stable isolated disk demo,
- head-on and off-center collision demos.

## Milestone 4: Snapshot I/O and Python analysis

Goal: make outputs easy to inspect, plot, and render.

Status: implemented with CSV snapshots, JSON metadata, diagnostics CSV, Python loaders, static plotting, and MP4/GIF animation scripts. HDF5 remains a future large-run format.

Deliverables:

- snapshot schema implementation,
- Python loader,
- diagnostic plots,
- basic scatter animation.

## Milestone 5: MPI distributed CPU FMM

Goal: run the solver across multiple MPI ranks.

Status: implemented for particle-count decomposition. Each rank owns a contiguous particle range, evaluates local direct/FMM accelerations, synchronizes full particle state with `MPI_Allgatherv`, and rank 0 writes snapshots/diagnostics.

Deliverables:

- distributed particle ownership,
- global bounding box reduction,
- tree summary exchange,
- scaling benchmarks.

## Milestone 6: CUDA acceleration

Goal: accelerate measured bottlenecks.

Status: implemented for the direct/P2P path with CUDA acceleration and CPU fallback. The current GPU path supports direct accelerations and leapfrog stepping for `solver = "cuda-direct"`.

Deliverables:

- GPU P2P near-field kernel,
- GPU integration kernel,
- CPU/GPU result comparison,
- kernel timing report.

## Milestone 7: Full galaxy collision experiments

Goal: run the scientifically interesting and visually compelling cases.

Deliverables:

- timestep stability experiment,
- impact parameter sweep,
- relative velocity sweep,
- mass ratio sweep,
- orientation sweep.

## Milestone 8: Animation pipeline

Goal: produce portfolio-quality videos.

Deliverables:

- density renderer,
- cinematic camera path,
- MP4 export,
- GIF export for README.

## Milestone 9: Benchmarking and evaluation

Goal: support all performance claims with measurements.

Deliverables:

- runtime vs N,
- speedup vs MPI ranks,
- CPU vs CUDA speedup,
- force error vs approximation setting,
- energy drift plots.

## Milestone 10: Portfolio packaging

Goal: make the project clear and impressive to reviewers.

Deliverables:

- final README,
- demo video,
- benchmark report,
- architecture diagram,
- resume bullets.
