# Technical Design

This project is a 3D softened-gravity N-body simulator for galaxy collision
experiments. The C++ engine provides direct, Barnes-Hut tree, FMM, MPI, and
optional CUDA execution paths. Python tooling covers reproducible experiment
runs, benchmark sweeps, snapshot loading, plotting, animation, and benchmark
analysis.

The design favors reproducibility and comparison across solvers. Direct
summation remains the correctness baseline. Approximate and accelerated solvers
are evaluated against force error, conservation diagnostics, wall time, and
hardware provenance.

## Particle Model

Each particle stores:

- 3D position, velocity, and acceleration.
- Mass.
- Integer group ID for source galaxy membership.

The simulator uses nondimensional code units by default. Planar experiments are
represented by keeping the `z` components at zero.

## Force Law

All solvers compute the same softened Newtonian acceleration:

```text
r_ij = x_j - x_i
s2   = dot(r_ij, r_ij) + eps^2
a_i += G * m_j * r_ij / s2^(3/2)
```

The gravitational constant `G` and Plummer-style softening length `eps` are
configured per run.

## Integration

The simulator uses kick-drift-kick leapfrog integration:

```text
v_{n+1/2} = v_n       + 0.5 * dt * a(x_n)
x_{n+1}   = x_n       + dt * v_{n+1/2}
a_{n+1}   = a(x_{n+1})
v_{n+1}   = v_{n+1/2} + 0.5 * dt * a_{n+1}
```

Direct, tree, FMM, and CUDA solver variants all plug into this integrator by
providing updated particle accelerations.

## Solvers

### Direct Summation

The direct solver computes every pairwise interaction in `O(N^2)` time. It is
used for small runs, regression tests, force-error benchmarks, and training data
where exact accelerations are needed.

### Barnes-Hut Tree

The tree solver builds a 3D octree and evaluates each target particle by walking
the tree. Far cells are accepted when the cell-size-to-distance ratio satisfies
the configured `tree_theta` criterion; nearby leaves fall back to direct P2P
interactions.

The octree root sizing and child-cell geometry are shared with the FMM tree
builder so both solvers partition space consistently. Solver-specific traversal
and approximation logic remain separate.

### Fast Multipole Method

The FMM path uses the same octree structure, but accumulates far-field work per
cell and evaluates local expansions for particles. The implemented pipeline is:

1. P2M leaf multipole construction.
2. M2M upward aggregation.
3. M2L-style far-cell contributions into target local expansions.
4. L2L downward propagation of local expansions.
5. L2P local expansion evaluation for each target particle.
6. P2P direct interactions for near leaves.

Expansion orders `0`, `2`, and `4` are supported. Higher orders are not
implemented.

### MPI

MPI uses contiguous particle ownership ranges:

```text
rank k owns particles [start_k, end_k)
```

Each rank computes accelerations for its owned range, synchronizes full particle
state with `MPI_Allgatherv`, and rank 0 writes snapshots, diagnostics, and
metadata. This decomposition prioritizes validation and reproducible output over
load-balanced spatial partitioning.

### CUDA

CUDA support is optional. When available, the simulator provides GPU execution
paths for direct, tree, and FMM solver modes. CPU fallback paths preserve the
same public solver names when CUDA is not available.

The CUDA implementation uses structure-of-arrays force inputs, pinned host
staging, device buffer reuse across steps, shared-memory tiling for direct P2P,
and specialized tree/FMM kernels for expansion orders `0`, `2`, and `4`.

## Output and Provenance

Each run writes `metadata.json` in the configured output directory. Metadata is
written even when snapshots are disabled with `[output] format = "none"`.

Metadata records:

- Git commit, branch, and dirty working-tree state.
- Build type, compiler, and requested CMake MPI/CUDA options.
- CUDA availability and device name.
- MPI rank count.
- Hostname and UTC timestamp.
- Config path and config SHA-256 hash.

Snapshot output supports CSV, Apache Parquet, and disabled output:

```toml
[output]
directory = "experiments/example"
format = "csv"      # csv, parquet, none
snapshot_every = 10
```

Parquet conversion is handled by `python.utils.parquet_io`. Set
`FMM_GALAXY_PYTHON` when the simulator should use a specific Python
interpreter.

Diagnostics are written as CSV when output is enabled. Acceleration dumps can be
enabled for direct-vs-approximate force diagnostics.

## Experiment Tooling

The Python runtime utilities provide shared helpers for:

- Simulator executable discovery.
- Temporary config generation.
- Dotted TOML updates.
- Galaxy particle-count synchronization.
- Subprocess execution and log capture.
- Metadata and diagnostics loading.
- Resume behavior for sweep runs.

These helpers are used by sweeps, benchmarks, force-error comparisons, README
artifact generation, and plotting workflows.

## Benchmarking

Benchmarking emphasizes both speed and accuracy:

- Runtime per step.
- Particle-steps per second.
- Force RMSE and relative force error against direct summation.
- Energy, momentum, and angular momentum drift.
- Optional memory usage.

## Validation

Validation is split across focused C++ tests and Python smoke commands:

- Math/direct force and integrator checks.
- Tree/FMM accuracy against direct summation.
- CUDA fallback parity against CPU paths.
- Config parsing, provenance, and snapshot I/O.
- Force-error benchmark smoke runs.
- Sweep dry-runs.

CUDA-enabled builds should also be tested on a machine with an actual CUDA
toolchain and device, because CPU-only builds validate fallback behavior but do
not compile or execute `.cu` kernels.
