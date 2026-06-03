# Architecture

## System overview

```text
                +------------------------------+
                |        Config files          |
                |  TOML/YAML/JSON experiment   |
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
|  | - Direct O(N^2) baseline                               |  |
|  | - Treecode/Barnes-Hut transitional solver              |  |
|  | - Fast Multipole Method                                |  |
|  +-------------------------------------------------------+  |
|          |                |                    |             |
|          v                v                    v             |
|  +------------+   +------------+   +----------------------+  |
|  | MPI        |   | CUDA       |   | Snapshot I/O         |  |
|  | ranks      |   | kernels    |   | HDF5/binary + JSON   |  |
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

## C++ modules

### `src/cpp/core`

Core types and algorithms that should not depend on MPI or CUDA:

- vectors,
- particles,
- simulation state,
- integrators,
- diagnostics,
- configuration helpers.

### `src/cpp/direct`

Direct force solver. This is the correctness baseline.

### `src/cpp/fmm`

Tree and FMM implementation:

- octree node layout,
- tree construction,
- P2M,
- M2M,
- M2L,
- L2L,
- L2P,
- P2P near-field.

### `src/cpp/mpi`

Distributed wrappers:

- rank-local particle ownership,
- global reductions,
- tree summary exchange,
- ghost particle exchange,
- distributed diagnostics.

### `src/cpp/cuda`

GPU-specific pieces:

- device buffers,
- force kernels,
- integration kernels,
- CUDA error wrappers.

### `src/cpp/io`

Snapshot and metadata writers.

## Python modules

### `src/python/analysis`

- force-error plots,
- energy drift plots,
- scaling plots,
- benchmark tables.

### `src/python/animation`

- scatter renderer,
- density renderer,
- camera paths,
- video export.

### `src/python/utils`

- snapshot loading,
- config parsing,
- shared plotting utilities.

## Build modes

Target build modes:

```text
Serial CPU:      direct + tree/FMM
MPI CPU:         distributed direct/FMM
Serial CUDA:     GPU-accelerated kernels on one device
MPI + CUDA:      one rank per GPU, eventual target
```

## Data flow

1. A config file defines the experiment.
2. The C++ engine generates or loads particles.
3. The selected solver computes accelerations.
4. The integrator advances particle state.
5. Snapshots are written periodically.
6. Python loads snapshots for diagnostics and rendering.
