# Run a Simulation

Run the smoke simulation after building the simulator:

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

`metadata.json` records reproducibility context for every run, including the git
commit and branch, dirty working-tree state, build type, compiler, requested
CMake MPI/CUDA options, CUDA availability and device name, MPI rank count,
hostname, UTC timestamp, config path, and config SHA-256 hash. This file is
written even when `[output] format = "none"` disables snapshots and diagnostics.

## Snapshot Format

For larger analysis workflows, switch the snapshot format to Parquet:

```toml
[output]
directory = "experiments/example"
format = "parquet" # csv, parquet, none
snapshot_every = 10
```

Parquet conversion uses the Python tooling and requires `pyarrow`. If the
simulator should use a specific interpreter, set `FMM_GALAXY_PYTHON` before
running it.

## Solver Selection

Choose a solver in the config:

```toml
[simulation]
solver = "fmm"          # direct, tree, fmm, cuda-direct, cuda-tree, cuda-fmm
dim = 3
tree_theta = 0.6
tree_leaf_capacity = 16
fmm_expansion_order = 4 # 0 = monopole, 2 = quadrupole, 4 = fourth-order Cartesian
```

## MPI

Run with MPI when available:

```bash
mpirun -np 4 ./build/fmm_galaxy_sim --config configs/smoke_test.toml
```

## CUDA

Run the CUDA direct/P2P kernel when a CUDA device is available:

```toml
[simulation]
solver = "cuda-direct"
```

For larger GPU throughput tests, use `cuda-tree` or `cuda-fmm`, set
`fmm_expansion_order = 0` for the fastest monopole path, and disable
CSV/diagnostic output. The CUDA layer reuses device buffers across steps, stages
transfers through pinned memory, caches static mass/group fields on the GPU,
uses SoA force inputs, tiles `cuda-direct` through shared memory, and
specializes tree/FMM kernels for p=0, p=2, and p=4:

```toml
[simulation]
solver = "cuda-fmm"
fmm_expansion_order = 0 # fastest high-scale setting

[output]
format = "none"
```
