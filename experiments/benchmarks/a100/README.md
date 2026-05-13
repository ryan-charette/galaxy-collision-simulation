# A100 CUDA Benchmark Results

Generated from the `a100-20260513-074454` benchmark archive collected on
2026-05-13. The run used an NVIDIA A100-SXM4-40GB with CUDA 13.0 and preserved
the raw benchmark CSVs, generated Markdown tables, per-case configs, build logs,
CTest logs, and system metadata in this folder.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 40 GiB |
| Driver | 580.65.06 |
| CUDA toolkit | 13.0, `nvcc` 13.0.88 |
| Host | Linux 5.15.0, x86_64 |
| CPU | 2x AMD EPYC 7532, 128 hardware threads |
| Memory | 503 GiB |
| CMake | 3.28.3 |
| Build | Release, CUDA enabled, MPI disabled |
| Validation | `ctest` smoke suite passed |

The repository was tested from a downloaded zip instead of a git checkout, so no
git SHA was captured in the benchmark bundle.

## Headline Results

The fastest measured path is `cuda-tree` with monopole forces
(`fmm_expansion_order = 0`). Its best throughput in this archive is the
500,000-particle case at 787,735 particle-steps/s.

| Solver | Expansion | Particles | Steps | Median wall time (s) | Particle-steps/s |
|---|---:|---:|---:|---:|---:|
| `cuda-tree` | 0 | 500,000 | 5 | 3.174 | 787,735 |
| `cuda-tree` | 0 | 1,000,000 | 5 | 7.079 | 706,330 |
| `cuda-fmm` | 0 | 1,000,000 | 5 | 48.268 | 103,589 |
| `cuda-tree` | 2 | 250,000 | 5 | 27.300 | 45,788 |
| `cuda-fmm` | 2 | 250,000 | 5 | 37.007 | 33,777 |
| `cuda-tree` | 4 | 250,000 | 5 | 54.505 | 22,933 |
| `cuda-fmm` | 4 | 250,000 | 5 | 66.925 | 18,678 |

## Monopole Scaling

`cuda-fmm` is slower than `cuda-tree` in the monopole run because this
implementation still builds the tree and FMM interaction data on the CPU, then
launches GPU kernels over flattened interaction lists. At `p=0`, the tree path
does less CPU-side and memory-movement work while retaining a compact GPU
evaluation kernel.

| Particles | `cuda-tree` median (s) | `cuda-tree` particle-steps/s | `cuda-fmm` median (s) | `cuda-fmm` particle-steps/s | FMM/tree wall-time ratio |
|---:|---:|---:|---:|---:|---:|
| 50,000 | 0.758 | 329,642 | 2.030 | 123,174 | 2.68x |
| 100,000 | 0.926 | 539,729 | 3.886 | 128,679 | 4.19x |
| 250,000 | 1.734 | 720,835 | 10.067 | 124,165 | 5.81x |
| 500,000 | 3.174 | 787,735 | 22.072 | 113,265 | 6.95x |
| 1,000,000 | 7.079 | 706,330 | 48.268 | 103,589 | 6.82x |

## Higher-Order Runs

Higher expansion orders reduce the tree/FMM gap, but both solvers are much
slower than the monopole path. These cases are useful when validating the
specialized p=2 and p=4 CUDA kernels, not as the current high-throughput setting.

| Expansion | Particles | `cuda-tree` median (s) | `cuda-fmm` median (s) | FMM/tree wall-time ratio |
|---:|---:|---:|---:|---:|
| 2 | 50,000 | 4.812 | 5.946 | 1.24x |
| 2 | 100,000 | 10.046 | 13.354 | 1.33x |
| 2 | 250,000 | 27.300 | 37.007 | 1.36x |
| 4 | 50,000 | 9.248 | 10.581 | 1.14x |
| 4 | 100,000 | 19.899 | 24.605 | 1.24x |
| 4 | 250,000 | 54.505 | 66.925 | 1.23x |

## Raw Artifacts

- [`system.txt`](system.txt): GPU, CUDA, CMake, CPU, and memory details.
- [`configure.log`](configure.log), [`build.log`](build.log), and
  [`ctest.log`](ctest.log): configure, compile, and validation output.
- [`sanity.csv`](sanity.csv) and [`sanity.md`](sanity.md): short CUDA direct,
  tree, and FMM smoke-scale timings.
- [`monopole.csv`](monopole.csv) and [`monopole.md`](monopole.md): p=0 tree/FMM
  runs from 50,000 to 1,000,000 particles.
- [`p2.csv`](p2.csv) and [`p2.md`](p2.md): p=2 tree/FMM runs from 50,000 to
  250,000 particles.
- [`p4.csv`](p4.csv) and [`p4.md`](p4.md): p=4 tree/FMM runs from 50,000 to
  250,000 particles.
