# Benchmark Results

Generated: 2026-05-13 07:51:23

Platform: `Linux-5.15.0-151-generic-x86_64-with-glibc2.39`

Build: Release executable. CUDA use depends on the selected solver and build configuration.

| Solver | Particles | Steps | Median wall time (s) | Steps/s | Particle-steps/s |
|---|---:|---:|---:|---:|---:|
| `cuda-tree` | 50000 | 5 | 0.758 | 6.59 | 329,642 |
| `cuda-fmm` | 50000 | 5 | 2.030 | 2.46 | 123,174 |
| `cuda-tree` | 100000 | 5 | 0.926 | 5.40 | 539,729 |
| `cuda-fmm` | 100000 | 5 | 3.886 | 1.29 | 128,679 |
| `cuda-tree` | 250000 | 5 | 1.734 | 2.88 | 720,835 |
| `cuda-fmm` | 250000 | 5 | 10.067 | 0.50 | 124,165 |
| `cuda-tree` | 500000 | 5 | 3.174 | 1.58 | 787,735 |
| `cuda-fmm` | 500000 | 5 | 22.072 | 0.23 | 113,265 |
| `cuda-tree` | 1000000 | 5 | 7.079 | 0.71 | 706,330 |
| `cuda-fmm` | 1000000 | 5 | 48.268 | 0.10 | 103,589 |
