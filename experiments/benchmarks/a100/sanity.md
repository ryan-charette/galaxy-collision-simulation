# Benchmark Results

Generated: 2026-05-13 07:46:10

Platform: `Linux-5.15.0-151-generic-x86_64-with-glibc2.39`

Build: Release executable. CUDA use depends on the selected solver and build configuration.

| Solver | Particles | Steps | Median wall time (s) | Steps/s | Particle-steps/s |
|---|---:|---:|---:|---:|---:|
| `cuda-direct` | 1000 | 3 | 0.547 | 5.48 | 5,481 |
| `cuda-tree` | 1000 | 3 | 0.524 | 5.73 | 5,728 |
| `cuda-fmm` | 1000 | 3 | 0.523 | 5.74 | 5,737 |
| `cuda-direct` | 5000 | 3 | 0.524 | 5.73 | 28,631 |
| `cuda-tree` | 5000 | 3 | 0.553 | 5.42 | 27,101 |
| `cuda-fmm` | 5000 | 3 | 0.583 | 5.15 | 25,738 |
| `cuda-direct` | 10000 | 3 | 0.537 | 5.58 | 55,847 |
| `cuda-tree` | 10000 | 3 | 0.576 | 5.21 | 52,091 |
| `cuda-fmm` | 10000 | 3 | 0.674 | 4.45 | 44,511 |
