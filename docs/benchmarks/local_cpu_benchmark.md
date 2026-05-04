# CPU Benchmark Results

Generated: 2026-05-02 14:40:11

Platform: `Windows-11-10.0.26200-SP0`

Build: Release CPU executable, MPI disabled, CUDA disabled.

| Solver | Particles | Steps | Median wall time (s) | Steps/s | Particle-steps/s |
|---|---:|---:|---:|---:|---:|
| `direct` | 250 | 20 | 0.035 | 573.56 | 143,391 |
| `tree` | 250 | 20 | 0.133 | 150.87 | 37,717 |
| `fmm` | 250 | 20 | 0.298 | 67.02 | 16,755 |
| `direct` | 500 | 20 | 0.051 | 390.19 | 195,096 |
| `tree` | 500 | 20 | 0.381 | 52.49 | 26,243 |
| `fmm` | 500 | 20 | 1.131 | 17.68 | 8,840 |
| `direct` | 1000 | 20 | 0.103 | 194.56 | 194,560 |
| `tree` | 1000 | 20 | 1.377 | 14.52 | 14,521 |
| `fmm` | 1000 | 20 | 5.878 | 3.40 | 3,402 |
