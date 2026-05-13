# Benchmarks

This directory keeps benchmark summaries and the raw artifacts used to produce
them. Results are end-to-end executable timings from `scripts/run_benchmarks.py`,
so they include process launch, config parsing, solver setup, integration, and
any enabled output work.

## Available Runs

| Run | Platform | Summary |
|---|---|---|
| [`local_cpu_benchmark.md`](local_cpu_benchmark.md) | Windows CPU-only build | Small smoke-scale CPU comparison used for README artifact generation. |
| [`a100-20260513-074454`](a100-20260513-074454/README.md) | NVIDIA A100-SXM4-40GB, CUDA 13.0 | CUDA sanity, monopole, p=2, and p=4 runs up to 1,000,000 particles. |

## Notes

- GPU benchmark cases used `[output] format = "none"` so snapshot and
  diagnostics I/O would not dominate large-particle throughput measurements.
- The A100 run was produced from a downloaded repository zip, so the archive did
  not capture a git commit SHA. The results correspond to the CUDA tree/FMM
  implementation present in this branch at the time of the run.
- Raw per-replicate CSVs, generated Markdown summaries, benchmark configs,
  build logs, CTest logs, and system metadata are preserved inside each run
  folder when available.
