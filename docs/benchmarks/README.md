# Benchmarks

This directory keeps benchmark summaries and the raw artifacts used to produce
them. Results are end-to-end executable timings from `scripts/run_benchmarks.py`,
so they include process launch, config parsing, solver setup, integration, and
any enabled output work.

Each benchmark output directory contains the simulator `metadata.json`, including
the git commit, dirty state, config path, config SHA-256 hash, compiler, build
type, CUDA/MPI context, hostname, and UTC timestamp. Generated benchmark Markdown
summaries include the commit SHA for each grouped result, or `unavailable` when
git metadata could not be read.

## Available Runs

| Run | Platform | Summary |
|---|---|---|
| [`local_cpu_benchmark.md`](local_cpu_benchmark.md) | Windows CPU-only build | Small smoke-scale CPU comparison used for README artifact generation. |
| [`a100-20260513-074454`](a100-20260513-074454/README.md) | NVIDIA A100-SXM4-40GB, CUDA 13.0 | CUDA sanity, monopole, p=2, and p=4 runs up to 1,000,000 particles. |

## Notes

- GPU benchmark cases used `[output] format = "none"` so snapshot and
  diagnostics I/O would not dominate large-particle throughput measurements.
- Output-format benchmarks can compare CSV and Parquet in one run, for example:
  `python scripts/run_benchmarks.py --output-formats csv parquet --particles 10000 --steps 10`.
- Force-error benchmarks compare tree/FMM accelerations against direct summation
  at step 0 and track diagnostics drift over a short integration window:
  `python scripts/run_force_error_benchmarks.py --smoke` for a CI-scale run, or
  omit `--smoke` for the standard sweep. Results are written under
  `experiments/accuracy/`.
- Generic parameter sweeps are defined in YAML and launched with
  `python scripts/sweep.py --grid configs/sweeps/theta_leaf_order.yaml`. The
  sweep runner writes generated configs, raw logs, per-run output directories,
  `sweep_summary.csv`, optional `sweep_summary.parquet`, and
  `sweep_metadata.json`.
- Solver crossover analysis is generated from runtime and accuracy CSVs with
  `python -m python.analysis.solver_crossover`. Use
  `scripts/run_benchmarks.py --crossover-suite` to produce a wider runtime sweep
  with both output-disabled and CSV-output cases before running the analysis.
- The A100 run was produced from a downloaded repository zip, so the archive did
  not capture a git commit SHA. The results correspond to the CUDA tree/FMM
  implementation present in this branch at the time of the run.
- Raw per-replicate CSVs, generated Markdown summaries, benchmark configs,
  build logs, CTest logs, and system metadata are preserved inside each run
  folder when available.
