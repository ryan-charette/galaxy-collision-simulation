# Benchmark Notes

This directory keeps benchmark summaries and raw artifacts used to produce
them. Results are end-to-end executable timings from `scripts/run_benchmarks.py`,
so they include process launch, config parsing, solver setup, integration, and
any enabled output work.

## Available Runs

| Run | Platform | Summary |
|---|---|---|
| [Local CPU benchmark](local_cpu_benchmark.md) | Windows CPU-only build | Small smoke-scale CPU comparison used for README artifact generation. |

Additional large-run artifacts may be stored under `experiments/benchmarks/`
when generated locally or on a GPU machine.

## Notes

- GPU benchmark cases should normally use `[output] format = "none"` so
  snapshot and diagnostics I/O do not dominate large-particle throughput
  measurements.
- Output-format benchmarks can compare CSV and Parquet in one run:

  ```bash
  python scripts/run_benchmarks.py \
    --output-formats csv parquet \
    --particles 10000 \
    --steps 10
  ```

- Force-error benchmarks compare tree/FMM accelerations against direct
  summation at step 0 and track diagnostics drift over a short integration
  window:

  ```bash
  python scripts/run_force_error_benchmarks.py --smoke
  ```

- Generic parameter sweeps are defined in YAML and launched with:

  ```bash
  python scripts/sweep.py --grid configs/sweeps/theta_leaf_order.yaml
  ```

  The sweep runner writes generated configs, raw logs, per-run output
  directories, `sweep_summary.csv`, optional `sweep_summary.parquet`, and
  `sweep_metadata.json`.

- Solver crossover analysis is generated from runtime and accuracy CSVs:

  ```bash
  python -m python.analysis.solver_crossover
  ```

  Use `scripts/run_benchmarks.py --crossover-suite` to produce a wider runtime
  sweep with both output-disabled and CSV-output cases before running the
  analysis.
