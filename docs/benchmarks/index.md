# Benchmarks

Benchmark documentation is split between durable notes and generated artifacts.
Runtime summaries are end-to-end executable timings from
`scripts/run_benchmarks.py`, so they include process launch, config parsing,
solver setup, integration, and any enabled output work.

```{toctree}
:maxdepth: 1

README
local_cpu_benchmark
```

## Reproducibility

Each benchmark output directory contains simulator metadata when available:

- Git commit, branch, and dirty working-tree state.
- Config path and config SHA-256 hash.
- Compiler, build type, CUDA/MPI context, hostname, and UTC timestamp.

Generated benchmark Markdown summaries include the commit SHA for each grouped
result, or `unavailable` when git metadata could not be read.

## Common Commands

Run a small benchmark suite:

```bash
python scripts/run_benchmarks.py --particles 250 500 1000 --steps 20
```

Compare output formats:

```bash
python scripts/run_benchmarks.py \
  --output-formats csv parquet \
  --particles 10000 \
  --steps 10
```

Run the force-error smoke suite:

```bash
python scripts/run_force_error_benchmarks.py --smoke
```

Generate solver-crossover analysis from runtime data:

```bash
python -m python.analysis.solver_crossover \
  --runtime-csv docs/benchmarks/local_cpu_benchmark.csv
```
