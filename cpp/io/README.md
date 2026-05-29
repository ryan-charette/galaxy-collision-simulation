# Snapshot I/O

This module writes portable CSV snapshots by default and Apache Parquet snapshots when
`[output] format = "parquet"`:

- `snapshot_000000.csv`, `snapshot_000010.csv`, ...
- `snapshot_000000.parquet`, `snapshot_000010.parquet`, ...
- `accelerations_000000.csv`, ... when `[output] acceleration_dump = true`
- `diagnostics.csv`
- `metadata.json`

CSV keeps the current simulator easy to build and inspect. Parquet output uses the
Python converter in `python.utils.parquet_io`, so the runtime Python environment must
include `pyarrow`. Set `FMM_GALAXY_PYTHON` if the simulator should use a specific
Python executable for conversion.

Acceleration dumps are lightweight CSV files with particle ids, positions,
velocities, masses, and computed accelerations. They are intended for
direct-vs-approximate residual datasets and can be written even when
`[output] format = "none"` disables snapshots and diagnostics.

`metadata.json` is always written, including for `format = "none"`. It captures
source provenance, build/runtime context, CUDA/MPI availability, timestamp,
hostname, config path, and config SHA-256 hash.
