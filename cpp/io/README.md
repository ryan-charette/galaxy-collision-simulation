# Snapshot I/O

This module writes portable CSV snapshots by default and Apache Parquet snapshots when
`[output] format = "parquet"`:

- `snapshot_000000.csv`, `snapshot_000010.csv`, ...
- `snapshot_000000.parquet`, `snapshot_000010.parquet`, ...
- `diagnostics.csv`
- `metadata.json`

CSV keeps the current simulator easy to build and inspect. Parquet output uses the
Python converter in `python.utils.parquet_io`, so the runtime Python environment must
include `pyarrow`. Set `FMM_GALAXY_PYTHON` if the simulator should use a specific
Python executable for conversion.
