# Snapshot I/O

This module writes portable CSV snapshots for the MVP:

- `snapshot_000000.csv`, `snapshot_000010.csv`, ...
- `diagnostics.csv`
- `metadata.json`

The long-term target is HDF5 for large runs, but CSV keeps the current simulator easy to build, inspect, and load from Python without native dependencies.
