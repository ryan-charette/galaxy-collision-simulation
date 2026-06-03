from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from python.utils.parquet_io import csv_snapshot_to_parquet
from python.utils.snapshots import load_snapshot


pytest.importorskip("pyarrow")


def test_parquet_snapshot_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "snapshot_000010.csv"
    parquet_path = tmp_path / "snapshot_000010.parquet"
    csv_path.write_text(
        "\n".join(
            [
                "# time=0.125",
                "id,group_id,mass,x,y,z,vx,vy,vz,ax,ay,az",
                "0,2,1.5,0.1,0.2,0.3,1.0,1.1,1.2,-0.1,-0.2,-0.3",
                "1,3,2.5,0.4,0.5,0.6,2.0,2.1,2.2,-0.4,-0.5,-0.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    csv_snapshot_to_parquet(csv_path, parquet_path, step=10, time=0.125)

    csv_snapshot = load_snapshot(csv_path)
    parquet_snapshot = load_snapshot(parquet_path)

    assert parquet_snapshot.step == csv_snapshot.step == 10
    assert parquet_snapshot.time == csv_snapshot.time == 0.125
    np.testing.assert_array_equal(parquet_snapshot.ids, csv_snapshot.ids)
    np.testing.assert_allclose(parquet_snapshot.positions, csv_snapshot.positions)
    np.testing.assert_allclose(parquet_snapshot.velocities, csv_snapshot.velocities)
    np.testing.assert_allclose(parquet_snapshot.masses, csv_snapshot.masses)
    np.testing.assert_array_equal(parquet_snapshot.group_id, csv_snapshot.group_id)
