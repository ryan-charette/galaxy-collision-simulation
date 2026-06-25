from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from python.utils.snapshots import (
    iter_group_masks,
    list_snapshot_files,
    load_acceleration_dump,
    load_diagnostics,
    load_latest_snapshot,
    load_snapshot,
    load_snapshots,
)


def _write_snapshot(path: Path, *, step_time: float | None = None, x_offset: float = 0.0) -> None:
    lines = []
    if step_time is not None:
        lines.append(f"# time={step_time}")
    lines.extend(
        [
            "id,group_id,mass,x,y,vx,vy,ax,ay",
            f"0,2,1.5,{0.1 + x_offset},0.2,1.0,1.1,-0.1,-0.2",
            f"1,1,2.5,{0.4 + x_offset},0.5,2.0,2.1,-0.4,-0.5",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_list_and_load_csv_snapshots_with_2d_fallbacks(tmp_path: Path) -> None:
    _write_snapshot(tmp_path / "snapshot_000010.csv", step_time=0.25, x_offset=10.0)
    _write_snapshot(tmp_path / "snapshot_000002.csv", step_time=0.05, x_offset=2.0)
    (tmp_path / "snapshot_final.csv").write_text("", encoding="utf-8")
    (tmp_path / "snapshot_000001.txt").write_text("", encoding="utf-8")

    files = list_snapshot_files(tmp_path)
    assert [path.name for path in files] == ["snapshot_000002.csv", "snapshot_000010.csv"]

    snapshot = load_snapshot(files[0])
    assert snapshot.step == 2
    assert snapshot.time == 0.05
    np.testing.assert_array_equal(snapshot.ids, np.array([0, 1]))
    np.testing.assert_array_equal(snapshot.group_id, np.array([2, 1]))
    np.testing.assert_allclose(snapshot.positions[:, 2], 0.0)
    np.testing.assert_allclose(snapshot.velocities[:, 2], 0.0)
    np.testing.assert_allclose(snapshot.accelerations[:, 2], 0.0)

    latest = load_latest_snapshot(tmp_path)
    assert latest.step == 10


def test_load_snapshots_stride_and_empty_directory_errors(tmp_path: Path) -> None:
    _write_snapshot(tmp_path / "snapshot_000001.csv", step_time=0.1)
    _write_snapshot(tmp_path / "snapshot_000002.csv", step_time=0.2)
    _write_snapshot(tmp_path / "snapshot_000003.csv", step_time=0.3)

    assert [snapshot.step for snapshot in load_snapshots(tmp_path, stride=2)] == [1, 3]
    with pytest.raises(ValueError, match="stride must be positive"):
        load_snapshots(tmp_path, stride=0)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"No snapshot_\*\.csv"):
        load_latest_snapshot(empty)


def test_load_acceleration_dump_and_diagnostics(tmp_path: Path) -> None:
    acceleration_path = tmp_path / "accelerations_000007.csv"
    _write_snapshot(acceleration_path, step_time=0.7)

    dump = load_acceleration_dump(acceleration_path)
    assert dump.step == 7
    assert dump.time == 0.7
    np.testing.assert_allclose(dump.accelerations[:, :2], [[-0.1, -0.2], [-0.4, -0.5]])

    with pytest.raises(ValueError, match="Acceleration dumps are CSV files"):
        load_acceleration_dump(tmp_path / "accelerations_000007.parquet")

    diagnostics_path = tmp_path / "diagnostics.csv"
    diagnostics_path.write_text(
        "step,time,total_energy\n0,0.0,-1.5\n1,0.1,-1.4\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(load_diagnostics(tmp_path)["total_energy"], [-1.5, -1.4])
    np.testing.assert_allclose(load_diagnostics(diagnostics_path)["time"], [0.0, 0.1])


def test_iter_group_masks_are_sorted_and_stable() -> None:
    group_id = np.array([3, 1, 3, 2, 1])

    groups = list(iter_group_masks(group_id))

    assert [group for group, _mask in groups] == [1, 2, 3]
    np.testing.assert_array_equal(groups[0][1], np.array([False, True, False, False, True]))
