from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from python.utils import parquet_io


def _write_full_snapshot_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# time=0.5",
                "id,group_id,mass,x,y,z,vx,vy,vz,ax,ay,az",
                "0,0,1.0,0.0,0.1,0.2,1.0,1.1,1.2,-1.0,-1.1,-1.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parquet_cli_converts_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    csv_path = tmp_path / "snapshot_000004.csv"
    parquet_path = tmp_path / "snapshot_000004.parquet"
    _write_full_snapshot_csv(csv_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fmm-galaxy-parquet",
            "--input",
            str(csv_path),
            "--output",
            str(parquet_path),
            "--step",
            "4",
            "--time",
            "0.5",
        ],
    )

    parquet_io.main()

    frame = pd.read_parquet(parquet_path, engine="pyarrow")
    assert frame.loc[0, "step"] == 4
    assert frame.loc[0, "time"] == 0.5


def test_parquet_conversion_reports_missing_pyarrow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "snapshot_000004.csv"
    _write_full_snapshot_csv(csv_path)

    def raise_import_error(*_args: object, **_kwargs: object) -> None:
        raise ImportError("no pyarrow")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", raise_import_error)

    with pytest.raises(RuntimeError, match="Parquet output requires pyarrow"):
        parquet_io.csv_snapshot_to_parquet(csv_path, tmp_path / "out.parquet", step=4, time=0.5)
