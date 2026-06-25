from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from python.utils.tables import markdown_table, write_csv_rows, write_table


def test_write_csv_rows_creates_parent_directories(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "rows.csv"

    write_csv_rows(output, [{"solver": "direct", "seconds": 1.25}], ["solver", "seconds"])

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"solver": "direct", "seconds": "1.25"}]


def test_write_table_uses_csv_for_non_parquet_suffix(tmp_path: Path) -> None:
    output = tmp_path / "summary.txt"

    write_table(output, [{"n": 128, "solver": "tree"}], ["n", "solver"])

    assert output.read_text(encoding="utf-8").splitlines() == ["n,solver", "128,tree"]


def test_write_table_uses_parquet_for_parquet_suffix(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    output = tmp_path / "summary.parquet"

    write_table(output, [{"n": 128, "solver": "tree"}], ["n", "solver"])

    frame = pd.read_parquet(output, engine="pyarrow")
    assert frame.to_dict(orient="records") == [{"n": 128, "solver": "tree"}]


def test_markdown_table_formats_values() -> None:
    lines = markdown_table(["solver", "rmse"], [["fmm", 1.2e-3], ["direct", 0.0]])

    assert lines == [
        "| solver | rmse |",
        "| --- | --- |",
        "| fmm | 0.0012 |",
        "| direct | 0.0 |",
    ]
