"""Shared CSV, Parquet, and Markdown table helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write dictionaries to a CSV file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows to CSV or Parquet based on the output suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            pd.DataFrame(rows, columns=columns).to_parquet(path, index=False, engine="pyarrow")
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires pandas and pyarrow. Install project dependencies "
                "or choose a .csv output path."
            ) from exc
        return

    write_csv_rows(path, rows, columns)


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    """Return a GitHub-flavored Markdown table as a list of lines."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines
