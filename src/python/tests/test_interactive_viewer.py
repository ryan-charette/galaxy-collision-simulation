from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from python.animation import interactive_viewer
from python.utils.snapshots import Snapshot


def _snapshot() -> Snapshot:
    return Snapshot(
        step=3,
        time=0.3,
        ids=np.arange(3),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        velocities=np.zeros((3, 3)),
        accelerations=np.zeros((3, 3)),
        masses=np.array([1.0, 2.0, 3.0]),
        group_id=np.array([0, 1, 1]),
        path=Path("snapshot_000003.csv"),
    )


def test_interactive_viewer_main_writes_self_contained_html(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "viewer.html"
    monkeypatch.setattr(interactive_viewer, "load_snapshots", lambda _input, stride: [_snapshot()])
    monkeypatch.setattr(
        "sys.argv",
        [
            "fmm-galaxy-viewer",
            "--input",
            str(tmp_path),
            "--output",
            str(output),
            "--stride",
            "2",
            "--max-particles",
            "2",
        ],
    )

    interactive_viewer.main()

    html = output.read_text(encoding="utf-8")
    assert "__SCENE_JSON__" not in html
    assert "FMM Galaxy Viewer" in html
    assert '"frames":[{"step":3' in html


def test_interactive_viewer_main_rejects_empty_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(interactive_viewer, "load_snapshots", lambda _input, stride: [])
    monkeypatch.setattr(
        "sys.argv",
        ["fmm-galaxy-viewer", "--input", str(tmp_path), "--output", str(tmp_path / "viewer.html")],
    )

    with pytest.raises(FileNotFoundError, match="No snapshots found"):
        interactive_viewer.main()
