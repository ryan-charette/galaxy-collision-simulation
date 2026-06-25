from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from python.analysis import plot_snapshots
from python.utils.snapshots import load_snapshot


def _write_snapshot(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# time=0.2",
                "id,group_id,mass,x,y,z,vx,vy,vz,ax,ay,az",
                "0,0,1.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
                "1,1,2.0,1.0,0.5,0.25,0.0,0.0,0.0,0.0,0.0,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_diagnostics(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "step,time,n,total_mass,kinetic_energy,potential_energy,total_energy,momentum_x,momentum_y,momentum_z,center_of_mass_x,center_of_mass_y,center_of_mass_z,angular_momentum_x,angular_momentum_y,angular_momentum_z",
                "0,0.0,2,3.0,1.0,-2.0,-1.0,0.1,0.2,0.3,0.0,0.0,0.0,0.01,0.02,0.03",
                "1,0.1,2,3.0,1.1,-2.1,-1.0,0.2,0.3,0.4,0.0,0.0,0.0,0.02,0.03,0.04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_projection_plot_helpers_write_outputs(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot_000001.csv"
    _write_snapshot(snapshot_path)
    _write_diagnostics(tmp_path / "diagnostics.csv")

    scatter_output = tmp_path / "scatter.png"
    density_output = tmp_path / "density.png"
    diagnostics_output = tmp_path / "diagnostics.png"

    plot_snapshots._plot_snapshot(snapshot_path, tmp_path, scatter_output)
    plot_snapshots._plot_density(snapshot_path, tmp_path, density_output, bins=8, cmap="magma")
    plot_snapshots._plot_diagnostics(tmp_path, diagnostics_output)

    assert scatter_output.exists()
    assert density_output.exists()
    assert diagnostics_output.exists()


def test_density_projection_handles_zero_mass_particles(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot_000001.csv"
    _write_snapshot(snapshot_path)
    snapshot = load_snapshot(snapshot_path)
    snapshot = snapshot.__class__(
        step=snapshot.step,
        time=snapshot.time,
        ids=snapshot.ids,
        positions=snapshot.positions,
        velocities=snapshot.velocities,
        accelerations=snapshot.accelerations,
        masses=np.zeros_like(snapshot.masses),
        group_id=snapshot.group_id,
        path=snapshot.path,
    )
    fig, ax = plt.subplots()
    try:
        image = plot_snapshots._density_projection(ax, snapshot, (0, 1), ("x", "y"), bins=4, cmap="magma")
        assert image.get_array().shape == (4, 4)
    finally:
        plt.close(fig)


def test_plot_diagnostics_noops_when_file_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "missing.png"

    plot_snapshots._plot_diagnostics(tmp_path, output)

    assert not output.exists()


def test_plot_snapshots_main_writes_requested_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshot_000001.csv"
    _write_snapshot(snapshot_path)
    _write_diagnostics(tmp_path / "diagnostics.csv")
    output = tmp_path / "main_snapshot.png"
    diagnostics_output = tmp_path / "main_diagnostics.png"
    density_output = tmp_path / "main_density.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "fmm-galaxy-plot",
            "--input",
            str(tmp_path),
            "--snapshot",
            str(snapshot_path),
            "--output",
            str(output),
            "--diagnostics-output",
            str(diagnostics_output),
            "--density-output",
            str(density_output),
            "--density-bins",
            "8",
        ],
    )

    plot_snapshots.main()

    assert output.exists()
    assert diagnostics_output.exists()
    assert density_output.exists()


def test_plot_snapshots_main_can_skip_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshot_000001.csv"
    _write_snapshot(snapshot_path)
    output = tmp_path / "main_snapshot.png"
    diagnostics_output = tmp_path / "main_snapshot_diagnostics.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "fmm-galaxy-plot",
            "--input",
            str(tmp_path),
            "--snapshot",
            str(snapshot_path),
            "--output",
            str(output),
            "--no-diagnostics",
        ],
    )

    plot_snapshots.main()

    assert output.exists()
    assert not diagnostics_output.exists()
