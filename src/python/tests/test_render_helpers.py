from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from python.animation import interactive_viewer, render_scientific_gif, render_snapshots
from python.utils.snapshots import Snapshot


def _snapshot(step: int, positions: np.ndarray | None = None) -> Snapshot:
    positions = (
        positions
        if positions is not None
        else np.array([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0], [-1.0, 3.0, 0.5]])
    )
    count = len(positions)
    return Snapshot(
        step=step,
        time=0.1 * step,
        ids=np.arange(count),
        positions=positions,
        velocities=np.zeros((count, 3)),
        accelerations=np.zeros((count, 3)),
        masses=np.linspace(1.0, 2.0, count),
        group_id=np.arange(count) % 2,
        path=Path(f"snapshot_{step:06d}.csv"),
    )


def test_render_snapshot_projection_and_bounds_helpers() -> None:
    snapshots = [_snapshot(0), _snapshot(1, np.array([[3.0, -2.0, 1.0], [4.0, 2.0, -3.0]]))]

    xy = render_snapshots._project(snapshots[0].positions, "xy", azimuth=0.0, elevation=0.0)
    np.testing.assert_allclose(xy, snapshots[0].positions[:, [0, 1]])
    yz = render_snapshots._project(snapshots[0].positions, "yz", azimuth=0.0, elevation=0.0)
    np.testing.assert_allclose(yz, snapshots[0].positions[:, [1, 2]])
    camera = render_snapshots._project(
        np.array([[1.0, 0.0, 0.0]]),
        "camera",
        azimuth=90.0,
        elevation=0.0,
    )
    np.testing.assert_allclose(camera, [[0.0, 1.0]], atol=1.0e-12)

    xmin, xmax, ymin, ymax = render_snapshots._bounds(snapshots, "xy", 0.0, 0.0)
    assert xmin < -1.0
    assert xmax > 4.0
    assert ymin < -2.0
    assert ymax > 3.0

    bounds3d = render_snapshots._bounds3d(snapshots)
    assert len(bounds3d) == 6
    assert bounds3d[0] < bounds3d[1]


def test_interactive_viewer_payload_decimates_particles() -> None:
    snapshots = [_snapshot(0), _snapshot(1)]

    np.testing.assert_array_equal(interactive_viewer._decimated_indices(4, 10), np.arange(4))
    np.testing.assert_array_equal(interactive_viewer._decimated_indices(5, 3), np.array([0, 2, 4]))

    center, radius = interactive_viewer._scene_bounds(snapshots)
    assert len(center) == 3
    assert radius > 0.0

    payload = interactive_viewer._viewer_payload(snapshots, max_particles=2)
    assert payload["radius"] == radius
    assert len(payload["frames"]) == 2
    assert len(payload["frames"][0]["points"]) == 2


def test_scientific_gif_sampling_projection_and_labels(tmp_path: Path) -> None:
    paths = [tmp_path / f"snapshot_{step:06d}.csv" for step in range(5)]
    selected = render_scientific_gif._select_frame_paths(paths, stride=2, max_frames=2)
    assert selected == [paths[0], paths[4]]

    snapshot = _snapshot(3)
    sample_ids = render_scientific_gif._sample_ids(snapshot, max_particles=2, seed=1)
    assert sample_ids is not None
    assert len(sample_ids) == 2
    sampled = render_scientific_gif._apply_sample(snapshot, sample_ids)
    assert len(sampled.ids) == 2
    assert render_scientific_gif._apply_sample(snapshot, {9999}) is snapshot

    rotated = render_scientific_gif._rotate(np.array([[1.0, 0.0, 0.0]]), 90.0, 0.0)
    np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1.0e-12)

    bounds = render_scientific_gif.SceneBounds(center=np.zeros(3), span=4.0)
    args = argparse.Namespace(
        azimuth_start=0.0,
        azimuth_span=90.0,
        elevation=0.0,
        zoom=1.0,
        width=200,
        height=100,
    )
    projected = render_scientific_gif._project_snapshot(snapshot, 10, bounds, 1, 3, args)
    assert projected.total_particles == 10
    assert projected.x.shape == snapshot.ids.shape

    label = render_scientific_gif._label_text(
        "n={n} rendered={rendered} step={step} path={path}",
        projected,
    )
    assert label == "n=10 rendered=3 step=3 path=snapshot_000003.csv"

    valid = render_scientific_gif._valid_mask(projected, width=200, height=100)
    assert valid.dtype == np.bool_

    try:
        render_scientific_gif._select_frame_paths(paths, stride=0, max_frames=0)
    except ValueError as exc:
        assert "--frame-stride must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid frame stride to raise")


def test_render_snapshot_modes_delegate_to_save_animation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshots = [_snapshot(0), _snapshot(1)]
    saved_outputs: list[Path] = []

    def fake_save_animation(fig, ani, output: Path, fps: int, dpi: int) -> None:
        ani._draw_was_started = True
        saved_outputs.append(output)
        assert fps == 12
        assert dpi == 80
        plt.close(fig)

    monkeypatch.setattr(render_snapshots, "_save_animation", fake_save_animation)
    args = argparse.Namespace(
        projection="xy",
        azimuth=20.0,
        elevation=15.0,
        camera_orbit=True,
        fps=12,
        dpi=80,
        output=tmp_path / "scatter.gif",
        point_size=3.0,
        density_bins=8,
        cmap="magma",
    )

    render_snapshots._render_scatter(snapshots, args)
    args.output = tmp_path / "density.gif"
    render_snapshots._render_density(snapshots, args)
    args.output = tmp_path / "scatter3d.gif"
    render_snapshots._render_scatter3d(snapshots, args)

    assert saved_outputs == [
        tmp_path / "scatter.gif",
        tmp_path / "density.gif",
        tmp_path / "scatter3d.gif",
    ]


def test_scientific_gif_rendering_pipeline_writes_small_gif(tmp_path: Path) -> None:
    for step in [0, 1]:
        path = tmp_path / f"snapshot_{step:06d}.csv"
        path.write_text(
            "\n".join(
                [
                    f"# time={0.1 * step}",
                    "id,group_id,mass,x,y,z,vx,vy,vz,ax,ay,az",
                    "0,0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
                    f"1,1,2.0,{1.0 + step},0.5,0.25,0.0,0.0,0.0,0.0,0.0,0.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    output = tmp_path / "out.gif"
    args = argparse.Namespace(
        input=tmp_path,
        output=output,
        mode="scatter",
        width=32,
        height=24,
        fps=2,
        frame_stride=1,
        max_frames=1,
        max_particles=1,
        sample_seed=3,
        bounds_stride=1,
        bounds_margin=0.1,
        zoom=0.9,
        azimuth_start=0.0,
        azimuth_span=45.0,
        elevation=0.0,
        density_percentile=99.0,
        point_radius=1.0,
        point_alpha=220,
        label="step {step} rendered {rendered}",
        no_label=False,
    )

    render_scientific_gif.render_gif(args)

    assert output.exists()
