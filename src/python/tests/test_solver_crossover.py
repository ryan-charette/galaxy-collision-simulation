from __future__ import annotations

import csv
import math
from pathlib import Path

from python.analysis.solver_crossover import (
    AccuracyPoint,
    RuntimePoint,
    as_float,
    as_int,
    best_solver_by_n,
    cuda_crossover_rows,
    load_points,
    main,
    median_runtime,
    plot_outputs,
    parse_thresholds,
    read_accuracy_csv,
    read_runtime_csv,
    target_accuracy_rows,
    write_best_solver_csv,
    write_markdown,
    write_target_accuracy_csv,
)


def _runtime_point(solver: str, particles: int, seconds: float) -> RuntimePoint:
    return RuntimePoint(
        solver=solver,
        particles=particles,
        output_format="csv",
        seconds=seconds,
        particle_steps_per_second=particles / seconds,
        build_type="Release",
        compiler="GNU",
        cuda_available="false",
        cuda_device_name="",
        mpi_enabled="false",
        hostname="node",
    )


def _accuracy_point(solver: str, particles: int, force_rmse: float, seconds: float) -> AccuracyPoint:
    return AccuracyPoint(
        solver=solver,
        particles=particles,
        force_rmse=force_rmse,
        max_force_error=2.0 * force_rmse,
        relative_force_error=0.1 * force_rmse,
        seconds=seconds,
        particle_steps_per_second=particles / seconds,
        build_type="Release",
        compiler="GNU",
        cuda_available="false",
        cuda_device_name="",
        mpi_enabled="false",
        hostname="node",
    )


def test_numeric_parsers_fall_back_for_missing_or_bad_values() -> None:
    assert math.isnan(as_float(""))
    assert as_float("bad", default=1.5) == 1.5
    assert as_int("128.0") == 128
    assert as_int(None, default=7) == 7
    assert parse_thresholds(["1e-2", "0.001"]) == [1.0e-2, 1.0e-3]


def test_runtime_and_accuracy_csv_readers_fill_defaults(tmp_path: Path) -> None:
    runtime_csv = tmp_path / "runtime.csv"
    runtime_csv.write_text(
        "\n".join(
            [
                "solver,particles,output_format,seconds,steps,particle_steps_per_second,build_type,compiler,cuda_available,cuda_device_name,mpi_enabled,hostname",
                "direct,64,csv,4.0,2,,Release,GNU,false,,false,node",
                "tree,64,csv,2.0,2,32.0,Release,GNU,false,,false,node",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    accuracy_csv = tmp_path / "accuracy.csv"
    accuracy_csv.write_text(
        "\n".join(
            [
                "solver,particles,force_rmse,max_force_error,relative_force_error,seconds,particle_steps_per_second,build_type,compiler,cuda_available,cuda_device_name,mpi_enabled,hostname",
                "tree,64,1e-3,2e-3,3e-3,2.0,32.0,Release,GNU,false,,false,node",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runtimes = read_runtime_csv(runtime_csv)
    assert runtimes[0].particle_steps_per_second == 32.0
    assert runtimes[0].hardware_key == "node / Release / GNU / cuda=false / mpi=false"

    accuracy = read_accuracy_csv(accuracy_csv)
    assert accuracy[0].relative_force_error == 3.0e-3

    loaded_runtimes, loaded_accuracy = load_points([runtime_csv], [accuracy_csv])
    assert [point.solver for point in loaded_runtimes] == ["direct", "tree"]
    assert [point.solver for point in loaded_accuracy] == ["tree"]


def test_crossover_summaries_choose_fastest_qualifying_solver(tmp_path: Path) -> None:
    runtime_points = [
        _runtime_point("direct", 64, 4.0),
        _runtime_point("direct", 64, 2.0),
        _runtime_point("tree", 64, 1.5),
        _runtime_point("direct", 128, 5.0),
        _runtime_point("cuda-direct", 128, 3.0),
    ]
    accuracy_points = [
        _accuracy_point("tree", 64, 5.0e-4, 1.5),
        _accuracy_point("fmm", 64, 2.0e-4, 2.0),
        _accuracy_point("tree", 128, 2.0e-3, 3.0),
    ]

    medians = median_runtime(runtime_points)
    direct_64 = [point for point in medians if point.solver == "direct" and point.particles == 64][0]
    assert direct_64.seconds == 3.0

    best = best_solver_by_n(medians)
    assert {(point.particles, point.solver) for point in best} == {(64, "tree"), (128, "cuda-direct")}

    target_rows = target_accuracy_rows(accuracy_points, [1.0e-3, 1.0e-4])
    assert target_rows == [(1.0e-3, 64, "tree", 5.0e-4, 1.5)]

    assert cuda_crossover_rows(medians)[0] == ("direct", "cuda-direct", "128")

    best_csv = tmp_path / "best.csv"
    target_csv = tmp_path / "target.csv"
    markdown = tmp_path / "summary.md"
    write_best_solver_csv(best_csv, best)
    write_target_accuracy_csv(target_csv, target_rows)
    write_markdown(markdown, medians, accuracy_points, [1.0e-3])

    with best_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["solver"] == "tree"
    assert target_csv.read_text(encoding="utf-8").splitlines()[1].startswith("0.001,64,tree")
    assert "## CPU vs CUDA Crossover" in markdown.read_text(encoding="utf-8")


def test_plot_outputs_and_main_write_crossover_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_points = [_runtime_point("direct", 64, 4.0), _runtime_point("tree", 64, 2.0)]
    accuracy_points = [_accuracy_point("tree", 64, 1.0e-3, 2.0)]
    plot_dir = tmp_path / "plots"

    plot_outputs(plot_dir, runtime_points, accuracy_points)

    assert (plot_dir / "runtime_vs_n.png").exists()
    assert (plot_dir / "particle_steps_vs_n.png").exists()
    assert (plot_dir / "force_error_vs_runtime.png").exists()

    runtime_csv = tmp_path / "runtime.csv"
    runtime_csv.write_text(
        "\n".join(
            [
                "solver,particles,output_format,seconds,particle_steps_per_second,build_type,compiler,cuda_available,cuda_device_name,mpi_enabled,hostname",
                "direct,64,csv,4.0,16.0,Release,GNU,false,,false,node",
                "tree,64,csv,2.0,32.0,Release,GNU,false,,false,node",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    accuracy_csv = tmp_path / "accuracy.csv"
    accuracy_csv.write_text(
        "\n".join(
            [
                "solver,particles,force_rmse,max_force_error,relative_force_error,seconds,particle_steps_per_second,build_type,compiler,cuda_available,cuda_device_name,mpi_enabled,hostname",
                "tree,64,1e-3,2e-3,3e-3,2.0,32.0,Release,GNU,false,,false,node",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "main"
    monkeypatch.setattr(
        "sys.argv",
        [
            "fmm-galaxy-crossover",
            "--runtime-csv",
            str(runtime_csv),
            "--accuracy-csv",
            str(accuracy_csv),
            "--output",
            str(output_dir),
            "--target-rmse",
            "1e-2",
            "1e-3",
        ],
    )

    main()

    assert (output_dir / "solver_crossover_summary.md").exists()
    assert (output_dir / "best_solver_by_n.csv").exists()
    assert (output_dir / "target_accuracy_summary.csv").exists()


def test_solver_crossover_main_reports_missing_runtime_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["fmm-galaxy-crossover", "--runtime-csv", str(tmp_path / "missing.csv")],
    )

    try:
        main()
    except FileNotFoundError as exc:
        assert "No runtime CSV inputs were found" in str(exc)
    else:
        raise AssertionError("Expected missing runtime inputs to raise")
