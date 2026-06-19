#pragma once

#include "core/initial_conditions.hpp"
#include "core/simulation_params.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fmmgalaxy {

/// Snapshot output backends supported by the simulator.
enum class OutputFormat {
    /// Write `snapshot_*.csv` files.
    Csv,
    /// Convert CSV snapshots to Apache Parquet via the Python helper.
    Parquet,
    /// Disable snapshot and diagnostics output while still allowing simulation execution.
    None,
};

/// Parse an output format name from configuration text.
OutputFormat parse_output_format(const std::string& value);
/// Return the canonical lowercase configuration name for an output format.
std::string output_format_name(OutputFormat format);

/// Output controls for snapshots, diagnostics, metadata, and acceleration dumps.
struct OutputConfig {
    /// Directory where run artifacts are written.
    std::filesystem::path directory{"experiments/validation/smoke_test"};
    /// Snapshot file format.
    OutputFormat format{OutputFormat::Csv};
    /// Whether to write per-step acceleration dumps for force diagnostics.
    bool acceleration_dump{false};
};

/// @brief Complete simulation configuration loaded from TOML-like config files.
///
/// The configuration is intentionally plain data so scripts can generate and mutate configs
/// for sweeps, benchmark grids, and validation runs.
struct SimulationConfig {
    /// Human-readable run name.
    std::string name{"smoke_test"};
    /// Solver name such as `direct`, `tree`, `fmm`, `cuda-direct`, `cuda-tree`, or `cuda-fmm`.
    std::string solver{"direct"};
    /// Spatial dimension requested by the config; current particle storage is three-dimensional.
    int dim{3};
    /// Random seed for initial-condition generation.
    std::uint64_t seed{42};
    /// Total particle count after galaxy-count synchronization.
    std::size_t n_particles{0};
    /// Number of integration steps.
    int steps{40};
    /// Integration timestep in code units.
    double dt{0.01};
    /// Snapshot cadence in integration steps.
    int snapshot_every{10};
    /// Barnes-Hut/FMM opening angle.
    double tree_theta{0.6};
    /// Maximum number of particles in a tree leaf before subdivision.
    std::size_t tree_leaf_capacity{16};
    /// Multipole expansion order; supported values are normalized by solver code.
    int fmm_expansion_order{4};
    /// Shared gravitational constant and softening.
    PhysicsParams physics{};
    /// Galaxy initial-condition blocks.
    std::vector<GalaxyConfig> galaxies{};
    /// Output and artifact settings.
    OutputConfig output{};
};

/// Return the built-in smoke-test configuration.
SimulationConfig default_config();
/// Load and validate a simulation configuration file.
SimulationConfig load_config(const std::filesystem::path& path);

}  // namespace fmmgalaxy
