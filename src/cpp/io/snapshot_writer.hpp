#pragma once

#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/particle.hpp"
#include "core/provenance.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

namespace fmmgalaxy {

/// @brief Writes simulator metadata, snapshots, diagnostics, and acceleration dumps.
///
/// The writer owns the output directory and diagnostics stream for one run. Snapshot output
/// can be CSV, Parquet, or disabled according to `SimulationConfig::output`.
class SnapshotWriter {
public:
    /// Create a writer and initialize the configured output directory.
    explicit SnapshotWriter(const SimulationConfig& config);

    /// Write `metadata.json` with configuration and provenance information.
    void write_metadata(
        const SimulationConfig& config,
        std::size_t particle_count,
        const RunProvenance& provenance
    );
    /// Write one particle snapshot at a simulation step/time.
    void write_snapshot(int step, double time, const std::vector<Particle>& particles);
    /// Write one acceleration dump for direct-vs-approximate residual datasets.
    void write_accelerations(int step, double time, const std::vector<Particle>& particles);
    /// Append one diagnostics row.
    void write_diagnostics(int step, double time, const Diagnostics& diagnostics, std::size_t particle_count);

private:
    std::filesystem::path directory_;
    std::ofstream diagnostics_stream_;
    OutputFormat format_{OutputFormat::Csv};
    bool enabled_{true};
    bool acceleration_dump_{false};
};

}  // namespace fmmgalaxy
