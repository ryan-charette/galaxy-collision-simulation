#pragma once

#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/particle.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

namespace fmmgalaxy {

class SnapshotWriter {
public:
    explicit SnapshotWriter(const SimulationConfig& config);

    void write_metadata(const SimulationConfig& config, std::size_t particle_count);
    void write_snapshot(int step, double time, const std::vector<Particle>& particles);
    void write_diagnostics(int step, double time, const Diagnostics& diagnostics, std::size_t particle_count);

private:
    std::filesystem::path directory_;
    std::ofstream diagnostics_stream_;
};

}  // namespace fmmgalaxy
