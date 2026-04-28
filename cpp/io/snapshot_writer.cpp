#include "io/snapshot_writer.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fmmgalaxy {

namespace {

std::string escaped_json(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '"' || ch == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

std::string snapshot_filename(int step) {
    std::ostringstream name;
    name << "snapshot_" << std::setw(6) << std::setfill('0') << step << ".csv";
    return name.str();
}

}  // namespace

SnapshotWriter::SnapshotWriter(const SimulationConfig& config) : directory_(config.output.directory) {
    if (config.output.format != "csv") {
        throw std::runtime_error("Only csv snapshot output is implemented in this MVP");
    }

    std::filesystem::create_directories(directory_);

    diagnostics_stream_.open(directory_ / "diagnostics.csv", std::ios::trunc);
    if (!diagnostics_stream_) {
        throw std::runtime_error("Could not open diagnostics output in " + directory_.string());
    }

    diagnostics_stream_
        << "step,time,n,total_mass,kinetic_energy,potential_energy,total_energy,"
        << "momentum_x,momentum_y,center_of_mass_x,center_of_mass_y,angular_momentum\n";
}

void SnapshotWriter::write_metadata(const SimulationConfig& config, std::size_t particle_count) {
    std::ofstream metadata(directory_ / "metadata.json", std::ios::trunc);
    if (!metadata) {
        throw std::runtime_error("Could not write metadata output in " + directory_.string());
    }

    metadata << "{\n";
    metadata << "  \"name\": \"" << escaped_json(config.name) << "\",\n";
    metadata << "  \"solver\": \"" << escaped_json(config.solver) << "\",\n";
    metadata << "  \"particle_count\": " << particle_count << ",\n";
    metadata << "  \"steps\": " << config.steps << ",\n";
    metadata << "  \"dt\": " << config.dt << ",\n";
    metadata << "  \"snapshot_every\": " << config.snapshot_every << ",\n";
    metadata << "  \"gravitational_constant\": " << config.physics.gravitational_constant << ",\n";
    metadata << "  \"softening\": " << config.physics.softening << ",\n";
    metadata << "  \"tree_theta\": " << config.tree_theta << ",\n";
    metadata << "  \"tree_leaf_capacity\": " << config.tree_leaf_capacity << ",\n";
    metadata << "  \"fmm_expansion_order\": " << config.fmm_expansion_order << "\n";
    metadata << "}\n";
}

void SnapshotWriter::write_snapshot(int step, double time, const std::vector<Particle>& particles) {
    std::ofstream output(directory_ / snapshot_filename(step), std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not write snapshot output in " + directory_.string());
    }

    output << std::setprecision(17);
    output << "# time=" << time << "\n";
    output << "id,group_id,mass,x,y,vx,vy,ax,ay\n";
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto& particle = particles[i];
        output << i << ','
               << particle.group_id << ','
               << particle.mass << ','
               << particle.position.x << ','
               << particle.position.y << ','
               << particle.velocity.x << ','
               << particle.velocity.y << ','
               << particle.acceleration.x << ','
               << particle.acceleration.y << '\n';
    }
}

void SnapshotWriter::write_diagnostics(
    int step,
    double time,
    const Diagnostics& diagnostics,
    std::size_t particle_count
) {
    diagnostics_stream_ << std::setprecision(17)
                        << step << ','
                        << time << ','
                        << particle_count << ','
                        << diagnostics.total_mass << ','
                        << diagnostics.kinetic_energy << ','
                        << diagnostics.potential_energy << ','
                        << diagnostics.total_energy << ','
                        << diagnostics.momentum.x << ','
                        << diagnostics.momentum.y << ','
                        << diagnostics.center_of_mass.x << ','
                        << diagnostics.center_of_mass.y << ','
                        << diagnostics.angular_momentum << '\n';
}

}  // namespace fmmgalaxy
