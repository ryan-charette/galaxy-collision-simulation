#include "io/snapshot_writer.hpp"

#include "io/json_writer.hpp"
#include "io/parquet_converter.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fmmgalaxy {

namespace {

std::string snapshot_stem(int step) {
    std::ostringstream name;
    name << "snapshot_" << std::setw(6) << std::setfill('0') << step;
    return name.str();
}

std::string snapshot_filename(int step, OutputFormat format) {
    return snapshot_stem(step) + (format == OutputFormat::Parquet ? ".parquet" : ".csv");
}

std::string acceleration_filename(int step) {
    std::ostringstream name;
    name << "accelerations_" << std::setw(6) << std::setfill('0') << step << ".csv";
    return name.str();
}

void write_csv_particle_table_file(
    const std::filesystem::path& path,
    double time,
    const std::vector<Particle>& particles,
    const char* output_name
) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Could not write " + std::string(output_name) + " output: " + path.string()
        );
    }

    output << std::setprecision(17);
    output << "# time=" << time << "\n";
    output << "id,group_id,mass,x,y,z,vx,vy,vz,ax,ay,az\n";
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto& particle = particles[i];
        output << i << ','
               << particle.group_id << ','
               << particle.mass << ','
               << particle.position.x << ','
               << particle.position.y << ','
               << particle.position.z << ','
               << particle.velocity.x << ','
               << particle.velocity.y << ','
               << particle.velocity.z << ','
               << particle.acceleration.x << ','
               << particle.acceleration.y << ','
               << particle.acceleration.z << '\n';
    }
}

}  // namespace

SnapshotWriter::SnapshotWriter(const SimulationConfig& config) : directory_(config.output.directory) {
    format_ = config.output.format;
    enabled_ = format_ != OutputFormat::None;
    acceleration_dump_ = config.output.acceleration_dump;
    std::filesystem::create_directories(directory_);
    if (!enabled_) {
        return;
    }
    python_commands.emplace_back("python");
    python_commands.emplace_back("python3");
    python_commands.emplace_back("py -3");

    diagnostics_stream_.open(directory_ / "diagnostics.csv", std::ios::trunc);
    if (!diagnostics_stream_) {
        throw std::runtime_error("Could not open diagnostics output in " + directory_.string());
    }

    diagnostics_stream_
        << "step,time,n,total_mass,kinetic_energy,potential_energy,total_energy,"
        << "momentum_x,momentum_y,momentum_z,"
        << "center_of_mass_x,center_of_mass_y,center_of_mass_z,"
        << "angular_momentum_x,angular_momentum_y,angular_momentum_z\n";
}

void SnapshotWriter::write_metadata(
    const SimulationConfig& config,
    std::size_t particle_count,
    const RunProvenance& provenance
) {
    std::ofstream metadata(directory_ / "metadata.json", std::ios::trunc);
    if (!metadata) {
        throw std::runtime_error("Could not write metadata output in " + directory_.string());
    }

    metadata << "{\n";
    metadata << "  \"name\": ";
    write_json_string(metadata, config.name);
    metadata << ",\n";
    metadata << "  \"solver\": ";
    write_json_string(metadata, config.solver);
    metadata << ",\n";
    metadata << "  \"particle_count\": " << particle_count << ",\n";
    metadata << "  \"steps\": " << config.steps << ",\n";
    metadata << "  \"dt\": " << config.dt << ",\n";
    metadata << "  \"snapshot_every\": " << config.snapshot_every << ",\n";
    metadata << "  \"output_format\": \"" << output_format_name(config.output.format) << "\",\n";
    metadata << "  \"acceleration_dump\": " << json_bool(config.output.acceleration_dump) << ",\n";
    metadata << "  \"dim\": " << config.dim << ",\n";
    metadata << "  \"gravitational_constant\": " << config.physics.gravitational_constant << ",\n";
    metadata << "  \"softening\": " << config.physics.softening << ",\n";
    metadata << "  \"tree_theta\": " << config.tree_theta << ",\n";
    metadata << "  \"tree_leaf_capacity\": " << config.tree_leaf_capacity << ",\n";
    metadata << "  \"fmm_expansion_order\": " << config.fmm_expansion_order << ",\n";
    metadata << "  \"git_commit\": ";
    write_json_string(metadata, provenance.git_commit);
    metadata << ",\n";
    metadata << "  \"git_branch\": ";
    write_json_string(metadata, provenance.git_branch);
    metadata << ",\n";
    metadata << "  \"git_dirty\": " << json_bool(provenance.git_dirty) << ",\n";
    metadata << "  \"build_type\": ";
    write_json_string(metadata, provenance.build_type);
    metadata << ",\n";
    metadata << "  \"compiler\": ";
    write_json_string(metadata, provenance.compiler);
    metadata << ",\n";
    metadata << "  \"compiler_version\": ";
    write_json_string(metadata, provenance.compiler_version);
    metadata << ",\n";
    metadata << "  \"cmake_options\": {\n";
    metadata << "    \"ENABLE_MPI\": " << json_bool(provenance.cmake_enable_mpi) << ",\n";
    metadata << "    \"ENABLE_CUDA\": " << json_bool(provenance.cmake_enable_cuda) << "\n";
    metadata << "  },\n";
    metadata << "  \"cuda_available\": " << json_bool(provenance.cuda_available) << ",\n";
    metadata << "  \"cuda_device_name\": ";
    write_json_string(metadata, provenance.cuda_device_name);
    metadata << ",\n";
    metadata << "  \"mpi_enabled\": " << json_bool(provenance.mpi_enabled) << ",\n";
    metadata << "  \"rank_count\": " << provenance.rank_count << ",\n";
    metadata << "  \"hostname\": ";
    write_json_string(metadata, provenance.hostname);
    metadata << ",\n";
    metadata << "  \"timestamp_utc\": ";
    write_json_string(metadata, provenance.timestamp_utc);
    metadata << ",\n";
    metadata << "  \"config_path\": ";
    write_json_string(metadata, provenance.config_path);
    metadata << ",\n";
    metadata << "  \"config_sha256\": ";
    write_json_string(metadata, provenance.config_sha256);
    metadata << "\n";
    metadata << "}\n";
}

void SnapshotWriter::write_snapshot(int step, double time, const std::vector<Particle>& particles) {
    if (!enabled_) {
        return;
    }

    const auto output_path = directory_ / snapshot_filename(step, format_);
    if (format_ == OutputFormat::Csv) {
        write_csv_particle_table_file(output_path, time, particles, "snapshot");
        return;
    }

    if (format_ == OutputFormat::Parquet) {
        const auto temp_csv_path = directory_ / (snapshot_stem(step) + ".parquet.tmp.csv");
        write_csv_particle_table_file(temp_csv_path, time, particles, "snapshot");
        const bool converted =
            ParquetConverter{}.convert_snapshot_csv(temp_csv_path, output_path, step, time);
        std::filesystem::remove(temp_csv_path);
        if (!converted) {
            throw std::runtime_error(
                "Could not convert snapshot to Parquet. Install pyarrow and ensure python is on PATH, "
                "or set FMM_GALAXY_PYTHON to the Python executable."
            );
        }
        return;
    }

    throw std::runtime_error("Unknown snapshot output format");
}

void SnapshotWriter::write_accelerations(int step, double time, const std::vector<Particle>& particles) {
    if (!acceleration_dump_) {
        return;
    }
    write_csv_particle_table_file(
        directory_ / acceleration_filename(step),
        time,
        particles,
        "acceleration"
    );
}

void SnapshotWriter::write_diagnostics(
    int step,
    double time,
    const Diagnostics& diagnostics,
    std::size_t particle_count
) {
    if (!enabled_) {
        return;
    }

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
                        << diagnostics.momentum.z << ','
                        << diagnostics.center_of_mass.x << ','
                        << diagnostics.center_of_mass.y << ','
                        << diagnostics.center_of_mass.z << ','
                        << diagnostics.angular_momentum.x << ','
                        << diagnostics.angular_momentum.y << ','
                        << diagnostics.angular_momentum.z << '\n';
}

}  // namespace fmmgalaxy
