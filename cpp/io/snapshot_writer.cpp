#include "io/snapshot_writer.hpp"

#include <cstdlib>
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

const char* json_bool(bool value) {
    return value ? "true" : "false";
}

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

std::string quote_command_arg(const std::string& value) {
    std::string quoted = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            quoted.push_back('\\');
        }
        quoted.push_back(ch);
    }
    quoted.push_back('"');
    return quoted;
}

void write_csv_snapshot_file(
    const std::filesystem::path& path,
    double time,
    const std::vector<Particle>& particles
) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not write snapshot output: " + path.string());
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

void write_csv_acceleration_file(
    const std::filesystem::path& path,
    double time,
    const std::vector<Particle>& particles
) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Could not write acceleration output: " + path.string());
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

bool run_parquet_converter(
    const std::filesystem::path& csv_path,
    const std::filesystem::path& parquet_path,
    int step,
    double time
) {
    std::vector<std::string> python_commands;
    if (const char* configured_python = std::getenv("FMM_GALAXY_PYTHON")) {
        python_commands.emplace_back(quote_command_arg(configured_python));
    }
    python_commands.emplace_back("python");
    python_commands.emplace_back("python3");
    python_commands.emplace_back("py -3");

    for (const auto& python : python_commands) {
        std::ostringstream command;
        command << python
                << " -m python.utils.parquet_io"
                << " --input " << quote_command_arg(csv_path.string())
                << " --output " << quote_command_arg(parquet_path.string())
                << " --step " << step
                << " --time " << std::setprecision(17) << time;
        if (std::system(command.str().c_str()) == 0) {
            return true;
        }
    }
    return false;
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
    metadata << "  \"name\": \"" << escaped_json(config.name) << "\",\n";
    metadata << "  \"solver\": \"" << escaped_json(config.solver) << "\",\n";
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
    metadata << "  \"git_commit\": \"" << escaped_json(provenance.git_commit) << "\",\n";
    metadata << "  \"git_branch\": \"" << escaped_json(provenance.git_branch) << "\",\n";
    metadata << "  \"git_dirty\": " << json_bool(provenance.git_dirty) << ",\n";
    metadata << "  \"build_type\": \"" << escaped_json(provenance.build_type) << "\",\n";
    metadata << "  \"compiler\": \"" << escaped_json(provenance.compiler) << "\",\n";
    metadata << "  \"compiler_version\": \"" << escaped_json(provenance.compiler_version) << "\",\n";
    metadata << "  \"cmake_options\": {\n";
    metadata << "    \"ENABLE_MPI\": " << json_bool(provenance.cmake_enable_mpi) << ",\n";
    metadata << "    \"ENABLE_CUDA\": " << json_bool(provenance.cmake_enable_cuda) << "\n";
    metadata << "  },\n";
    metadata << "  \"cuda_available\": " << json_bool(provenance.cuda_available) << ",\n";
    metadata << "  \"cuda_device_name\": \"" << escaped_json(provenance.cuda_device_name) << "\",\n";
    metadata << "  \"mpi_enabled\": " << json_bool(provenance.mpi_enabled) << ",\n";
    metadata << "  \"rank_count\": " << provenance.rank_count << ",\n";
    metadata << "  \"hostname\": \"" << escaped_json(provenance.hostname) << "\",\n";
    metadata << "  \"timestamp_utc\": \"" << escaped_json(provenance.timestamp_utc) << "\",\n";
    metadata << "  \"config_path\": \"" << escaped_json(provenance.config_path) << "\",\n";
    metadata << "  \"config_sha256\": \"" << escaped_json(provenance.config_sha256) << "\"\n";
    metadata << "}\n";
}
