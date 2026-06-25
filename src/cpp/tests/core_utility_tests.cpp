#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/provenance.hpp"
#include "core/simulation_info.hpp"
#include "core/simulation_runner.hpp"
#include "cuda/cuda_solver.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/multipole.hpp"
#include "fmm/quadtree.hpp"
#include "fmm/tree_geometry.hpp"
#include "io/json_writer.hpp"
#include "io/parquet_converter.hpp"
#include "io/snapshot_writer.hpp"
#include "mpi/distributed_solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include <vector>

namespace {

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::trunc);
    output << contents;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void set_environment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void unset_environment(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedEnvironmentVariable {
public:
    ScopedEnvironmentVariable(const char* name, const std::string& value) : name_(name) {
        if (const char* previous = std::getenv(name)) {
            had_previous_ = true;
            previous_ = previous;
        }
        set_environment(name_.c_str(), value);
    }

    ~ScopedEnvironmentVariable() {
        if (had_previous_) {
            set_environment(name_.c_str(), previous_);
        } else {
            unset_environment(name_.c_str());
        }
    }

private:
    std::string name_;
    bool had_previous_{false};
    std::string previous_;
};

std::filesystem::path write_fake_python_bridge(const std::filesystem::path& path) {
#ifdef _WIN32
    const auto script_path = std::filesystem::absolute(path.string() + ".cmd");
    write_text(
        script_path,
        "@echo off\n"
        "set \"out=\"\n"
        ":loop\n"
        "if \"%~1\"==\"\" goto done\n"
        "if \"%~1\"==\"--output\" (\n"
        "  shift\n"
        "  set \"out=%~1\"\n"
        ")\n"
        "shift\n"
        "goto loop\n"
        ":done\n"
        "if not \"%out%\"==\"\" echo fake parquet>\"%out%\"\n"
        "exit /b 0\n"
    );
#else
    const auto script_path = std::filesystem::absolute(path);
    write_text(
        script_path,
        "#!/bin/sh\n"
        "out=\"\"\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = \"--output\" ]; then\n"
        "    shift\n"
        "    out=\"$1\"\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "if [ -n \"$out\" ]; then\n"
        "  printf 'fake parquet\\n' > \"$out\"\n"
        "fi\n"
        "exit 0\n"
    );
    std::filesystem::permissions(
        script_path,
        std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
            std::filesystem::perms::owner_exec,
        std::filesystem::perm_options::replace
    );
#endif
    return script_path;
}

fmmgalaxy::RunProvenance test_provenance() {
    fmmgalaxy::RunProvenance provenance;
    provenance.git_commit = "test";
    provenance.git_branch = "test";
    provenance.build_type = "Debug";
    provenance.compiler = "test-compiler";
    provenance.compiler_version = "0";
    provenance.hostname = "localhost";
    provenance.timestamp_utc = "2026-01-01T00:00:00Z";
    return provenance;
}

fmmgalaxy::SimulationConfig small_runner_config(const std::filesystem::path& output_dir) {
    fmmgalaxy::SimulationConfig config = fmmgalaxy::default_config();
    config.name = "coverage-runner";
    config.solver = "direct";
    config.steps = 1;
    config.dt = 0.001;
    config.snapshot_every = 1;
    config.seed = 13;
    config.tree_theta = 0.5;
    config.tree_leaf_capacity = 1;
    config.fmm_expansion_order = 2;
    config.output.directory = output_dir;
    config.output.format = fmmgalaxy::OutputFormat::None;
    config.output.acceleration_dump = false;
    config.galaxies = {
        fmmgalaxy::GalaxyConfig{3, 1.0, 1.0, {-0.5, 0.0, 0.0}, {0.0, 0.1, 0.0}, 0.0, 0, 0.02, 0.1},
    };
    config.n_particles = 3;
    return config;
}

}  // namespace

TEST_CASE("Config parser handles synonyms, defaults, and validation errors", "[config]") {
    using Catch::Matchers::WithinAbs;
    const std::filesystem::path config_path = "core_utility_config.toml";
    write_text(
        config_path,
        "[simulation]\n"
        "name=\"coverage\"\n"
        "solver=\"DIRECT\"\n"
        "dimension=2\n"
        "seed=42\n"
        "steps=0\n"
        "dt=0.02\n"
        "snapshot_every=3\n"
        "theta=0.4\n"
        "leaf_capacity=6\n"
        "expansion_order=5\n"
        "[physics]\n"
        "gravitational_constant=2.0\n"
        "softening=0.01\n"
        "[output]\n"
        "directory=\"core_utility_output\"\n"
        "format=\"none\"\n"
        "dump_accelerations=yes\n"
        "[galaxy.primary]\n"
        "n_particles=3\n"
        "mass=3.0\n"
        "radius=2.0\n"
        "position=[1.0,2.0,9.0]\n"
        "velocity=[0.1,0.2,0.3]\n"
        "orientation=0.5\n"
        "group_id=4\n"
        "thickness=0.8\n"
        "inclination=1.2\n"
    );

    const auto config = fmmgalaxy::load_config(config_path);
    CHECK(config.name == "coverage");
    CHECK(config.solver == "direct");
    CHECK(config.dim == 2);
    CHECK(config.snapshot_every == 3);
    CHECK(config.tree_leaf_capacity == 6);
    CHECK(config.fmm_expansion_order == 5);
    CHECK(config.output.format == fmmgalaxy::OutputFormat::None);
    CHECK(config.output.acceleration_dump);
    REQUIRE(config.galaxies.size() == 1);
    CHECK(config.galaxies[0].group_id == 4);
    CHECK_THAT(config.galaxies[0].position.z, WithinAbs(0.0, 1.0e-12));
    CHECK_THAT(config.galaxies[0].velocity.z, WithinAbs(0.0, 1.0e-12));
    CHECK_THAT(config.galaxies[0].thickness, WithinAbs(0.0, 1.0e-12));

    CHECK(fmmgalaxy::parse_output_format("CSV") == fmmgalaxy::OutputFormat::Csv);
    CHECK(fmmgalaxy::output_format_name(fmmgalaxy::OutputFormat::Parquet) == "parquet");
    CHECK(fmmgalaxy::output_format_name(fmmgalaxy::OutputFormat::None) == "none");
    CHECK_THROWS_AS(fmmgalaxy::parse_output_format("hdf5"), std::runtime_error);

    write_text(
        "bad_config.toml",
        "[simulation]\ndim=4\n"
        "[galaxy.primary]\n"
        "n_particles=1\n"
        "mass=1.0\n"
        "radius=1.0\n"
    );
    CHECK_THROWS_AS(fmmgalaxy::load_config("bad_config.toml"), std::runtime_error);
    CHECK_THROWS_AS(fmmgalaxy::load_config("missing_config.toml"), std::runtime_error);
}

TEST_CASE("Config parser covers defaults, 3D fallback, and validation edge cases", "[config]") {
    using Catch::Matchers::WithinAbs;
    const std::string valid_galaxy =
        "[galaxy.primary]\n"
        "n_particles=1\n"
        "mass=1.0\n"
        "radius=2.0\n";

    write_text(
        "config_defaults_only.toml",
        "# Empty galaxy list intentionally falls back to built-in defaults.\n"
        "[simulation]\n"
        "solver=\"fmm\"\n"
    );
    const auto defaults = fmmgalaxy::load_config("config_defaults_only.toml");
    CHECK(defaults.name == "smoke_test");
    CHECK(defaults.n_particles == 256);
    CHECK(defaults.galaxies.size() == 2);

    write_text(
        "config_three_dimensional.toml",
        "[simulation]\n"
        "name=\"three-dimensional\"\n"
        "dim=3\n"
        "steps=1\n"
        "dt=0.01\n"
        "snapshot_every=1\n"
        "[tree]\n"
        "tree_theta=0.25\n"
        "tree_leaf_capacity=2\n"
        "[fmm]\n"
        "fmm_expansion_order=3\n"
        "[output]\n"
        "acceleration_dump=off\n"
        "[galaxy.primary]\n"
        "n_particles=1\n"
        "mass=1.0\n"
        "radius=2.0\n"
        "position=[1.0,2.0,3.0]\n"
        "velocity=[0.1,0.2,0.3]\n"
        "thickness=0.0\n"
    );
    const auto three_dimensional = fmmgalaxy::load_config("config_three_dimensional.toml");
    REQUIRE(three_dimensional.galaxies.size() == 1);
    CHECK_THAT(three_dimensional.galaxies[0].position.z, WithinAbs(3.0, 1.0e-12));
    CHECK_THAT(three_dimensional.galaxies[0].velocity.z, WithinAbs(0.3, 1.0e-12));
    CHECK_THAT(three_dimensional.galaxies[0].thickness, WithinAbs(0.06, 1.0e-12));
    CHECK_THAT(three_dimensional.tree_theta, WithinAbs(0.25, 1.0e-12));
    CHECK(three_dimensional.tree_leaf_capacity == 2);
    CHECK(three_dimensional.fmm_expansion_order == 3);
    CHECK_FALSE(three_dimensional.output.acceleration_dump);

    const std::vector<std::pair<std::string, std::string>> invalid_configs{
        {"config_invalid_line.toml", "[simulation]\nnot_a_key_value_line\n"},
        {"config_invalid_bool.toml", "[output]\nacceleration_dump=maybe\n"},
        {"config_invalid_vector_shape.toml", valid_galaxy + "position=1.0\n"},
        {"config_invalid_vector_count.toml", valid_galaxy + "position=[1.0]\n"},
        {"config_invalid_snapshot_every.toml", "[simulation]\nsnapshot_every=0\n" + valid_galaxy},
        {"config_invalid_steps.toml", "[simulation]\nsteps=-1\n" + valid_galaxy},
        {"config_invalid_dt.toml", "[simulation]\ndt=0\n" + valid_galaxy},
        {"config_invalid_softening.toml", "[physics]\nsoftening=-0.01\n" + valid_galaxy},
    };

    for (const auto& [path, contents] : invalid_configs) {
        write_text(path, contents);
        CHECK_THROWS_AS(fmmgalaxy::load_config(path), std::runtime_error);
    }
    CHECK_THROWS_AS(fmmgalaxy::output_format_name(static_cast<fmmgalaxy::OutputFormat>(99)), std::runtime_error);
}

TEST_CASE("Integrator range overloads clamp work to owned particles", "[integrator]") {
    using Catch::Matchers::WithinAbs;
    std::vector<fmmgalaxy::Particle> particles(3);
    particles[0].velocity = {1.0, 0.0, 0.0};
    particles[1].velocity = {2.0, 0.0, 0.0};
    particles[2].velocity = {3.0, 0.0, 0.0};
    particles[1].acceleration = {4.0, 0.0, 0.0};
    particles[2].acceleration = {6.0, 0.0, 0.0};

    fmmgalaxy::kick(particles, 1, 99, 0.5);
    CHECK_THAT(particles[0].velocity.x, WithinAbs(1.0, 1.0e-12));
    CHECK_THAT(particles[1].velocity.x, WithinAbs(4.0, 1.0e-12));
    CHECK_THAT(particles[2].velocity.x, WithinAbs(6.0, 1.0e-12));

    fmmgalaxy::drift(particles, 2, 99, 0.25);
    CHECK_THAT(particles[0].position.x, WithinAbs(0.0, 1.0e-12));
    CHECK_THAT(particles[2].position.x, WithinAbs(1.5, 1.0e-12));

    bool callback_called = false;
    fmmgalaxy::leapfrog_step(particles, 0.1, [&callback_called](std::vector<fmmgalaxy::Particle>& state) {
        callback_called = true;
        for (auto& particle : state) {
            particle.acceleration = {0.0, 1.0, 0.0};
        }
    });
    CHECK(callback_called);
    CHECK_THAT(particles[0].velocity.y, WithinAbs(0.05, 1.0e-12));
}

TEST_CASE("Diagnostics and direct target ranges handle edge cases", "[diagnostics][direct]") {
    using Catch::Matchers::WithinAbs;
    fmmgalaxy::PhysicsParams physics;
    physics.gravitational_constant = 2.0;
    physics.softening = 0.0;

    std::vector<fmmgalaxy::Particle> particles(3);
    particles[0].position = {0.0, 0.0, 0.0};
    particles[0].velocity = {1.0, 0.0, 0.0};
    particles[0].mass = 2.0;
    particles[1].position = {0.0, 0.0, 0.0};
    particles[1].mass = 3.0;
    particles[2].position = {2.0, 0.0, 0.0};
    particles[2].velocity = {0.0, 1.0, 0.0};
    particles[2].mass = 1.0;

    const auto diagnostics = fmmgalaxy::compute_diagnostics(particles, physics);
    CHECK_THAT(diagnostics.total_mass, WithinAbs(6.0, 1.0e-12));
    CHECK_THAT(diagnostics.center_of_mass.x, WithinAbs(1.0 / 3.0, 1.0e-12));
    CHECK_THAT(diagnostics.kinetic_energy, WithinAbs(1.5, 1.0e-12));
    CHECK_THAT(diagnostics.potential_energy, WithinAbs(-5.0, 1.0e-12));
    CHECK_THAT(diagnostics.total_energy, WithinAbs(-3.5, 1.0e-12));
    CHECK_THAT(diagnostics.angular_momentum.z, WithinAbs(2.0, 1.0e-12));

    const auto empty_diagnostics = fmmgalaxy::compute_diagnostics({}, physics);
    CHECK_THAT(empty_diagnostics.total_mass, WithinAbs(0.0, 1.0e-12));
    CHECK_THAT(empty_diagnostics.center_of_mass.x, WithinAbs(0.0, 1.0e-12));

    for (auto& particle : particles) {
        particle.acceleration = {9.0, 9.0, 9.0};
    }
    fmmgalaxy::compute_direct_accelerations_for_targets(particles, physics, 1, 2);
    CHECK_THAT(particles[0].acceleration.x, WithinAbs(9.0, 1.0e-12));
    CHECK_THAT(particles[1].acceleration.x, WithinAbs(0.5, 1.0e-12));
    CHECK_THAT(particles[2].acceleration.x, WithinAbs(9.0, 1.0e-12));
    CHECK_THAT(
        fmmgalaxy::softened_acceleration({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0, physics).x,
        WithinAbs(0.0, 1.0e-12)
    );
}

TEST_CASE("MPI ownership helpers partition serial and distributed ranges", "[mpi]") {
    const auto serial = fmmgalaxy::ownership_for_rank(10, 0, 1);
    CHECK(serial.begin == 0);
    CHECK(serial.end == 10);

    const auto rank0 = fmmgalaxy::ownership_for_rank(10, 0, 3);
    const auto rank1 = fmmgalaxy::ownership_for_rank(10, 1, 3);
    const auto rank2 = fmmgalaxy::ownership_for_rank(10, 2, 3);
    CHECK(rank0.begin == 0);
    CHECK(rank0.end == 4);
    CHECK(rank1.begin == 4);
    CHECK(rank1.end == 7);
    CHECK(rank2.begin == 7);
    CHECK(rank2.end == 10);

    std::vector<fmmgalaxy::Particle> particles(2);
    fmmgalaxy::mpi_synchronize_particles(particles, serial);
    const auto execution = fmmgalaxy::mpi_execution();
    CHECK(execution.size >= 1);
}

TEST_CASE("Provenance hashing matches SHA-256 vectors and optional config metadata", "[provenance]") {
    write_text("sha256_empty.txt", "");
    write_text("sha256_abc.txt", "abc");
    write_text("sha256_long.txt", std::string(64, 'a'));

    CHECK(fmmgalaxy::sha256_file("sha256_empty.txt") ==
          "e3b0c44298fc1c149afbf4c8996fb924"
          "27ae41e4649b934ca495991b7852b855");
    CHECK(fmmgalaxy::sha256_file("sha256_abc.txt") ==
          "ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad");
    CHECK(fmmgalaxy::sha256_file("sha256_long.txt") ==
          "ffe054fe7ae0cb6dc65c3af9b61d5209"
          "f439851db43d0ba5997337df154668eb");
    CHECK_THROWS_AS(fmmgalaxy::sha256_file("sha256_missing.txt"), std::runtime_error);

    fmmgalaxy::MpiExecution execution;
    execution.enabled = true;
    execution.rank = 2;
    execution.size = 4;
    const auto provenance = fmmgalaxy::collect_run_provenance(nullptr, execution);
    CHECK(provenance.config_path == "builtin:default");
    CHECK(provenance.config_sha256.empty());
    CHECK(provenance.mpi_enabled);
    CHECK(provenance.rank_count == 4);
    CHECK_FALSE(provenance.timestamp_utc.empty());
}

TEST_CASE("Parquet conversion and snapshot writer use the configured Python bridge", "[io][parquet]") {
    using Catch::Matchers::WithinAbs;
    const std::filesystem::path output_dir = "core_utility_parquet_output";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);

    const auto fake_python = write_fake_python_bridge(output_dir / "fake_python_bridge");
    ScopedEnvironmentVariable python_env("FMM_GALAXY_PYTHON", fake_python.string());

    const auto csv_path = output_dir / "snapshot input.csv";
    const auto parquet_path = output_dir / "snapshot output.parquet";
    write_text(csv_path, "# time=0\nid,group_id,mass,x,y,z,vx,vy,vz,ax,ay,az\n");
    CHECK(fmmgalaxy::ParquetConverter{}.convert_snapshot_csv(csv_path, parquet_path, 12, 0.125));
    CHECK(std::filesystem::exists(parquet_path));

    fmmgalaxy::SimulationConfig config = small_runner_config(output_dir / "writer");
    config.name = "json \"escape\" test";
    config.solver = "fmm";
    config.output.format = fmmgalaxy::OutputFormat::Parquet;
    config.output.acceleration_dump = true;

    std::vector<fmmgalaxy::Particle> particles(2);
    particles[0].position = {-0.5, 0.0, 0.0};
    particles[0].velocity = {0.0, 0.1, 0.0};
    particles[0].acceleration = {1.0, 0.0, 0.0};
    particles[0].mass = 0.5;
    particles[1].position = {0.5, 0.0, 0.0};
    particles[1].velocity = {0.0, -0.1, 0.0};
    particles[1].acceleration = {-1.0, 0.0, 0.0};
    particles[1].mass = 0.5;

    fmmgalaxy::SnapshotWriter writer(config);
    writer.write_metadata(config, particles.size(), test_provenance());
    writer.write_snapshot(5, 1.25, particles);
    writer.write_accelerations(5, 1.25, particles);
    writer.write_diagnostics(5, 1.25, fmmgalaxy::compute_diagnostics(particles, config.physics), particles.size());

    CHECK(std::filesystem::exists(config.output.directory / "metadata.json"));
    CHECK(std::filesystem::exists(config.output.directory / "snapshot_000005.parquet"));
    CHECK_FALSE(std::filesystem::exists(config.output.directory / "snapshot_000005.parquet.tmp.csv"));
    CHECK(std::filesystem::exists(config.output.directory / "accelerations_000005.csv"));
    CHECK(std::filesystem::exists(config.output.directory / "diagnostics.csv"));
    CHECK(read_text(config.output.directory / "metadata.json").find("json \\\"escape\\\" test") != std::string::npos);

    std::ostringstream json;
    fmmgalaxy::write_json_string(json, "quote\"and\\slash");
    CHECK(json.str() == "\"quote\\\"and\\\\slash\"");
    CHECK(std::string(fmmgalaxy::json_bool(true)) == "true");
    CHECK(std::string(fmmgalaxy::json_bool(false)) == "false");
}

TEST_CASE("Tree geometry, multipoles, and flat exports preserve spatial structure", "[tree][fmm]") {
    using Catch::Matchers::WithinAbs;
    fmmgalaxy::PhysicsParams physics;
    physics.softening = 0.05;

    CHECK(fmmgalaxy::normalize_expansion_order(-2) == 0);
    CHECK(fmmgalaxy::normalize_expansion_order(1) == 2);
    CHECK(fmmgalaxy::normalize_expansion_order(3) == 4);
    CHECK(fmmgalaxy::normalize_expansion_order(9) == 4);

    std::vector<fmmgalaxy::Particle> particles(8);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].position = {
            (i & 1U) ? 1.0 : -1.0,
            (i & 2U) ? 1.0 : -1.0,
            (i & 4U) ? 1.0 : -1.0,
        };
        particles[i].mass = 1.0 + static_cast<double>(i);
    }

    const auto root = fmmgalaxy::root_cube_for_particles(particles, physics);
    CHECK_THAT(root.center.x, WithinAbs(0.0, 1.0e-12));
    CHECK(root.half_width > 1.0);
    CHECK(fmmgalaxy::child_index_for_position({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}) == 7);
    const auto child = fmmgalaxy::child_center({0.0, 0.0, 0.0}, 0.5, 0);
    CHECK_THAT(child.x, WithinAbs(-0.5, 1.0e-12));
    CHECK_THAT(child.y, WithinAbs(-0.5, 1.0e-12));
    CHECK_THAT(child.z, WithinAbs(-0.5, 1.0e-12));

    CHECK(fmmgalaxy::build_flat_tree({}, physics, 0.5, 1, 8, 2).nodes.empty());
    auto flat_tree = fmmgalaxy::build_flat_tree(particles, physics, 0.5, 1, 8, 2);
    REQUIRE_FALSE(flat_tree.nodes.empty());
    CHECK_FALSE(flat_tree.nodes.front().is_leaf);
    CHECK(flat_tree.particle_indices.size() == particles.size());
    const auto leaf_count = std::count_if(flat_tree.nodes.begin(), flat_tree.nodes.end(), [](const auto& node) {
        return node.is_leaf;
    });
    CHECK(leaf_count >= 8);

    auto moments = fmmgalaxy::zero_multipole_moments();
    fmmgalaxy::add_multipole_point(moments, {-0.5, 0.0, 0.0}, 1.0);
    fmmgalaxy::add_multipole_point(moments, {0.5, 0.0, 0.0}, 1.0);
    const auto monopole = fmmgalaxy::multipole_acceleration(
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        2.0,
        moments,
        physics,
        0
    );
    const auto direct_monopole =
        fmmgalaxy::softened_acceleration({0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, 2.0, physics);
    CHECK_THAT(monopole.x, WithinAbs(direct_monopole.x, 1.0e-12));
    const auto quadrupole = fmmgalaxy::multipole_acceleration(
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        2.0,
        moments,
        physics,
        4
    );
    CHECK(quadrupole.x > 0.0);

    auto parent_moments = fmmgalaxy::zero_multipole_moments();
    fmmgalaxy::add_multipole_shifted_child(parent_moments, moments, {1.0, 0.0, 0.0}, 2.0);
    CHECK(std::any_of(parent_moments.values.begin(), parent_moments.values.end(), [](double value) {
        return value != 0.0;
    }));

    auto clamped_local = fmmgalaxy::zero_local_expansion({0.0, 0.0, 0.0}, 0.0, 3);
    CHECK(clamped_local.radius > 0.0);
    CHECK(clamped_local.order == 4);

    auto local = fmmgalaxy::zero_local_expansion({0.0, 0.0, 0.0}, 1.0, 3);
    fmmgalaxy::add_multipole_to_local(local, {10.0, 0.0, 0.0}, 2.0, moments, physics);
    const auto local_acceleration = fmmgalaxy::evaluate_local_acceleration(local, {0.1, 0.2, 0.0});
    CHECK(std::isfinite(local_acceleration.x));
    CHECK(fmmgalaxy::norm(local_acceleration) > 0.0);

    auto child_local = fmmgalaxy::zero_local_expansion({0.25, 0.0, 0.0}, 0.5, 4);
    fmmgalaxy::add_local_to_local(child_local, local);
    const auto translated_acceleration =
        fmmgalaxy::evaluate_local_acceleration(child_local, {0.25, 0.1, 0.0});
    CHECK(std::isfinite(translated_acceleration.x));
    CHECK(fmmgalaxy::norm(translated_acceleration) > 0.0);

    fmmgalaxy::add_multipole_to_local(local, {1.0, 0.0, 0.0}, 0.0, moments, physics);
    CHECK_THAT(
        fmmgalaxy::multipole_acceleration({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.0, moments, physics, 4).x,
        WithinAbs(0.0, 1.0e-12)
    );
}

TEST_CASE("FMM target ranges and flat exports cover sparse leaves", "[fmm]") {
    using Catch::Matchers::WithinAbs;
    fmmgalaxy::PhysicsParams physics;
    physics.softening = 0.03;
    std::vector<fmmgalaxy::Particle> particles(12);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].position = {
            static_cast<double>(i % 4),
            static_cast<double>((i / 4) % 3),
            static_cast<double>(i % 2) * 0.5,
        };
        particles[i].mass = 1.0 / static_cast<double>(particles.size());
        particles[i].acceleration = {99.0, 99.0, 99.0};
    }

    fmmgalaxy::FmmOptions options;
    options.theta = 0.75;
    options.leaf_capacity = 1;
    options.max_depth = 5;
    options.expansion_order = 3;

    CHECK(fmmgalaxy::build_flat_fmm({}, physics, options).tree.nodes.empty());
    auto flat_fmm = fmmgalaxy::build_flat_fmm(particles, physics, options);
    CHECK_FALSE(flat_fmm.tree.nodes.empty());
    CHECK_FALSE(flat_fmm.leaves.empty());
    CHECK(flat_fmm.particle_leaf_indices.size() == particles.size());
    CHECK(flat_fmm.near_leaf_node_indices.size() >= flat_fmm.leaves.size());

    fmmgalaxy::compute_fmm_accelerations_for_targets(particles, physics, 3, 7, options);
    CHECK_THAT(particles[0].acceleration.x, WithinAbs(99.0, 1.0e-12));
    CHECK(particles[3].acceleration.x != 99.0);
    CHECK_THAT(particles[8].acceleration.x, WithinAbs(99.0, 1.0e-12));

    auto unchanged = particles;
    fmmgalaxy::compute_fmm_accelerations_for_targets(unchanged, physics, 5, 5, options);
    CHECK_THAT(unchanged[0].acceleration.x, WithinAbs(particles[0].acceleration.x, 1.0e-12));
}

TEST_CASE("Initial condition generation rejects degenerate galaxies", "[initial-conditions]") {
    std::mt19937_64 rng(11);
    fmmgalaxy::PhysicsParams physics;
    fmmgalaxy::GalaxyConfig galaxy;
    galaxy.n_particles = 0;
    CHECK(fmmgalaxy::generate_disk_galaxy(galaxy, physics, rng).empty());

    galaxy.n_particles = 4;
    galaxy.mass = 0.0;
    CHECK(fmmgalaxy::generate_disk_galaxy(galaxy, physics, rng).empty());

    galaxy.mass = 2.0;
    galaxy.radius = 0.0;
    CHECK(fmmgalaxy::generate_disk_galaxy(galaxy, physics, rng).empty());

    galaxy.radius = 1.5;
    galaxy.group_id = 7;
    auto particles = fmmgalaxy::generate_disk_galaxy(galaxy, physics, rng);
    REQUIRE(particles.size() == 4);
    CHECK(particles[0].group_id == 7);
    CHECK(fmmgalaxy::generate_galaxies({galaxy}, physics, 123).size() == 4);
}

TEST_CASE("Serial runner supports disabled snapshot output and rejects unknown solvers", "[runner]") {
    const std::filesystem::path output_dir = "core_utility_runner_output";
    std::filesystem::remove_all(output_dir);

    fmmgalaxy::SimulationConfig config = fmmgalaxy::default_config();
    config.name = "runner";
    config.solver = "direct";
    config.steps = 1;
    config.dt = 0.001;
    config.snapshot_every = 5;
    config.seed = 9;
    config.output.directory = output_dir;
    config.output.format = fmmgalaxy::OutputFormat::None;
    config.output.acceleration_dump = false;
    config.galaxies = {
        fmmgalaxy::GalaxyConfig{2, 1.0, 1.0, {-0.5, 0.0, 0.0}, {0.0, 0.1, 0.0}, 0.0, 0, 0.0, 0.0},
    };

    fmmgalaxy::MpiExecution execution;
    fmmgalaxy::run_configured_simulation(config, execution, test_provenance());
    CHECK(std::filesystem::exists(output_dir / "metadata.json"));
    CHECK_FALSE(std::filesystem::exists(output_dir / "diagnostics.csv"));
    CHECK_FALSE(std::filesystem::exists(output_dir / "snapshot_000000.csv"));

    config.solver = "not-a-solver";
    CHECK_THROWS_AS(
        fmmgalaxy::run_configured_simulation(config, execution, test_provenance()),
        std::runtime_error
    );
}

TEST_CASE("Serial runner writes CSV output and advances each solver family", "[runner]") {
    const std::filesystem::path output_root = "core_utility_serial_solver_outputs";
    std::filesystem::remove_all(output_root);

    fmmgalaxy::MpiExecution serial_execution;
    auto csv_config = small_runner_config(output_root / "direct_csv");
    csv_config.output.format = fmmgalaxy::OutputFormat::Csv;
    csv_config.output.acceleration_dump = true;
    fmmgalaxy::run_configured_simulation(csv_config, serial_execution, test_provenance());
    CHECK(std::filesystem::exists(csv_config.output.directory / "metadata.json"));
    CHECK(std::filesystem::exists(csv_config.output.directory / "snapshot_000000.csv"));
    CHECK(std::filesystem::exists(csv_config.output.directory / "snapshot_000001.csv"));
    CHECK(std::filesystem::exists(csv_config.output.directory / "accelerations_000001.csv"));
    CHECK(std::filesystem::exists(csv_config.output.directory / "diagnostics.csv"));

    const std::vector<std::string> solvers{
        "barnes-hut",
        "quadrupole-fmm",
        "cuda-direct",
        "gpu-direct",
        "cuda-barnes-hut",
        "cuda-fmm",
        "p4-fmm",
        "cartesian-fmm",
    };
    for (const auto& solver : solvers) {
        auto config = small_runner_config(output_root / solver);
        config.solver = solver;
        fmmgalaxy::run_configured_simulation(config, serial_execution, test_provenance());
        CHECK(std::filesystem::exists(config.output.directory / "metadata.json"));
    }

    auto empty_config = small_runner_config(output_root / "empty");
    empty_config.galaxies[0].n_particles = 0;
    empty_config.galaxies[0].mass = 0.0;
    CHECK_THROWS_AS(
        fmmgalaxy::run_configured_simulation(empty_config, serial_execution, test_provenance()),
        std::runtime_error
    );
}

TEST_CASE("Distributed runner dispatches owned-range solver paths without MPI runtime", "[runner][mpi]") {
    const std::filesystem::path output_root = "core_utility_distributed_solver_outputs";
    std::filesystem::remove_all(output_root);

    const std::vector<std::pair<std::string, int>> runs{
        {"direct", 0},
        {"tree", 1},
        {"fmm", 0},
        {"cuda-direct", 1},
        {"cuda-tree", 0},
        {"cuda-fmm", 1},
    };
    for (const auto& [solver, rank] : runs) {
        auto config = small_runner_config(output_root / (solver + "_rank" + std::to_string(rank)));
        config.solver = solver;
        fmmgalaxy::MpiExecution execution;
        execution.enabled = true;
        execution.rank = rank;
        execution.size = 2;
        fmmgalaxy::run_configured_simulation(config, execution, test_provenance());
        if (rank == 0) {
            CHECK(std::filesystem::exists(config.output.directory / "metadata.json"));
        }
    }
}

TEST_CASE("Serial runner dispatches CPU and CUDA fallback solver aliases", "[runner][cuda]") {
    fmmgalaxy::SimulationConfig config = fmmgalaxy::default_config();
    config.name = "dispatch";
    config.steps = 0;
    config.dt = 0.001;
    config.snapshot_every = 1;
    config.output.format = fmmgalaxy::OutputFormat::None;
    config.output.acceleration_dump = false;
    config.galaxies = {
        fmmgalaxy::GalaxyConfig{2, 1.0, 1.0, {-0.5, 0.0, 0.0}, {0.0, 0.1, 0.0}, 0.0, 0, 0.0, 0.0},
    };
    fmmgalaxy::MpiExecution execution;

    const std::vector<std::string> solvers{
        "treecode",
        "monopole-fmm",
        "cuda",
        "gpu-tree",
        "gpu-fmm",
    };
    for (const auto& solver : solvers) {
        config.solver = solver;
        config.output.directory = std::filesystem::path("core_utility_dispatch_output") / solver;
        fmmgalaxy::run_configured_simulation(config, execution, test_provenance());
        CHECK(std::filesystem::exists(config.output.directory / "metadata.json"));
    }
}
