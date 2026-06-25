#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/provenance.hpp"
#include "core/simulation_info.hpp"
#include "core/simulation_runner.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/multipole.hpp"
#include "fmm/quadtree.hpp"
#include "fmm/tree_geometry.hpp"
#include "mpi/distributed_solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::trunc);
    output << contents;
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
