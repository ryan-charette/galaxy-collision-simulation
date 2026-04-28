#include "core/simulation_info.hpp"
#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/vector2.hpp"
#include "cuda/cuda_solver.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"
#include "io/snapshot_writer.hpp"
#include "mpi/distributed_solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

bool near(double a, double b, double tolerance) {
    return std::abs(a - b) <= tolerance;
}

}  // namespace

int main() {
    using fmmgalaxy::Vec2;
    int failures = 0;

    Vec2 a{1.0, 2.0};
    Vec2 b{3.0, 4.0, 5.0};
    Vec2 c = a + b;

    failures += !require(c.x == 4.0, "Vec2 x addition");
    failures += !require(c.y == 6.0, "Vec2 y addition");
    failures += !require(c.z == 5.0, "Vec3 z addition");
    failures += !require(near(fmmgalaxy::dot(a, b), 11.0, 1.0e-12), "Vec3 dot product");
    failures += !require(near(fmmgalaxy::cross({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}).z, 1.0, 1.0e-12), "Vec3 cross product");
    failures += !require(!fmmgalaxy::build_summary().empty(), "build summary is non-empty");

    fmmgalaxy::PhysicsParams physics;
    physics.gravitational_constant = 1.0;
    physics.softening = 0.0;

    std::vector<fmmgalaxy::Particle> two_body(2);
    two_body[0].position = {-0.5, 0.0, 0.2};
    two_body[0].mass = 2.0;
    two_body[1].position = {0.5, 0.0, -0.2};
    two_body[1].mass = 3.0;
    fmmgalaxy::compute_direct_accelerations(two_body, physics);
    failures += !require(two_body[0].acceleration.x > 0.0, "direct acceleration attracts particle 0");
    failures += !require(two_body[1].acceleration.x < 0.0, "direct acceleration attracts particle 1");
    const double net_force_x =
        two_body[0].mass * two_body[0].acceleration.x +
        two_body[1].mass * two_body[1].acceleration.x;
    failures += !require(near(net_force_x, 0.0, 1.0e-12), "pairwise direct force is symmetric");

    two_body[0].velocity = {0.0, 0.5};
    two_body[1].velocity = {0.0, -1.0 / 3.0};
    auto direct_acceleration = [&physics](std::vector<fmmgalaxy::Particle>& particles) {
        fmmgalaxy::compute_direct_accelerations(particles, physics);
    };
    fmmgalaxy::leapfrog_step(two_body, 0.001, direct_acceleration);
    failures += !require(std::isfinite(two_body[0].position.x), "leapfrog position stays finite");

    fmmgalaxy::SimulationConfig config = fmmgalaxy::default_config();
    auto generated = fmmgalaxy::generate_galaxies(config.galaxies, config.physics, config.seed);
    failures += !require(generated.size() == config.n_particles, "default config generates expected particle count");

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<fmmgalaxy::Particle> direct_particles(80);
    for (auto& particle : direct_particles) {
        particle.position = {uniform(rng), uniform(rng), uniform(rng)};
        particle.mass = 1.0 / static_cast<double>(direct_particles.size());
    }
    auto tree_particles = direct_particles;
    auto fmm_particles = direct_particles;
    auto cuda_particles = direct_particles;
    fmmgalaxy::PhysicsParams softened;
    softened.softening = 0.03;
    fmmgalaxy::compute_direct_accelerations(direct_particles, softened);
    fmmgalaxy::compute_tree_accelerations(tree_particles, softened, 0.25, 4);
    fmmgalaxy::FmmOptions fmm_options;
    fmm_options.theta = 0.35;
    fmm_options.leaf_capacity = 4;
    fmm_options.expansion_order = 4;
    fmmgalaxy::compute_fmm_accelerations(fmm_particles, softened, fmm_options);
    fmmgalaxy::compute_cuda_direct_accelerations(cuda_particles, softened);

    double relative_error_sum = 0.0;
    double fmm_relative_error_sum = 0.0;
    double cuda_relative_error_sum = 0.0;
    for (std::size_t i = 0; i < direct_particles.size(); ++i) {
        const Vec2 diff = tree_particles[i].acceleration - direct_particles[i].acceleration;
        const Vec2 fmm_diff = fmm_particles[i].acceleration - direct_particles[i].acceleration;
        const Vec2 cuda_diff = cuda_particles[i].acceleration - direct_particles[i].acceleration;
        const double denom = std::max(fmmgalaxy::norm(direct_particles[i].acceleration), 1.0e-12);
        relative_error_sum += fmmgalaxy::norm(diff) / denom;
        fmm_relative_error_sum += fmmgalaxy::norm(fmm_diff) / denom;
        cuda_relative_error_sum += fmmgalaxy::norm(cuda_diff) / denom;
    }
    const double mean_relative_error = relative_error_sum / static_cast<double>(direct_particles.size());
    const double fmm_mean_relative_error =
        fmm_relative_error_sum / static_cast<double>(direct_particles.size());
    const double cuda_mean_relative_error =
        cuda_relative_error_sum / static_cast<double>(direct_particles.size());
    failures += !require(mean_relative_error < 0.08, "tree solver stays close to direct solver");
    failures += !require(fmm_mean_relative_error < 0.25, "p=4 FMM solver stays close to direct solver");
    failures += !require(cuda_mean_relative_error < 1.0e-10, "CUDA direct solver matches direct solver");

    const auto serial_owned = fmmgalaxy::ownership_for_rank(direct_particles.size(), 0, 1);
    failures += !require(serial_owned.begin == 0, "MPI serial ownership starts at zero");
    failures += !require(serial_owned.end == direct_particles.size(), "MPI serial ownership owns all particles");

    const auto diagnostics = fmmgalaxy::compute_diagnostics(direct_particles, softened);
    failures += !require(diagnostics.total_mass > 0.0, "diagnostics compute total mass");
    failures += !require(std::isfinite(diagnostics.total_energy), "diagnostics energy is finite");

    std::ofstream config_file("test_config.toml", std::ios::trunc);
    config_file << "[simulation]\nname=\"unit\"\nsolver=\"tree\"\ndim=3\nsteps=2\ndt=0.01\nsnapshot_every=1\n"
                << "[physics]\nG=1.0\nsoftening=0.02\n"
                << "[galaxy.primary]\nn_particles=4\nmass=1.0\nradius=1.0\n"
                << "position=[0.0,0.0,0.1]\nvelocity=[0.0,0.0,0.0]\norientation=0.0\ngroup_id=3\n"
                << "thickness=0.05\ninclination=0.2\n"
                << "[output]\ndirectory=\"test_output\"\nformat=\"csv\"\n";
    config_file.close();

    const auto loaded = fmmgalaxy::load_config("test_config.toml");
    failures += !require(loaded.name == "unit", "config parser reads simulation name");
    failures += !require(loaded.dim == 3, "config parser reads 3D dimension");
    failures += !require(loaded.galaxies.size() == 1, "config parser reads galaxy section");
    failures += !require(loaded.galaxies[0].group_id == 3, "config parser reads group id");
    failures += !require(near(loaded.galaxies[0].position.z, 0.1, 1.0e-12), "config parser reads z position");

    fmmgalaxy::SnapshotWriter writer(loaded);
    writer.write_metadata(loaded, generated.size());
    writer.write_snapshot(0, 0.0, generated);
    writer.write_diagnostics(0, 0.0, diagnostics, generated.size());
    failures += !require(std::filesystem::exists("test_output/snapshot_000000.csv"), "snapshot writer creates csv");
    failures += !require(std::filesystem::exists("test_output/diagnostics.csv"), "snapshot writer creates diagnostics");

    if (failures != 0) {
        std::cerr << failures << " smoke test checks failed\n";
        return 1;
    }

    std::cout << "smoke_tests passed\n";
    return 0;
}
