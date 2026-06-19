#include "core/diagnostics.hpp"
#include "core/vector2.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"
#include "mpi/distributed_solver.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

std::vector<fmmgalaxy::Particle> random_particles(std::size_t count) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<fmmgalaxy::Particle> particles(count);
    for (auto& particle : particles) {
        particle.position = {uniform(rng), uniform(rng), uniform(rng)};
        particle.mass = 1.0 / static_cast<double>(particles.size());
    }
    return particles;
}

}  // namespace

TEST_CASE("Approximate solvers stay close to direct summation", "[tree][fmm]") {
    using fmmgalaxy::Vec2;

    std::vector<fmmgalaxy::Particle> direct_particles = random_particles(80);
    auto tree_particles = direct_particles;
    auto fmm_particles = direct_particles;

    fmmgalaxy::PhysicsParams softened;
    softened.softening = 0.03;
    fmmgalaxy::compute_direct_accelerations(direct_particles, softened);
    fmmgalaxy::compute_tree_accelerations(tree_particles, softened, 0.25, 4);

    fmmgalaxy::FmmOptions fmm_options;
    fmm_options.theta = 0.35;
    fmm_options.leaf_capacity = 4;
    fmm_options.expansion_order = 4;
    fmmgalaxy::compute_fmm_accelerations(fmm_particles, softened, fmm_options);

    double relative_error_sum = 0.0;
    double fmm_relative_error_sum = 0.0;
    for (std::size_t i = 0; i < direct_particles.size(); ++i) {
        const Vec2 diff = tree_particles[i].acceleration - direct_particles[i].acceleration;
        const Vec2 fmm_diff = fmm_particles[i].acceleration - direct_particles[i].acceleration;
        const double denom = std::max(fmmgalaxy::norm(direct_particles[i].acceleration), 1.0e-12);
        relative_error_sum += fmmgalaxy::norm(diff) / denom;
        fmm_relative_error_sum += fmmgalaxy::norm(fmm_diff) / denom;
    }

    const double mean_relative_error = relative_error_sum / static_cast<double>(direct_particles.size());
    const double fmm_mean_relative_error =
        fmm_relative_error_sum / static_cast<double>(direct_particles.size());
    CHECK(mean_relative_error < 0.08);
    CHECK(fmm_mean_relative_error < 0.25);

    const auto serial_owned = fmmgalaxy::ownership_for_rank(direct_particles.size(), 0, 1);
    CHECK(serial_owned.begin == 0);
    CHECK(serial_owned.end == direct_particles.size());

    const auto diagnostics = fmmgalaxy::compute_diagnostics(direct_particles, softened);
    CHECK(diagnostics.total_mass > 0.0);
    CHECK(std::isfinite(diagnostics.total_energy));
}
