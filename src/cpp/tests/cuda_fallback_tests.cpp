#include "core/integrator.hpp"
#include "core/vector2.hpp"
#include "cuda/cuda_solver.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

TEST_CASE("CUDA fallback paths match CPU solver behavior", "[cuda][fallback]") {
    using fmmgalaxy::Vec2;

    std::vector<fmmgalaxy::Particle> direct_particles = random_particles(80);
    const auto initial_particles = direct_particles;
    auto tree_particles = direct_particles;
    auto fmm_particles = direct_particles;
    auto cuda_particles = direct_particles;
    auto cuda_tree_particles = direct_particles;
    auto cuda_fmm_particles = direct_particles;

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
    fmmgalaxy::CudaTreeOptions cuda_tree_options;
    cuda_tree_options.theta = 0.25;
    cuda_tree_options.leaf_capacity = 4;
    cuda_tree_options.expansion_order = 4;
    fmmgalaxy::CudaTreeOptions cuda_fmm_options;
    cuda_fmm_options.theta = 0.35;
    cuda_fmm_options.leaf_capacity = 4;
    cuda_fmm_options.expansion_order = 4;
    fmmgalaxy::compute_cuda_tree_accelerations(cuda_tree_particles, softened, cuda_tree_options);
    fmmgalaxy::compute_cuda_fmm_accelerations(cuda_fmm_particles, softened, cuda_fmm_options);

    double cuda_relative_error_sum = 0.0;
    double cuda_tree_relative_error_sum = 0.0;
    double cuda_fmm_relative_error_sum = 0.0;
    for (std::size_t i = 0; i < direct_particles.size(); ++i) {
        const Vec2 cuda_diff = cuda_particles[i].acceleration - direct_particles[i].acceleration;
        const Vec2 cuda_tree_diff = cuda_tree_particles[i].acceleration - tree_particles[i].acceleration;
        const Vec2 cuda_fmm_diff = cuda_fmm_particles[i].acceleration - fmm_particles[i].acceleration;
        const double denom = std::max(fmmgalaxy::norm(direct_particles[i].acceleration), 1.0e-12);
        const double tree_denom = std::max(fmmgalaxy::norm(tree_particles[i].acceleration), 1.0e-12);
        const double fmm_denom = std::max(fmmgalaxy::norm(fmm_particles[i].acceleration), 1.0e-12);
        cuda_relative_error_sum += fmmgalaxy::norm(cuda_diff) / denom;
        cuda_tree_relative_error_sum += fmmgalaxy::norm(cuda_tree_diff) / tree_denom;
        cuda_fmm_relative_error_sum += fmmgalaxy::norm(cuda_fmm_diff) / fmm_denom;
    }

    const double cuda_mean_relative_error =
        cuda_relative_error_sum / static_cast<double>(direct_particles.size());
    const double cuda_tree_mean_relative_error =
        cuda_tree_relative_error_sum / static_cast<double>(direct_particles.size());
    const double cuda_fmm_mean_relative_error =
        cuda_fmm_relative_error_sum / static_cast<double>(direct_particles.size());
    CHECK(cuda_mean_relative_error < 1.0e-10);
    CHECK(cuda_tree_mean_relative_error < 1.0e-8);
    CHECK(cuda_fmm_mean_relative_error < 1.0e-8);

    auto tree_step_particles = initial_particles;
    auto cuda_tree_step_particles = initial_particles;
    fmmgalaxy::compute_tree_accelerations(tree_step_particles, softened, 0.25, 4, 4);
    fmmgalaxy::compute_cuda_tree_accelerations(cuda_tree_step_particles, softened, cuda_tree_options);
    auto compute_tree_step = [&softened](std::vector<fmmgalaxy::Particle>& state) {
        fmmgalaxy::compute_tree_accelerations(state, softened, 0.25, 4, 4);
    };
    fmmgalaxy::leapfrog_step(tree_step_particles, 0.01, compute_tree_step);
    fmmgalaxy::cuda_tree_leapfrog_step(cuda_tree_step_particles, 0.01, softened, cuda_tree_options);

    auto fmm_step_particles = initial_particles;
    auto cuda_fmm_step_particles = initial_particles;
    fmmgalaxy::compute_fmm_accelerations(fmm_step_particles, softened, fmm_options);
    fmmgalaxy::compute_cuda_fmm_accelerations(cuda_fmm_step_particles, softened, cuda_fmm_options);
    auto compute_fmm_step = [&softened, fmm_options](std::vector<fmmgalaxy::Particle>& state) {
        fmmgalaxy::compute_fmm_accelerations(state, softened, fmm_options);
    };
    fmmgalaxy::leapfrog_step(fmm_step_particles, 0.01, compute_fmm_step);
    fmmgalaxy::cuda_fmm_leapfrog_step(cuda_fmm_step_particles, 0.01, softened, cuda_fmm_options);

    double cuda_tree_step_error = 0.0;
    double cuda_fmm_step_error = 0.0;
    for (std::size_t i = 0; i < initial_particles.size(); ++i) {
        cuda_tree_step_error = std::max(
            cuda_tree_step_error,
            fmmgalaxy::norm(cuda_tree_step_particles[i].position - tree_step_particles[i].position) +
                fmmgalaxy::norm(cuda_tree_step_particles[i].velocity - tree_step_particles[i].velocity) +
                fmmgalaxy::norm(
                    cuda_tree_step_particles[i].acceleration - tree_step_particles[i].acceleration
                )
        );
        cuda_fmm_step_error = std::max(
            cuda_fmm_step_error,
            fmmgalaxy::norm(cuda_fmm_step_particles[i].position - fmm_step_particles[i].position) +
                fmmgalaxy::norm(cuda_fmm_step_particles[i].velocity - fmm_step_particles[i].velocity) +
                fmmgalaxy::norm(
                    cuda_fmm_step_particles[i].acceleration - fmm_step_particles[i].acceleration
                )
        );
    }
    CHECK(cuda_tree_step_error < 1.0e-7);
    CHECK(cuda_fmm_step_error < 1.0e-7);
}
