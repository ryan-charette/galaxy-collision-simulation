#include "cuda/cuda_solver.hpp"

#include "build_config.hpp"
#include "core/integrator.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"

#if !FMM_GALAXY_HAVE_CUDA

namespace fmmgalaxy {

bool cuda_solver_available() {
    return false;
}

void compute_cuda_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params) {
    compute_direct_accelerations(particles, params);
}

void cuda_direct_leapfrog_step(std::vector<Particle>& particles, double dt, const PhysicsParams& params) {
    auto compute = [&params](std::vector<Particle>& state) {
        compute_direct_accelerations(state, params);
    };
    leapfrog_step(particles, dt, compute);
}

void compute_cuda_tree_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    compute_tree_accelerations(
        particles,
        params,
        options.theta,
        options.leaf_capacity,
        options.expansion_order
    );
}

void compute_cuda_fmm_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    FmmOptions fmm_options;
    fmm_options.theta = options.theta;
    fmm_options.leaf_capacity = options.leaf_capacity;
    fmm_options.max_depth = options.max_depth;
    fmm_options.expansion_order = options.expansion_order;
    compute_fmm_accelerations(particles, params, fmm_options);
}

void cuda_tree_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    auto compute = [&params, options](std::vector<Particle>& state) {
        compute_cuda_tree_accelerations(state, params, options);
    };
    leapfrog_step(particles, dt, compute);
}

void cuda_fmm_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    auto compute = [&params, options](std::vector<Particle>& state) {
        compute_cuda_fmm_accelerations(state, params, options);
    };
    leapfrog_step(particles, dt, compute);
}

}  // namespace fmmgalaxy

#endif
