#include "cuda/cuda_solver.hpp"

#include "build_config.hpp"
#include "core/integrator.hpp"
#include "direct/direct_solver.hpp"

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

}  // namespace fmmgalaxy

#endif
