#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace fmmgalaxy {

struct CudaTreeOptions {
    double theta{0.6};
    std::size_t leaf_capacity{16};
    int max_depth{32};
    int expansion_order{4};
};

bool cuda_solver_available();
std::string cuda_device_name();
void compute_cuda_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params);
void cuda_direct_leapfrog_step(std::vector<Particle>& particles, double dt, const PhysicsParams& params);
void compute_cuda_tree_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);
void compute_cuda_fmm_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);
void cuda_tree_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);
void cuda_fmm_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);

}  // namespace fmmgalaxy
