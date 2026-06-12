#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace fmmgalaxy {

/// CUDA tree/FMM evaluation options mirroring the CPU tree controls.
struct CudaTreeOptions {
    /// Opening angle used for tree/FMM traversal.
    double theta{0.6};
    /// Maximum particles per tree leaf.
    std::size_t leaf_capacity{16};
    /// Maximum tree depth.
    int max_depth{32};
    /// Cartesian expansion order for GPU tree/FMM kernels.
    int expansion_order{4};
};

/// Return whether CUDA solver kernels are available in this build/runtime.
bool cuda_solver_available();
/// Return the active CUDA device name, or an empty string when unavailable.
std::string cuda_device_name();
/// Compute all-pairs accelerations with the CUDA direct kernel or CPU fallback.
void compute_cuda_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params);
/// Advance one leapfrog step using CUDA direct accelerations or CPU fallback.
void cuda_direct_leapfrog_step(std::vector<Particle>& particles, double dt, const PhysicsParams& params);
/// Compute treecode accelerations using CUDA evaluation when available.
void compute_cuda_tree_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);
/// Compute FMM accelerations using CUDA evaluation when available.
void compute_cuda_fmm_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);
/// Advance one leapfrog step using the CUDA tree path or CPU fallback.
void cuda_tree_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);
/// Advance one leapfrog step using the CUDA FMM path or CPU fallback.
void cuda_fmm_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options = {}
);

}  // namespace fmmgalaxy
