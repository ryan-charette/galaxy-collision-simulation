#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"

#include <vector>

namespace fmmgalaxy {

bool cuda_solver_available();
void compute_cuda_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params);
void cuda_direct_leapfrog_step(std::vector<Particle>& particles, double dt, const PhysicsParams& params);

}  // namespace fmmgalaxy
