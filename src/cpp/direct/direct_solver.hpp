#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <cstddef>
#include <vector>

namespace fmmgalaxy {

/// Return the softened Newtonian acceleration on one target from one source mass.
Vec2 softened_acceleration(
    const Vec2& target_position,
    const Vec2& source_position,
    double source_mass,
    const PhysicsParams& params
);

/// Set all particle accelerations to zero.
void reset_accelerations(std::vector<Particle>& particles);
/// Set accelerations to zero for the half-open particle range `[begin, end)`.
void reset_accelerations(std::vector<Particle>& particles, std::size_t begin, std::size_t end);

/// Compute all-pairs `O(N^2)` accelerations for every particle.
void compute_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params);

/// @brief Compute direct accelerations for a subset of target particles.
///
/// Source particles are read from the full vector, but only targets in `[begin, end)` are
/// updated. This is used by MPI rank ownership and validation paths.
void compute_direct_accelerations_for_targets(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    std::size_t begin,
    std::size_t end
);

}  // namespace fmmgalaxy
