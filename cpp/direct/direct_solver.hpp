#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <vector>

namespace fmmgalaxy {

Vec2 softened_acceleration(
    const Vec2& target_position,
    const Vec2& source_position,
    double source_mass,
    const PhysicsParams& params
);

void reset_accelerations(std::vector<Particle>& particles);

void compute_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params);

}  // namespace fmmgalaxy
