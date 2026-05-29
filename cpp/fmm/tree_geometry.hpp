#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <vector>

namespace fmmgalaxy {

struct TreeRootCube {
    Vec2 center{};
    double half_width{1.0};
};

TreeRootCube root_cube_for_particles(
    const std::vector<Particle>& particles,
    const PhysicsParams& params
);

int child_index_for_position(const Vec2& center, const Vec2& position);

Vec2 child_center(const Vec2& parent_center, double child_half_width, int child_index);

}  // namespace fmmgalaxy
