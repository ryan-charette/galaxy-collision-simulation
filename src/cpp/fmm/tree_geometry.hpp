#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <vector>

namespace fmmgalaxy {

/// Root cube enclosing the particle domain.
struct TreeRootCube {
    /// Cube center.
    Vec2 center{};
    /// Cube half-width.
    double half_width{1.0};
};

/// Compute a cubic root cell that encloses all particles with a small softening-aware margin.
TreeRootCube root_cube_for_particles(
    const std::vector<Particle>& particles,
    const PhysicsParams& params
);

/// Return the octree child index containing `position` relative to `center`.
int child_index_for_position(const Vec2& center, const Vec2& position);

/// Return the center of a child cube.
Vec2 child_center(const Vec2& parent_center, double child_half_width, int child_index);

}  // namespace fmmgalaxy
