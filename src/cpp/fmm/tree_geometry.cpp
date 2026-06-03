#include "fmm/tree_geometry.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>

namespace fmmgalaxy {

TreeRootCube root_cube_for_particles(
    const std::vector<Particle>& particles,
    const PhysicsParams& params
) {
    Vec2 min_position{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec2 max_position{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };

    for (const auto& particle : particles) {
        min_position.x = std::min(min_position.x, particle.position.x);
        min_position.y = std::min(min_position.y, particle.position.y);
        min_position.z = std::min(min_position.z, particle.position.z);
        max_position.x = std::max(max_position.x, particle.position.x);
        max_position.y = std::max(max_position.y, particle.position.y);
        max_position.z = std::max(max_position.z, particle.position.z);
    }

    TreeRootCube root;
    root.center = (min_position + max_position) * 0.5;
    root.half_width = 0.5 * std::max({
        max_position.x - min_position.x,
        max_position.y - min_position.y,
        max_position.z - min_position.z,
    });
    root.half_width = std::max(root.half_width, params.softening + 1.0e-6);
    root.half_width *= 1.0001;
    return root;
}

int child_index_for_position(const Vec2& center, const Vec2& position) {
    const int east = position.x >= center.x ? 1 : 0;
    const int north = position.y >= center.y ? 1 : 0;
    const int up = position.z >= center.z ? 1 : 0;
    return east + 2 * north + 4 * up;
}

Vec2 child_center(const Vec2& parent_center, double child_half_width, int child_index) {
    const double x_sign = (child_index & 1) ? 1.0 : -1.0;
    const double y_sign = (child_index & 2) ? 1.0 : -1.0;
    const double z_sign = (child_index & 4) ? 1.0 : -1.0;
    return {
        parent_center.x + x_sign * child_half_width,
        parent_center.y + y_sign * child_half_width,
        parent_center.z + z_sign * child_half_width,
    };
}

}  // namespace fmmgalaxy
