#pragma once

#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <array>

namespace fmmgalaxy {

using Quadrupole = std::array<double, 6>;

Quadrupole zero_quadrupole();
void add_quadrupole_point(Quadrupole& quadrupole, const Vec2& offset, double mass);
void add_quadrupole_shifted_child(
    Quadrupole& parent,
    const Quadrupole& child,
    const Vec2& child_offset,
    double child_mass
);

Vec2 multipole_acceleration(
    const Vec2& target_position,
    const Vec2& source_center_of_mass,
    double source_mass,
    const Quadrupole& source_quadrupole,
    const PhysicsParams& params,
    int expansion_order
);

}  // namespace fmmgalaxy
