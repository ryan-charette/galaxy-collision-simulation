#pragma once

#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <array>

namespace fmmgalaxy {

using MomentArray = std::array<double, 35>;

struct CartesianMoments {
    MomentArray values{};
};

struct LocalExpansion {
    Vec2 center{};
    double radius{1.0};
    int order{0};
    MomentArray ax{};
    MomentArray ay{};
    MomentArray az{};
};

CartesianMoments zero_multipole_moments();
LocalExpansion zero_local_expansion(const Vec2& center, double radius, int expansion_order);
void add_multipole_point(CartesianMoments& moments, const Vec2& offset, double mass);
void add_multipole_shifted_child(
    CartesianMoments& parent,
    const CartesianMoments& child,
    const Vec2& child_offset,
    double child_mass
);

Vec2 multipole_acceleration(
    const Vec2& target_position,
    const Vec2& source_center_of_mass,
    double source_mass,
    const CartesianMoments& source_moments,
    const PhysicsParams& params,
    int expansion_order
);

void add_multipole_to_local(
    LocalExpansion& target,
    const Vec2& source_center_of_mass,
    double source_mass,
    const CartesianMoments& source_moments,
    const PhysicsParams& params
);

void add_local_to_local(LocalExpansion& target, const LocalExpansion& source);
Vec2 evaluate_local_acceleration(const LocalExpansion& local, const Vec2& target_position);
int normalize_expansion_order(int expansion_order);

}  // namespace fmmgalaxy
