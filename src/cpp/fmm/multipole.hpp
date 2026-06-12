#pragma once

#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <array>

namespace fmmgalaxy {

/// Fixed storage for Cartesian multipole/local coefficients up to the implemented order.
using MomentArray = std::array<double, 35>;

/// Multipole moments accumulated about a source cell center of mass.
struct CartesianMoments {
    /// Packed Cartesian moment coefficients.
    MomentArray values{};
};

/// Local expansion accumulated for one target cell.
struct LocalExpansion {
    /// Expansion center.
    Vec2 center{};
    /// Target-cell radius used for diagnostics/export.
    double radius{1.0};
    /// Expansion order represented in the coefficient arrays.
    int order{0};
    /// Coefficients for the x acceleration component.
    MomentArray ax{};
    /// Coefficients for the y acceleration component.
    MomentArray ay{};
    /// Coefficients for the z acceleration component.
    MomentArray az{};
};

/// Return zero-initialized multipole coefficients.
CartesianMoments zero_multipole_moments();
/// Return a zero-initialized local expansion at a target cell.
LocalExpansion zero_local_expansion(const Vec2& center, double radius, int expansion_order);
/// Add one source particle to a cell multipole expansion.
void add_multipole_point(CartesianMoments& moments, const Vec2& offset, double mass);

/// Shift and add a child cell's multipole moments into a parent cell.
void add_multipole_shifted_child(
    CartesianMoments& parent,
    const CartesianMoments& child,
    const Vec2& child_offset,
    double child_mass
);

/// Evaluate source-cell multipole acceleration at a target position.
Vec2 multipole_acceleration(
    const Vec2& target_position,
    const Vec2& source_center_of_mass,
    double source_mass,
    const CartesianMoments& source_moments,
    const PhysicsParams& params,
    int expansion_order
);

/// Add one far source cell's contribution to a target cell local expansion.
void add_multipole_to_local(
    LocalExpansion& target,
    const Vec2& source_center_of_mass,
    double source_mass,
    const CartesianMoments& source_moments,
    const PhysicsParams& params
);

/// Translate a parent local expansion into a child local expansion.
void add_local_to_local(LocalExpansion& target, const LocalExpansion& source);
/// Evaluate a local expansion at a particle position.
Vec2 evaluate_local_acceleration(const LocalExpansion& local, const Vec2& target_position);
/// Normalize requested expansion order to an implemented order.
int normalize_expansion_order(int expansion_order);

}  // namespace fmmgalaxy
