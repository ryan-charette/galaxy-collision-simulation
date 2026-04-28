#include "fmm/multipole.hpp"

#include "direct/direct_solver.hpp"

#include <cstddef>
#include <cmath>

namespace fmmgalaxy {

namespace {

Vec2 multiply_quadrupole(const Quadrupole& q, const Vec2& v) {
    return {
        q[0] * v.x + q[1] * v.y + q[2] * v.z,
        q[1] * v.x + q[3] * v.y + q[4] * v.z,
        q[2] * v.x + q[4] * v.y + q[5] * v.z,
    };
}

double quadratic_form(const Quadrupole& q, const Vec2& v) {
    const Vec2 qv = multiply_quadrupole(q, v);
    return dot(v, qv);
}

}  // namespace

Quadrupole zero_quadrupole() {
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

void add_quadrupole_point(Quadrupole& quadrupole, const Vec2& offset, double mass) {
    const double r2 = norm_squared(offset);
    quadrupole[0] += mass * (3.0 * offset.x * offset.x - r2);
    quadrupole[1] += mass * (3.0 * offset.x * offset.y);
    quadrupole[2] += mass * (3.0 * offset.x * offset.z);
    quadrupole[3] += mass * (3.0 * offset.y * offset.y - r2);
    quadrupole[4] += mass * (3.0 * offset.y * offset.z);
    quadrupole[5] += mass * (3.0 * offset.z * offset.z - r2);
}

void add_quadrupole_shifted_child(
    Quadrupole& parent,
    const Quadrupole& child,
    const Vec2& child_offset,
    double child_mass
) {
    for (std::size_t i = 0; i < parent.size(); ++i) {
        parent[i] += child[i];
    }
    add_quadrupole_point(parent, child_offset, child_mass);
}

Vec2 multipole_acceleration(
    const Vec2& target_position,
    const Vec2& source_center_of_mass,
    double source_mass,
    const Quadrupole& source_quadrupole,
    const PhysicsParams& params,
    int expansion_order
) {
    Vec2 acceleration = softened_acceleration(
        target_position,
        source_center_of_mass,
        source_mass,
        params
    );

    if (expansion_order < 2 || source_mass <= 0.0) {
        return acceleration;
    }

    const Vec2 delta = source_center_of_mass - target_position;
    const double s2 = norm_squared(delta) + params.softening * params.softening;
    if (s2 == 0.0) {
        return acceleration;
    }

    const double inv_r = 1.0 / std::sqrt(s2);
    const double inv_r2 = inv_r * inv_r;
    const double inv_r5 = inv_r2 * inv_r2 * inv_r;
    const double inv_r7 = inv_r5 * inv_r2;
    const Vec2 q_delta = multiply_quadrupole(source_quadrupole, delta);
    const double delta_q_delta = quadratic_form(source_quadrupole, delta);

    acceleration +=
        (q_delta * inv_r5 - delta * (2.5 * delta_q_delta * inv_r7)) *
        params.gravitational_constant;

    return acceleration;
}

}  // namespace fmmgalaxy
