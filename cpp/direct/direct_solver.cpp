#include "direct/direct_solver.hpp"

#include <cmath>

namespace fmmgalaxy {

Vec2 softened_acceleration(
    const Vec2& target_position,
    const Vec2& source_position,
    double source_mass,
    const PhysicsParams& params
) {
    const Vec2 r = source_position - target_position;
    const double eps2 = params.softening * params.softening;
    const double s2 = norm_squared(r) + eps2;
    if (s2 == 0.0) {
        return {};
    }
    const double inv_r = 1.0 / std::sqrt(s2);
    const double inv_r3 = inv_r * inv_r * inv_r;
    return r * (params.gravitational_constant * source_mass * inv_r3);
}

void reset_accelerations(std::vector<Particle>& particles) {
    for (auto& particle : particles) {
        particle.acceleration = {};
    }
}

void compute_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params) {
    reset_accelerations(particles);

    for (std::size_t i = 0; i < particles.size(); ++i) {
        for (std::size_t j = i + 1; j < particles.size(); ++j) {
            const Vec2 delta = particles[j].position - particles[i].position;
            const double eps2 = params.softening * params.softening;
            const double s2 = norm_squared(delta) + eps2;
            if (s2 == 0.0) {
                continue;
            }
            const double inv_r = 1.0 / std::sqrt(s2);
            const double inv_r3 = inv_r * inv_r * inv_r;
            const double scale = params.gravitational_constant * inv_r3;

            particles[i].acceleration += delta * (scale * particles[j].mass);
            particles[j].acceleration -= delta * (scale * particles[i].mass);
        }
    }
}

}  // namespace fmmgalaxy
