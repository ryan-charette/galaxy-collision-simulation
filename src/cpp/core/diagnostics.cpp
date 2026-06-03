#include "core/diagnostics.hpp"

#include <cmath>

namespace fmmgalaxy {

Diagnostics compute_diagnostics(const std::vector<Particle>& particles, const PhysicsParams& params) {
    Diagnostics diagnostics;

    for (const auto& particle : particles) {
        diagnostics.total_mass += particle.mass;
        diagnostics.kinetic_energy += 0.5 * particle.mass * norm_squared(particle.velocity);
        diagnostics.momentum += particle.velocity * particle.mass;
        diagnostics.center_of_mass += particle.position * particle.mass;
        diagnostics.angular_momentum += particle.mass * cross(particle.position, particle.velocity);
    }

    if (diagnostics.total_mass > 0.0) {
        diagnostics.center_of_mass /= diagnostics.total_mass;
    }

    const double eps2 = params.softening * params.softening;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        for (std::size_t j = i + 1; j < particles.size(); ++j) {
            const Vec2 delta = particles[j].position - particles[i].position;
            const double softened_distance = std::sqrt(norm_squared(delta) + eps2);
            if (softened_distance == 0.0) {
                continue;
            }
            diagnostics.potential_energy -= params.gravitational_constant * particles[i].mass *
                                             particles[j].mass / softened_distance;
        }
    }

    diagnostics.total_energy = diagnostics.kinetic_energy + diagnostics.potential_energy;
    return diagnostics;
}

}  // namespace fmmgalaxy
