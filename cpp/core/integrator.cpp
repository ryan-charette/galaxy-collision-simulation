#include "core/integrator.hpp"

namespace fmmgalaxy {

void kick(std::vector<Particle>& particles, double dt) {
    for (auto& particle : particles) {
        particle.velocity += particle.acceleration * dt;
    }
}

void drift(std::vector<Particle>& particles, double dt) {
    for (auto& particle : particles) {
        particle.position += particle.velocity * dt;
    }
}

void leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const AccelerationFunction& compute_accelerations
) {
    kick(particles, 0.5 * dt);
    drift(particles, dt);
    compute_accelerations(particles);
    kick(particles, 0.5 * dt);
}

}  // namespace fmmgalaxy
