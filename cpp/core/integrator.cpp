#include "core/integrator.hpp"

#include <algorithm>

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

void kick(std::vector<Particle>& particles, std::size_t begin, std::size_t end, double dt) {
    end = std::min(end, particles.size());
    for (std::size_t i = begin; i < end; ++i) {
        particles[i].velocity += particles[i].acceleration * dt;
    }
}

void drift(std::vector<Particle>& particles, std::size_t begin, std::size_t end, double dt) {
    end = std::min(end, particles.size());
    for (std::size_t i = begin; i < end; ++i) {
        particles[i].position += particles[i].velocity * dt;
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
