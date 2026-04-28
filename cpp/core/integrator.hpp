#pragma once

#include "core/particle.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace fmmgalaxy {

using AccelerationFunction = std::function<void(std::vector<Particle>&)>;

void kick(std::vector<Particle>& particles, double dt);
void drift(std::vector<Particle>& particles, double dt);
void kick(std::vector<Particle>& particles, std::size_t begin, std::size_t end, double dt);
void drift(std::vector<Particle>& particles, std::size_t begin, std::size_t end, double dt);
void leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const AccelerationFunction& compute_accelerations
);

}  // namespace fmmgalaxy
