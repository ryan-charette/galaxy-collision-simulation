#pragma once

#include "core/particle.hpp"

#include <functional>
#include <vector>

namespace fmmgalaxy {

using AccelerationFunction = std::function<void(std::vector<Particle>&)>;

void kick(std::vector<Particle>& particles, double dt);
void drift(std::vector<Particle>& particles, double dt);
void leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const AccelerationFunction& compute_accelerations
);

}  // namespace fmmgalaxy
