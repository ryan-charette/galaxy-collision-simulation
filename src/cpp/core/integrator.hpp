#pragma once

#include "core/particle.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace fmmgalaxy {

/// Function object that updates all particle accelerations in place.
using AccelerationFunction = std::function<void(std::vector<Particle>&)>;

/// Apply a velocity kick over `dt` using already-computed accelerations.
void kick(std::vector<Particle>& particles, double dt);
/// Drift particle positions over `dt` using current velocities.
void drift(std::vector<Particle>& particles, double dt);
/// Apply a velocity kick to the half-open particle range `[begin, end)`.
void kick(std::vector<Particle>& particles, std::size_t begin, std::size_t end, double dt);
/// Drift particle positions for the half-open particle range `[begin, end)`.
void drift(std::vector<Particle>& particles, std::size_t begin, std::size_t end, double dt);

/// @brief Advance particles by one kick-drift-kick leapfrog step.
///
/// The supplied acceleration function is called after the drift to refresh accelerations at
/// the new positions before the final half-kick.
void leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const AccelerationFunction& compute_accelerations
);

}  // namespace fmmgalaxy
