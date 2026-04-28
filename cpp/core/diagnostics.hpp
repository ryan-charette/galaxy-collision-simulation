#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <vector>

namespace fmmgalaxy {

struct Diagnostics {
    double kinetic_energy{0.0};
    double potential_energy{0.0};
    double total_energy{0.0};
    double total_mass{0.0};
    Vec2 momentum{};
    Vec2 center_of_mass{};
    double angular_momentum{0.0};
};

Diagnostics compute_diagnostics(const std::vector<Particle>& particles, const PhysicsParams& params);

}  // namespace fmmgalaxy
