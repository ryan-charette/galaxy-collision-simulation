#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <vector>

namespace fmmgalaxy {

/// @brief Conserved-quantity diagnostics for one simulator state.
///
/// Energies use the softened gravitational potential corresponding to `PhysicsParams`.
/// Momentum and angular momentum are aggregated over all particles.
struct Diagnostics {
    /// Sum of particle kinetic energies.
    double kinetic_energy{0.0};
    /// Pairwise softened gravitational potential energy.
    double potential_energy{0.0};
    /// `kinetic_energy + potential_energy`.
    double total_energy{0.0};
    /// Sum of all particle masses.
    double total_mass{0.0};
    /// Total linear momentum.
    Vec2 momentum{};
    /// Mass-weighted center of mass.
    Vec2 center_of_mass{};
    /// Total angular momentum about the origin.
    Vec2 angular_momentum{};
};

/// Compute energy, mass, momentum, center-of-mass, and angular-momentum diagnostics.
Diagnostics compute_diagnostics(const std::vector<Particle>& particles, const PhysicsParams& params);

}  // namespace fmmgalaxy
