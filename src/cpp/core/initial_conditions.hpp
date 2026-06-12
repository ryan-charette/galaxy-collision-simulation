#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace fmmgalaxy {

/// @brief Parameters for generating one rotating disk galaxy.
///
/// Positions, velocities, radius, and mass are nondimensional code-unit values. Orientation
/// rotates the disk in the x-y plane; inclination tilts it out of plane; thickness adds
/// vertical scatter for three-dimensional runs.
struct GalaxyConfig {
    /// Number of particles to sample for this galaxy.
    std::size_t n_particles{256};
    /// Total galaxy mass distributed equally across sampled particles.
    double mass{1.0};
    /// Characteristic disk radius.
    double radius{1.0};
    /// Center-of-mass position.
    Vec2 position{};
    /// Center-of-mass velocity.
    Vec2 velocity{};
    /// In-plane rotation angle in radians.
    double orientation{0.0};
    /// Group label assigned to generated particles.
    std::uint32_t group_id{0};
    /// Vertical disk thickness scale.
    double thickness{0.0};
    /// Out-of-plane inclination angle in radians.
    double inclination{0.0};
};

/// Generate one randomized disk galaxy using the provided random-number generator.
std::vector<Particle> generate_disk_galaxy(
    const GalaxyConfig& config,
    const PhysicsParams& physics,
    std::mt19937_64& rng
);

/// Generate and concatenate all configured galaxies with a reproducible seed.
std::vector<Particle> generate_galaxies(
    const std::vector<GalaxyConfig>& galaxies,
    const PhysicsParams& physics,
    std::uint64_t seed
);

}  // namespace fmmgalaxy
