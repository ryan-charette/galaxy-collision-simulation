#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace fmmgalaxy {

struct GalaxyConfig {
    std::size_t n_particles{256};
    double mass{1.0};
    double radius{1.0};
    Vec2 position{};
    Vec2 velocity{};
    double orientation{0.0};
    std::uint32_t group_id{0};
};

std::vector<Particle> generate_disk_galaxy(
    const GalaxyConfig& config,
    const PhysicsParams& physics,
    std::mt19937_64& rng
);

std::vector<Particle> generate_galaxies(
    const std::vector<GalaxyConfig>& galaxies,
    const PhysicsParams& physics,
    std::uint64_t seed
);

}  // namespace fmmgalaxy
