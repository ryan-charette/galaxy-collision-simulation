#pragma once

#include <cstdint>
#include "core/vector2.hpp"

namespace fmmgalaxy {

/// @brief One collisionless N-body particle in code units.
///
/// Particles store their current phase-space state plus a group label identifying the source
/// galaxy. Solver functions update the acceleration field in place.
struct Particle {
    /// Cartesian position.
    Vec2 position{};
    /// Cartesian velocity.
    Vec2 velocity{};
    /// Cartesian acceleration from the most recent force evaluation.
    Vec2 acceleration{};
    /// Particle mass in code units.
    double mass{1.0};
    /// Stable source-galaxy or population identifier used by plotting and diagnostics.
    std::uint32_t group_id{0};
};

}  // namespace fmmgalaxy
