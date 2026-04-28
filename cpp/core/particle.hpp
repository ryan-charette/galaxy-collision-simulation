#pragma once

#include <cstdint>
#include "core/vector2.hpp"

namespace fmmgalaxy {

struct Particle {
    Vec2 position{};
    Vec2 velocity{};
    Vec2 acceleration{};
    double mass{1.0};
    std::uint32_t group_id{0};
};

}  // namespace fmmgalaxy
