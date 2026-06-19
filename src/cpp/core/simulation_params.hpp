#pragma once

namespace fmmgalaxy {

/// @brief Physical constants shared by all force solvers.
///
/// Values are interpreted in nondimensional code units. The same parameters are passed to
/// direct, Barnes-Hut, FMM, and CUDA force paths so solver comparisons use identical physics.
struct PhysicsParams {
    /// Newtonian gravitational constant in code units.
    double gravitational_constant{1.0};
    /// Plummer-style softening length used as `eps` in `r^2 + eps^2`.
    double softening{0.02};
};

}  // namespace fmmgalaxy
