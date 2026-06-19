#include "core/config.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/simulation_info.hpp"
#include "core/vector2.hpp"
#include "direct/direct_solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

TEST_CASE("Vector math, direct forces, and leapfrog integration stay consistent", "[math][direct]") {
    using Catch::Matchers::WithinAbs;
    using fmmgalaxy::Vec2;

    Vec2 a{1.0, 2.0};
    Vec2 b{3.0, 4.0, 5.0};
    Vec2 c = a + b;

    CHECK(c.x == 4.0);
    CHECK(c.y == 6.0);
    CHECK(c.z == 5.0);
    CHECK_THAT(fmmgalaxy::dot(a, b), WithinAbs(11.0, 1.0e-12));
    CHECK_THAT(
        fmmgalaxy::cross({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}).z,
        WithinAbs(1.0, 1.0e-12)
    );
    CHECK_FALSE(fmmgalaxy::build_summary().empty());

    fmmgalaxy::PhysicsParams physics;
    physics.gravitational_constant = 1.0;
    physics.softening = 0.0;

    std::vector<fmmgalaxy::Particle> two_body(2);
    two_body[0].position = {-0.5, 0.0, 0.2};
    two_body[0].mass = 2.0;
    two_body[1].position = {0.5, 0.0, -0.2};
    two_body[1].mass = 3.0;
    fmmgalaxy::compute_direct_accelerations(two_body, physics);
    CHECK(two_body[0].acceleration.x > 0.0);
    CHECK(two_body[1].acceleration.x < 0.0);

    const double net_force_x =
        two_body[0].mass * two_body[0].acceleration.x +
        two_body[1].mass * two_body[1].acceleration.x;
    CHECK_THAT(net_force_x, WithinAbs(0.0, 1.0e-12));

    two_body[0].velocity = {0.0, 0.5};
    two_body[1].velocity = {0.0, -1.0 / 3.0};
    auto direct_acceleration = [&physics](std::vector<fmmgalaxy::Particle>& particles) {
        fmmgalaxy::compute_direct_accelerations(particles, physics);
    };
    fmmgalaxy::leapfrog_step(two_body, 0.001, direct_acceleration);
    CHECK(std::isfinite(two_body[0].position.x));

    fmmgalaxy::SimulationConfig config = fmmgalaxy::default_config();
    const auto generated = fmmgalaxy::generate_galaxies(config.galaxies, config.physics, config.seed);
    CHECK(generated.size() == config.n_particles);
}
