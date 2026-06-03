#include "tests/test_support.hpp"

#include "core/config.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/simulation_info.hpp"
#include "core/vector2.hpp"
#include "direct/direct_solver.hpp"

#include <cmath>
#include <vector>

int run_math_direct_tests() {
    using fmmgalaxy::Vec2;
    using fmmgalaxy::tests::near;
    using fmmgalaxy::tests::require;

    int failures = 0;

    Vec2 a{1.0, 2.0};
    Vec2 b{3.0, 4.0, 5.0};
    Vec2 c = a + b;

    failures += !require(c.x == 4.0, "Vec2 x addition");
    failures += !require(c.y == 6.0, "Vec2 y addition");
    failures += !require(c.z == 5.0, "Vec3 z addition");
    failures += !require(near(fmmgalaxy::dot(a, b), 11.0, 1.0e-12), "Vec3 dot product");
    failures += !require(
        near(fmmgalaxy::cross({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}).z, 1.0, 1.0e-12),
        "Vec3 cross product"
    );
    failures += !require(!fmmgalaxy::build_summary().empty(), "build summary is non-empty");

    fmmgalaxy::PhysicsParams physics;
    physics.gravitational_constant = 1.0;
    physics.softening = 0.0;

    std::vector<fmmgalaxy::Particle> two_body(2);
    two_body[0].position = {-0.5, 0.0, 0.2};
    two_body[0].mass = 2.0;
    two_body[1].position = {0.5, 0.0, -0.2};
    two_body[1].mass = 3.0;
    fmmgalaxy::compute_direct_accelerations(two_body, physics);
    failures += !require(two_body[0].acceleration.x > 0.0, "direct acceleration attracts particle 0");
    failures += !require(two_body[1].acceleration.x < 0.0, "direct acceleration attracts particle 1");

    const double net_force_x =
        two_body[0].mass * two_body[0].acceleration.x +
        two_body[1].mass * two_body[1].acceleration.x;
    failures += !require(near(net_force_x, 0.0, 1.0e-12), "pairwise direct force is symmetric");

    two_body[0].velocity = {0.0, 0.5};
    two_body[1].velocity = {0.0, -1.0 / 3.0};
    auto direct_acceleration = [&physics](std::vector<fmmgalaxy::Particle>& particles) {
        fmmgalaxy::compute_direct_accelerations(particles, physics);
    };
    fmmgalaxy::leapfrog_step(two_body, 0.001, direct_acceleration);
    failures += !require(std::isfinite(two_body[0].position.x), "leapfrog position stays finite");

    fmmgalaxy::SimulationConfig config = fmmgalaxy::default_config();
    const auto generated = fmmgalaxy::generate_galaxies(config.galaxies, config.physics, config.seed);
    failures += !require(
        generated.size() == config.n_particles,
        "default config generates expected particle count"
    );

    return failures;
}
