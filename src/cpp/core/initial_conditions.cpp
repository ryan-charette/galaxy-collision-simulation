#include "core/initial_conditions.hpp"

#include <algorithm>
#include <cmath>

namespace fmmgalaxy {

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

Vec2 rotate_z(const Vec2& v, double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return {c * v.x - s * v.y, s * v.x + c * v.y, v.z};
}

Vec2 rotate_x(const Vec2& v, double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return {v.x, c * v.y - s * v.z, s * v.y + c * v.z};
}

Vec2 orient_disk_vector(const Vec2& v, double orientation, double inclination) {
    return rotate_z(rotate_x(v, inclination), orientation);
}

double sample_truncated_exponential_radius(double radius, std::uniform_real_distribution<double>& unit, std::mt19937_64& rng) {
    const double scale = radius / 3.0;
    const double max_cdf = 1.0 - std::exp(-radius / scale);
    const double u = std::clamp(unit(rng), 1.0e-12, 1.0 - 1.0e-12);
    return -scale * std::log(1.0 - u * max_cdf);
}

}  // namespace

std::vector<Particle> generate_disk_galaxy(
    const GalaxyConfig& config,
    const PhysicsParams& physics,
    std::mt19937_64& rng
) {
    std::vector<Particle> particles;
    particles.reserve(config.n_particles);

    if (config.n_particles == 0 || config.mass <= 0.0 || config.radius <= 0.0) {
        return particles;
    }

    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double particle_mass = config.mass / static_cast<double>(config.n_particles);
    const double velocity_scale = std::sqrt(physics.gravitational_constant * config.mass / config.radius);
    const double dispersion = 0.025 * velocity_scale;

    for (std::size_t i = 0; i < config.n_particles; ++i) {
        const double radius = sample_truncated_exponential_radius(config.radius, unit, rng);
        const double theta = 2.0 * pi * unit(rng);

        const Vec2 local_position{
            radius * std::cos(theta),
            radius * std::sin(theta),
            config.thickness > 0.0 ? config.thickness * normal(rng) : 0.0,
        };
        const Vec2 tangent{-std::sin(theta), std::cos(theta), 0.0};

        const double enclosed_fraction = std::min(1.0, (radius * radius) / (config.radius * config.radius));
        const double enclosed_mass = std::max(config.mass * enclosed_fraction, particle_mass);
        const double softened_r2 = radius * radius + physics.softening * physics.softening;
        const double circular_speed =
            std::sqrt(physics.gravitational_constant * enclosed_mass * radius * radius /
                      std::pow(softened_r2, 1.5));

        Vec2 local_velocity = tangent * circular_speed;
        local_velocity += {
            dispersion * normal(rng),
            dispersion * normal(rng),
            0.25 * dispersion * normal(rng),
        };

        Particle particle;
        particle.position =
            config.position + orient_disk_vector(local_position, config.orientation, config.inclination);
        particle.velocity =
            config.velocity + orient_disk_vector(local_velocity, config.orientation, config.inclination);
        particle.mass = particle_mass;
        particle.group_id = config.group_id;
        particles.push_back(particle);
    }

    return particles;
}

std::vector<Particle> generate_galaxies(
    const std::vector<GalaxyConfig>& galaxies,
    const PhysicsParams& physics,
    std::uint64_t seed
) {
    std::mt19937_64 rng(seed);
    std::vector<Particle> particles;

    std::size_t total_particles = 0;
    for (const auto& galaxy : galaxies) {
        total_particles += galaxy.n_particles;
    }
    particles.reserve(total_particles);

    for (const auto& galaxy : galaxies) {
        auto generated = generate_disk_galaxy(galaxy, physics, rng);
        particles.insert(particles.end(), generated.begin(), generated.end());
    }

    return particles;
}

}  // namespace fmmgalaxy
