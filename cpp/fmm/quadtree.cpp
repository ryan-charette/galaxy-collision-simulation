#include "fmm/quadtree.hpp"

#include "direct/direct_solver.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace fmmgalaxy {

BarnesHutSolver::BarnesHutSolver(
    PhysicsParams params,
    double theta,
    std::size_t leaf_capacity,
    int max_depth,
    int expansion_order
)
    : params_(params),
      theta_(theta),
      leaf_capacity_(std::max<std::size_t>(1, leaf_capacity)),
      max_depth_(std::max(1, max_depth)),
      expansion_order_(std::clamp(expansion_order, 0, 4)) {}

void BarnesHutSolver::compute(std::vector<Particle>& particles) {
    reset_accelerations(particles);
    if (particles.empty()) {
        return;
    }

    build(particles);

    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].acceleration = accumulate_from_node(0, i, particles[i].position);
    }
}

void BarnesHutSolver::build(const std::vector<Particle>& particles) {
    particles_ = &particles;
    nodes_.clear();
    nodes_.reserve(particles.size() * 2 + 1);

    Vec2 min_position{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec2 max_position{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };

    for (const auto& particle : particles) {
        min_position.x = std::min(min_position.x, particle.position.x);
        min_position.y = std::min(min_position.y, particle.position.y);
        min_position.z = std::min(min_position.z, particle.position.z);
        max_position.x = std::max(max_position.x, particle.position.x);
        max_position.y = std::max(max_position.y, particle.position.y);
        max_position.z = std::max(max_position.z, particle.position.z);
    }

    Node root;
    root.center = (min_position + max_position) * 0.5;
    root.half_width = 0.5 * std::max({
        max_position.x - min_position.x,
        max_position.y - min_position.y,
        max_position.z - min_position.z,
    });
    root.half_width = std::max(root.half_width, params_.softening + 1.0e-6);
    root.half_width *= 1.0001;
    nodes_.push_back(root);

    for (std::size_t i = 0; i < particles.size(); ++i) {
        insert_particle(0, i, 0);
    }

    compute_moments(0);
    compute_multipole_moments(0);
}

bool BarnesHutSolver::is_leaf(const Node& node) const {
    return node.children[0] < 0;
}

bool BarnesHutSolver::contains(const Node& node, const Vec2& position) const {
    return std::abs(position.x - node.center.x) <= node.half_width &&
           std::abs(position.y - node.center.y) <= node.half_width &&
           std::abs(position.z - node.center.z) <= node.half_width;
}

int BarnesHutSolver::child_index_for(const Node& node, const Vec2& position) const {
    const int east = position.x >= node.center.x ? 1 : 0;
    const int north = position.y >= node.center.y ? 1 : 0;
    const int up = position.z >= node.center.z ? 1 : 0;
    return east + 2 * north + 4 * up;
}

void BarnesHutSolver::subdivide(int node_index) {
    const Node node = nodes_[static_cast<std::size_t>(node_index)];
    const double child_half_width = node.half_width * 0.5;

    for (int child = 0; child < 8; ++child) {
        const double x_sign = (child & 1) ? 1.0 : -1.0;
        const double y_sign = (child & 2) ? 1.0 : -1.0;
        const double z_sign = (child & 4) ? 1.0 : -1.0;

        Node child_node;
        child_node.center = {
            node.center.x + x_sign * child_half_width,
            node.center.y + y_sign * child_half_width,
            node.center.z + z_sign * child_half_width,
        };
        child_node.half_width = child_half_width;
        nodes_.push_back(child_node);
        nodes_[static_cast<std::size_t>(node_index)].children[static_cast<std::size_t>(child)] =
            static_cast<int>(nodes_.size() - 1);
    }
}

void BarnesHutSolver::insert_particle(int node_index, std::size_t particle_index, int depth) {
    Node& node = nodes_[static_cast<std::size_t>(node_index)];

    if (is_leaf(node) && (node.particle_indices.size() < leaf_capacity_ || depth >= max_depth_)) {
        node.particle_indices.push_back(particle_index);
        return;
    }

    if (is_leaf(node)) {
        const std::vector<std::size_t> existing_particles = node.particle_indices;
        node.particle_indices.clear();
        subdivide(node_index);

        for (const std::size_t existing_index : existing_particles) {
            insert_particle(node_index, existing_index, depth);
        }
    }

    const Node& current = nodes_[static_cast<std::size_t>(node_index)];
    const int child = child_index_for(current, (*particles_)[particle_index].position);
    insert_particle(current.children[static_cast<std::size_t>(child)], particle_index, depth + 1);
}

double BarnesHutSolver::compute_moments(int node_index) {
    Node& node = nodes_[static_cast<std::size_t>(node_index)];

    double mass = 0.0;
    Vec2 weighted_position{};

    if (is_leaf(node)) {
        for (const std::size_t particle_index : node.particle_indices) {
            const Particle& particle = (*particles_)[particle_index];
            mass += particle.mass;
            weighted_position += particle.position * particle.mass;
        }
    } else {
        for (const int child_index : node.children) {
            if (child_index >= 0) {
                const double child_mass = compute_moments(child_index);
                const Node& child = nodes_[static_cast<std::size_t>(child_index)];
                mass += child_mass;
                weighted_position += child.center_of_mass * child_mass;
            }
        }
    }

    node.mass = mass;
    node.center_of_mass = mass > 0.0 ? weighted_position / mass : node.center;
    return mass;
}

void BarnesHutSolver::compute_multipole_moments(int node_index) {
    Node& node = nodes_[static_cast<std::size_t>(node_index)];
    node.moments = zero_multipole_moments();

    if (is_leaf(node)) {
        for (const std::size_t particle_index : node.particle_indices) {
            const Particle& particle = (*particles_)[particle_index];
            add_multipole_point(node.moments, particle.position - node.center_of_mass, particle.mass);
        }
        return;
    }

    for (const int child_index : node.children) {
        if (child_index >= 0) {
            compute_multipole_moments(child_index);
            const Node& child = nodes_[static_cast<std::size_t>(child_index)];
            add_multipole_shifted_child(
                node.moments,
                child.moments,
                child.center_of_mass - node.center_of_mass,
                child.mass
            );
        }
    }
}

Vec2 BarnesHutSolver::accumulate_from_node(
    int node_index,
    std::size_t target_index,
    const Vec2& target_position
) const {
    const Node& node = nodes_[static_cast<std::size_t>(node_index)];
    if (node.mass <= 0.0) {
        return {};
    }

    if (is_leaf(node)) {
        Vec2 acceleration{};
        for (const std::size_t source_index : node.particle_indices) {
            if (source_index == target_index) {
                continue;
            }
            const Particle& source = (*particles_)[source_index];
            acceleration += softened_acceleration(target_position, source.position, source.mass, params_);
        }
        return acceleration;
    }

    const Vec2 delta = node.center_of_mass - target_position;
    const double distance = norm(delta);
    const double node_width = 2.0 * node.half_width;
    const bool target_inside_node = contains(node, target_position);

    if (!target_inside_node && distance > 0.0 && node_width / distance < theta_) {
        return multipole_acceleration(
            target_position,
            node.center_of_mass,
            node.mass,
            node.moments,
            params_,
            expansion_order_
        );
    }

    Vec2 acceleration{};
    for (const int child_index : node.children) {
        if (child_index >= 0) {
            acceleration += accumulate_from_node(child_index, target_index, target_position);
        }
    }
    return acceleration;
}

void compute_tree_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    double theta,
    std::size_t leaf_capacity,
    int expansion_order
) {
    BarnesHutSolver solver(params, theta, leaf_capacity, 32, expansion_order);
    solver.compute(particles);
}

}  // namespace fmmgalaxy
