#include "fmm/quadtree.hpp"

#include "direct/direct_solver.hpp"
#include "fmm/tree_geometry.hpp"

#include <algorithm>
#include <cmath>

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

FlatTreeData BarnesHutSolver::build_flat_tree(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return {};
    }
    build(particles);
    return export_flat_tree();
}

void BarnesHutSolver::build(const std::vector<Particle>& particles) {
    particles_ = &particles;
    nodes_.clear();
    nodes_.reserve(particles.size() * 2 + 1);

    const TreeRootCube root_cube = root_cube_for_particles(particles, params_);

    Node root;
    root.center = root_cube.center;
    root.half_width = root_cube.half_width;
    nodes_.push_back(root);

    for (std::size_t i = 0; i < particles.size(); ++i) {
        insert_particle(0, i, 0);
    }

    compute_moments(0);
    if (expansion_order_ > 0) {
        compute_multipole_moments(0);
    }
}

bool BarnesHutSolver::is_leaf(const Node& node) const {
    return node.children[0] < 0;
}

bool BarnesHutSolver::contains(const Node& node, const Vec2& position) const {
    return std::abs(position.x - node.center.x) <= node.half_width &&
           std::abs(position.y - node.center.y) <= node.half_width &&
           std::abs(position.z - node.center.z) <= node.half_width;
}

void BarnesHutSolver::subdivide(int node_index) {
    const Node node = nodes_[static_cast<std::size_t>(node_index)];
    const double child_half_width = node.half_width * 0.5;

    for (int child = 0; child < 8; ++child) {
        Node child_node;
        child_node.center = child_center(node.center, child_half_width, child);
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
    const int child = child_index_for_position(current.center, (*particles_)[particle_index].position);
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

FlatTreeData BarnesHutSolver::export_flat_tree() const {
    FlatTreeData flat;
    flat.nodes.reserve(nodes_.size());

    for (const Node& node : nodes_) {
        FlatTreeNode flat_node;
        flat_node.center = node.center;
        flat_node.half_width = node.half_width;
        flat_node.mass = node.mass;
        flat_node.center_of_mass = node.center_of_mass;
        flat_node.moments = node.moments;
        flat_node.children = node.children;
        flat_node.particle_begin = flat.particle_indices.size();
        flat_node.particle_count = node.particle_indices.size();
        flat_node.is_leaf = is_leaf(node);
        flat.particle_indices.insert(
            flat.particle_indices.end(),
            node.particle_indices.begin(),
            node.particle_indices.end()
        );
        flat.nodes.push_back(flat_node);
    }

    return flat;
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

FlatTreeData build_flat_tree(
    const std::vector<Particle>& particles,
    const PhysicsParams& params,
    double theta,
    std::size_t leaf_capacity,
    int max_depth,
    int expansion_order
) {
    BarnesHutSolver solver(params, theta, leaf_capacity, max_depth, expansion_order);
    return solver.build_flat_tree(particles);
}

}  // namespace fmmgalaxy
