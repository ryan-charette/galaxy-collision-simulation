#include "fmm/fmm_solver.hpp"

#include "direct/direct_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fmmgalaxy {

FastMultipoleSolver::FastMultipoleSolver(PhysicsParams params, FmmOptions options)
    : params_(params), options_(options) {
    options_.leaf_capacity = std::max<std::size_t>(1, options_.leaf_capacity);
    options_.max_depth = std::max(1, options_.max_depth);
    options_.theta = std::max(1.0e-6, options_.theta);
}

void FastMultipoleSolver::compute(std::vector<Particle>& particles) {
    compute_targets(particles, 0, particles.size());
}

void FastMultipoleSolver::compute_targets(
    std::vector<Particle>& particles,
    std::size_t begin,
    std::size_t end
) {
    end = std::min(end, particles.size());
    reset_accelerations(particles, begin, end);

    if (particles.empty() || begin >= end) {
        return;
    }

    build(particles);

    for (const int leaf_index : leaf_indices_) {
        const Node& leaf = nodes_[static_cast<std::size_t>(leaf_index)];
        for (const std::size_t particle_index : leaf.particle_indices) {
            if (particle_index >= begin && particle_index < end) {
                particles[particle_index].acceleration = evaluate_particle(particle_index, leaf);
            }
        }
    }
}

void FastMultipoleSolver::build(const std::vector<Particle>& particles) {
    particles_ = &particles;
    nodes_.clear();
    leaf_indices_.clear();
    stats_ = {};

    nodes_.reserve(particles.size() * 2 + 1);

    Vec2 min_position{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec2 max_position{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };

    for (const auto& particle : particles) {
        min_position.x = std::min(min_position.x, particle.position.x);
        min_position.y = std::min(min_position.y, particle.position.y);
        max_position.x = std::max(max_position.x, particle.position.x);
        max_position.y = std::max(max_position.y, particle.position.y);
    }

    Node root;
    root.center = (min_position + max_position) * 0.5;
    root.half_width = 0.5 * std::max(max_position.x - min_position.x, max_position.y - min_position.y);
    root.half_width = std::max(root.half_width, params_.softening + 1.0e-6);
    root.half_width *= 1.0001;
    nodes_.push_back(root);

    for (std::size_t i = 0; i < particles.size(); ++i) {
        insert_particle(0, i, 0);
    }

    p2m_m2m(0);
    collect_leaves(0);
    build_interaction_lists();

    stats_.node_count = nodes_.size();
    stats_.leaf_count = leaf_indices_.size();
}

bool FastMultipoleSolver::is_leaf(const Node& node) const {
    return node.children[0] < 0;
}

int FastMultipoleSolver::child_index_for(const Node& node, const Vec2& position) const {
    const int east = position.x >= node.center.x ? 1 : 0;
    const int north = position.y >= node.center.y ? 1 : 0;
    return east + 2 * north;
}

void FastMultipoleSolver::subdivide(int node_index) {
    const Node node = nodes_[static_cast<std::size_t>(node_index)];
    const double child_half_width = node.half_width * 0.5;

    for (int child = 0; child < 4; ++child) {
        const double x_sign = (child & 1) ? 1.0 : -1.0;
        const double y_sign = (child & 2) ? 1.0 : -1.0;

        Node child_node;
        child_node.center = {
            node.center.x + x_sign * child_half_width,
            node.center.y + y_sign * child_half_width,
        };
        child_node.half_width = child_half_width;
        nodes_.push_back(child_node);
        nodes_[static_cast<std::size_t>(node_index)].children[static_cast<std::size_t>(child)] =
            static_cast<int>(nodes_.size() - 1);
    }
}

void FastMultipoleSolver::insert_particle(int node_index, std::size_t particle_index, int depth) {
    Node& node = nodes_[static_cast<std::size_t>(node_index)];

    if (is_leaf(node) &&
        (node.particle_indices.size() < options_.leaf_capacity || depth >= options_.max_depth)) {
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

double FastMultipoleSolver::p2m_m2m(int node_index) {
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
                const double child_mass = p2m_m2m(child_index);
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

void FastMultipoleSolver::collect_leaves(int node_index) {
    const Node& node = nodes_[static_cast<std::size_t>(node_index)];
    if (node.mass <= 0.0) {
        return;
    }

    if (is_leaf(node)) {
        leaf_indices_.push_back(node_index);
        return;
    }

    for (const int child_index : node.children) {
        if (child_index >= 0) {
            collect_leaves(child_index);
        }
    }
}

bool FastMultipoleSolver::well_separated(const Node& target, const Node& source) const {
    if (source.mass <= 0.0) {
        return false;
    }

    const Vec2 delta = source.center - target.center;
    const double distance = norm(delta);
    if (distance == 0.0) {
        return false;
    }

    const double combined_width = 2.0 * (target.half_width + source.half_width);
    return combined_width / distance < options_.theta;
}

void FastMultipoleSolver::build_interaction_lists() {
    for (const int target_leaf_index : leaf_indices_) {
        Node& leaf = nodes_[static_cast<std::size_t>(target_leaf_index)];
        leaf.far_nodes.clear();
        leaf.near_leaves.clear();
        build_leaf_interactions(target_leaf_index, 0);
        stats_.far_interactions += leaf.far_nodes.size();
        stats_.near_interactions += leaf.near_leaves.size();
    }
}

void FastMultipoleSolver::build_leaf_interactions(int target_leaf_index, int source_node_index) {
    Node& target = nodes_[static_cast<std::size_t>(target_leaf_index)];
    const Node& source = nodes_[static_cast<std::size_t>(source_node_index)];

    if (source.mass <= 0.0) {
        return;
    }

    if (source_node_index != target_leaf_index && well_separated(target, source)) {
        target.far_nodes.push_back(source_node_index);
        return;
    }

    if (is_leaf(source)) {
        target.near_leaves.push_back(source_node_index);
        return;
    }

    for (const int child_index : source.children) {
        if (child_index >= 0) {
            build_leaf_interactions(target_leaf_index, child_index);
        }
    }
}

Vec2 FastMultipoleSolver::evaluate_particle(std::size_t target_index, const Node& target_leaf) const {
    const Particle& target = (*particles_)[target_index];
    Vec2 acceleration{};

    for (const int source_node_index : target_leaf.far_nodes) {
        const Node& source = nodes_[static_cast<std::size_t>(source_node_index)];
        acceleration += softened_acceleration(
            target.position,
            source.center_of_mass,
            source.mass,
            params_
        );
    }

    for (const int source_leaf_index : target_leaf.near_leaves) {
        const Node& source_leaf = nodes_[static_cast<std::size_t>(source_leaf_index)];
        for (const std::size_t source_particle_index : source_leaf.particle_indices) {
            if (source_particle_index == target_index) {
                continue;
            }

            const Particle& source = (*particles_)[source_particle_index];
            acceleration += softened_acceleration(
                target.position,
                source.position,
                source.mass,
                params_
            );
        }
    }

    return acceleration;
}

void compute_fmm_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    FmmOptions options
) {
    FastMultipoleSolver solver(params, options);
    solver.compute(particles);
}

void compute_fmm_accelerations_for_targets(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    std::size_t begin,
    std::size_t end,
    FmmOptions options
) {
    FastMultipoleSolver solver(params, options);
    solver.compute_targets(particles, begin, end);
}

}  // namespace fmmgalaxy
