#include "fmm/fmm_solver.hpp"

#include "direct/direct_solver.hpp"
#include "fmm/tree_geometry.hpp"

#include <algorithm>
#include <cmath>

namespace fmmgalaxy {

FastMultipoleSolver::FastMultipoleSolver(PhysicsParams params, FmmOptions options)
    : params_(params), options_(options) {
    options_.leaf_capacity = std::max<std::size_t>(1, options_.leaf_capacity);
    options_.max_depth = std::max(1, options_.max_depth);
    options_.theta = std::max(1.0e-6, options_.theta);
    options_.expansion_order = std::clamp(options_.expansion_order, 0, 4);
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

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int leaf_position = 0; leaf_position < static_cast<int>(leaf_indices_.size()); ++leaf_position) {
        const int leaf_index = leaf_indices_[static_cast<std::size_t>(leaf_position)];
        const Node& leaf = nodes_[static_cast<std::size_t>(leaf_index)];
        for (const std::size_t particle_index : leaf.particle_indices) {
            if (particle_index >= begin && particle_index < end) {
                particles[particle_index].acceleration = evaluate_particle(particle_index, leaf);
            }
        }
    }
}

FlatFmmData FastMultipoleSolver::build_flat_fmm(const std::vector<Particle>& particles) {
    if (particles.empty()) {
        return {};
    }
    build(particles);
    return export_flat_fmm();
}

void FastMultipoleSolver::build(const std::vector<Particle>& particles) {
    particles_ = &particles;
    nodes_.clear();
    leaf_indices_.clear();
    levels_.clear();
    stats_ = {};

    nodes_.reserve(particles.size() * 2 + 1);

    const TreeRootCube root_cube = root_cube_for_particles(particles, params_);

    Node root;
    root.center = root_cube.center;
    root.half_width = root_cube.half_width;
    root.local = zero_local_expansion(root.center, root.half_width, options_.expansion_order);
    nodes_.push_back(root);

    for (std::size_t i = 0; i < particles.size(); ++i) {
        insert_particle(0, i, 0);
    }

    build_levels();
    compute_masses_and_moments();
    collect_leaves(0);
    build_interaction_lists();
    compute_local_expansions();

    stats_.node_count = nodes_.size();
    stats_.leaf_count = leaf_indices_.size();
}

bool FastMultipoleSolver::is_leaf(const Node& node) const {
    return node.children[0] < 0;
}

void FastMultipoleSolver::subdivide(int node_index) {
    const Node node = nodes_[static_cast<std::size_t>(node_index)];
    const double child_half_width = node.half_width * 0.5;

    for (int child = 0; child < 8; ++child) {
        Node child_node;
        child_node.center = child_center(node.center, child_half_width, child);
        child_node.half_width = child_half_width;
        child_node.parent = node_index;
        child_node.depth = node.depth + 1;
        child_node.local = zero_local_expansion(
            child_node.center,
            child_node.half_width,
            options_.expansion_order
        );
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
    const int child = child_index_for_position(current.center, (*particles_)[particle_index].position);
    insert_particle(current.children[static_cast<std::size_t>(child)], particle_index, depth + 1);
}

void FastMultipoleSolver::build_levels() {
    int max_depth = 0;
    for (const Node& node : nodes_) {
        max_depth = std::max(max_depth, node.depth);
    }

    levels_.assign(static_cast<std::size_t>(max_depth + 1), {});
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
        levels_[static_cast<std::size_t>(nodes_[index].depth)].push_back(static_cast<int>(index));
    }
}

void FastMultipoleSolver::compute_masses_and_moments() {
    if (particles_ == nullptr || levels_.empty()) {
        return;
    }

    for (Node& node : nodes_) {
        node.mass = 0.0;
        node.center_of_mass = node.center;
        node.moments = zero_multipole_moments();
        node.local = zero_local_expansion(node.center, node.half_width, options_.expansion_order);
    }

    for (int depth = static_cast<int>(levels_.size()) - 1; depth >= 0; --depth) {
        const std::vector<int>& level = levels_[static_cast<std::size_t>(depth)];

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int position = 0; position < static_cast<int>(level.size()); ++position) {
            Node& node = nodes_[static_cast<std::size_t>(level[static_cast<std::size_t>(position)])];
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
                    if (child_index < 0) {
                        continue;
                    }
                    const Node& child = nodes_[static_cast<std::size_t>(child_index)];
                    mass += child.mass;
                    weighted_position += child.center_of_mass * child.mass;
                }
            }

            node.mass = mass;
            node.center_of_mass = mass > 0.0 ? weighted_position / mass : node.center;
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int position = 0; position < static_cast<int>(level.size()); ++position) {
            Node& node = nodes_[static_cast<std::size_t>(level[static_cast<std::size_t>(position)])];
            if (node.mass <= 0.0) {
                continue;
            }

            node.moments = zero_multipole_moments();
            if (is_leaf(node)) {
                for (const std::size_t particle_index : node.particle_indices) {
                    const Particle& particle = (*particles_)[particle_index];
                    add_multipole_point(
                        node.moments,
                        particle.position - node.center_of_mass,
                        particle.mass
                    );
                }
                continue;
            }

            for (const int child_index : node.children) {
                if (child_index < 0) {
                    continue;
                }
                const Node& child = nodes_[static_cast<std::size_t>(child_index)];
                if (child.mass <= 0.0) {
                    continue;
                }
                add_multipole_shifted_child(
                    node.moments,
                    child.moments,
                    child.center_of_mass - node.center_of_mass,
                    child.mass
                );
            }
        }
    }
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
    if (nodes_.empty() || levels_.empty()) {
        return;
    }

    for (Node& node : nodes_) {
        node.far_nodes.clear();
        node.near_nodes.clear();
        node.near_leaves.clear();
    }

    nodes_[0].near_nodes.push_back(0);

    for (std::size_t depth = 1; depth < levels_.size(); ++depth) {
        const std::vector<int>& level = levels_[depth];

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int position = 0; position < static_cast<int>(level.size()); ++position) {
            const int target_index = level[static_cast<std::size_t>(position)];
            Node& target = nodes_[static_cast<std::size_t>(target_index)];
            if (target.mass <= 0.0 || target.parent < 0) {
                continue;
            }

            const Node& parent = nodes_[static_cast<std::size_t>(target.parent)];
            for (const int candidate_index : parent.near_nodes) {
                const Node& candidate = nodes_[static_cast<std::size_t>(candidate_index)];
                if (candidate.mass <= 0.0) {
                    continue;
                }

                if (is_leaf(candidate)) {
                    classify_node_interaction(target_index, candidate_index);
                    continue;
                }

                for (const int source_child_index : candidate.children) {
                    if (source_child_index >= 0) {
                        classify_node_interaction(target_index, source_child_index);
                    }
                }
            }

            std::sort(target.far_nodes.begin(), target.far_nodes.end());
            target.far_nodes.erase(
                std::unique(target.far_nodes.begin(), target.far_nodes.end()),
                target.far_nodes.end()
            );
            std::sort(target.near_nodes.begin(), target.near_nodes.end());
            target.near_nodes.erase(
                std::unique(target.near_nodes.begin(), target.near_nodes.end()),
                target.near_nodes.end()
            );
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int position = 0; position < static_cast<int>(leaf_indices_.size()); ++position) {
        const int target_leaf_index = leaf_indices_[static_cast<std::size_t>(position)];
        Node& leaf = nodes_[static_cast<std::size_t>(target_leaf_index)];
        for (const int near_node_index : leaf.near_nodes) {
            append_leaf_descendants(near_node_index, leaf.near_leaves);
        }
        std::sort(leaf.near_leaves.begin(), leaf.near_leaves.end());
        leaf.near_leaves.erase(
            std::unique(leaf.near_leaves.begin(), leaf.near_leaves.end()),
            leaf.near_leaves.end()
        );
    }

    for (const Node& node : nodes_) {
        stats_.far_interactions += node.far_nodes.size();
    }
    for (const int target_leaf_index : leaf_indices_) {
        stats_.near_interactions +=
            nodes_[static_cast<std::size_t>(target_leaf_index)].near_leaves.size();
    }
}

void FastMultipoleSolver::classify_node_interaction(int target_node_index, int source_node_index) {
    Node& target = nodes_[static_cast<std::size_t>(target_node_index)];
    const Node& source = nodes_[static_cast<std::size_t>(source_node_index)];

    if (source.mass <= 0.0) {
        return;
    }

    if (source_node_index != target_node_index && well_separated(target, source)) {
        target.far_nodes.push_back(source_node_index);
        return;
    }

    target.near_nodes.push_back(source_node_index);
}

void FastMultipoleSolver::append_leaf_descendants(
    int source_node_index,
    std::vector<int>& leaves
) const {
    const Node& source = nodes_[static_cast<std::size_t>(source_node_index)];
    if (source.mass <= 0.0) {
        return;
    }
    if (is_leaf(source)) {
        leaves.push_back(source_node_index);
        return;
    }
    for (const int child_index : source.children) {
        if (child_index >= 0) {
            append_leaf_descendants(child_index, leaves);
        }
    }
}

void FastMultipoleSolver::compute_local_expansions() {
    if (levels_.empty()) {
        return;
    }

    for (std::size_t depth = 1; depth < levels_.size(); ++depth) {
        const std::vector<int>& level = levels_[depth];

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int position = 0; position < static_cast<int>(level.size()); ++position) {
            const int node_index = level[static_cast<std::size_t>(position)];
            Node& node = nodes_[static_cast<std::size_t>(node_index)];
            if (node.mass <= 0.0) {
                continue;
            }

            if (node.parent >= 0) {
                const Node& parent = nodes_[static_cast<std::size_t>(node.parent)];
                add_local_to_local(node.local, parent.local);
            }

            for (const int source_node_index : node.far_nodes) {
                const Node& source = nodes_[static_cast<std::size_t>(source_node_index)];
                add_multipole_to_local(
                    node.local,
                    source.center_of_mass,
                    source.mass,
                    source.moments,
                    params_
                );
            }
        }
    }
}

Vec2 FastMultipoleSolver::evaluate_particle(std::size_t target_index, const Node& target_leaf) const {
    const Particle& target = (*particles_)[target_index];
    Vec2 acceleration = evaluate_local_acceleration(target_leaf.local, target.position);

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

FlatFmmData FastMultipoleSolver::export_flat_fmm() const {
    FlatFmmData flat;
    flat.tree.nodes.reserve(nodes_.size());

    for (const Node& node : nodes_) {
        FlatTreeNode flat_node;
        flat_node.center = node.center;
        flat_node.half_width = node.half_width;
        flat_node.mass = node.mass;
        flat_node.center_of_mass = node.center_of_mass;
        flat_node.moments = node.moments;
        flat_node.local = node.local;
        flat_node.children = node.children;
        flat_node.particle_begin = flat.tree.particle_indices.size();
        flat_node.particle_count = node.particle_indices.size();
        flat_node.is_leaf = is_leaf(node);
        flat.tree.particle_indices.insert(
            flat.tree.particle_indices.end(),
            node.particle_indices.begin(),
            node.particle_indices.end()
        );
        flat.tree.nodes.push_back(flat_node);
    }

    if (particles_ != nullptr) {
        flat.particle_leaf_indices.assign(particles_->size(), -1);
    }

    flat.leaves.reserve(leaf_indices_.size());
    for (std::size_t leaf_position = 0; leaf_position < leaf_indices_.size(); ++leaf_position) {
        const int leaf_node_index = leaf_indices_[leaf_position];
        const Node& leaf = nodes_[static_cast<std::size_t>(leaf_node_index)];

        FlatFmmLeaf flat_leaf;
        flat_leaf.node_index = leaf_node_index;
        flat_leaf.near_begin = flat.near_leaf_node_indices.size();
        flat_leaf.near_count = leaf.near_leaves.size();
        flat.near_leaf_node_indices.insert(
            flat.near_leaf_node_indices.end(),
            leaf.near_leaves.begin(),
            leaf.near_leaves.end()
        );

        for (const std::size_t particle_index : leaf.particle_indices) {
            if (particle_index < flat.particle_leaf_indices.size()) {
                flat.particle_leaf_indices[particle_index] = static_cast<int>(leaf_position);
            }
        }

        flat.leaves.push_back(flat_leaf);
    }

    return flat;
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

FlatFmmData build_flat_fmm(
    const std::vector<Particle>& particles,
    const PhysicsParams& params,
    FmmOptions options
) {
    FastMultipoleSolver solver(params, options);
    return solver.build_flat_fmm(particles);
}

}  // namespace fmmgalaxy
