#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"
#include "fmm/flat_tree.hpp"
#include "fmm/multipole.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace fmmgalaxy {

/// @brief Barnes-Hut treecode solver using the shared octree geometry.
///
/// The solver approximates far cells using monopole or Cartesian multipole terms depending
/// on `expansion_order`, and falls back to particle-particle interactions for nearby leaves.
class BarnesHutSolver {
public:
    /// Construct a tree solver with physics and opening-criterion controls.
    BarnesHutSolver(
        PhysicsParams params,
        double theta = 0.6,
        std::size_t leaf_capacity = 16,
        int max_depth = 32,
        int expansion_order = 4
    );

    /// Build the tree and update every particle acceleration in place.
    void compute(std::vector<Particle>& particles);
    /// Build and export flattened tree data for CUDA evaluation paths.
    FlatTreeData build_flat_tree(const std::vector<Particle>& particles);

private:
    struct Node {
        Vec2 center{};
        double half_width{1.0};
        double mass{0.0};
        Vec2 center_of_mass{};
        CartesianMoments moments{};
        std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
        std::vector<std::size_t> particle_indices{};
    };

    const std::vector<Particle>* particles_{nullptr};
    PhysicsParams params_{};
    double theta_{0.6};
    std::size_t leaf_capacity_{16};
    int max_depth_{32};
    int expansion_order_{0};
    std::vector<Node> nodes_{};

    void build(const std::vector<Particle>& particles);
    void insert_particle(int node_index, std::size_t particle_index, int depth);
    void subdivide(int node_index);
    double compute_moments(int node_index);
    void compute_multipole_moments(int node_index);
    Vec2 accumulate_from_node(int node_index, std::size_t target_index, const Vec2& target_position) const;
    FlatTreeData export_flat_tree() const;
    bool is_leaf(const Node& node) const;
    bool contains(const Node& node, const Vec2& position) const;
};

/// Convenience wrapper that computes Barnes-Hut accelerations for all particles.
void compute_tree_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    double theta = 0.6,
    std::size_t leaf_capacity = 16,
    int expansion_order = 4
);

/// Build flattened Barnes-Hut tree data without computing particle accelerations.
FlatTreeData build_flat_tree(
    const std::vector<Particle>& particles,
    const PhysicsParams& params,
    double theta = 0.6,
    std::size_t leaf_capacity = 16,
    int max_depth = 32,
    int expansion_order = 4
);

}  // namespace fmmgalaxy
