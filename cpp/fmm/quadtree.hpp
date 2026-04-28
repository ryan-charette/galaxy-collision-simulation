#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "core/vector2.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace fmmgalaxy {

class BarnesHutSolver {
public:
    BarnesHutSolver(
        PhysicsParams params,
        double theta = 0.6,
        std::size_t leaf_capacity = 16,
        int max_depth = 32
    );

    void compute(std::vector<Particle>& particles);

private:
    struct Node {
        Vec2 center{};
        double half_width{1.0};
        double mass{0.0};
        Vec2 center_of_mass{};
        std::array<int, 4> children{{-1, -1, -1, -1}};
        std::vector<std::size_t> particle_indices{};
    };

    const std::vector<Particle>* particles_{nullptr};
    PhysicsParams params_{};
    double theta_{0.6};
    std::size_t leaf_capacity_{16};
    int max_depth_{32};
    std::vector<Node> nodes_{};

    void build(const std::vector<Particle>& particles);
    void insert_particle(int node_index, std::size_t particle_index, int depth);
    void subdivide(int node_index);
    double compute_moments(int node_index);
    Vec2 accumulate_from_node(int node_index, std::size_t target_index, const Vec2& target_position) const;
    bool is_leaf(const Node& node) const;
    bool contains(const Node& node, const Vec2& position) const;
    int child_index_for(const Node& node, const Vec2& position) const;
};

void compute_tree_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    double theta = 0.6,
    std::size_t leaf_capacity = 16
);

}  // namespace fmmgalaxy
