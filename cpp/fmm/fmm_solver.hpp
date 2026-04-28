#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace fmmgalaxy {

struct FmmOptions {
    double theta{0.6};
    std::size_t leaf_capacity{16};
    int max_depth{32};
    int expansion_order{0};
};

struct FmmStats {
    std::size_t node_count{0};
    std::size_t leaf_count{0};
    std::size_t far_interactions{0};
    std::size_t near_interactions{0};
};

class FastMultipoleSolver {
public:
    FastMultipoleSolver(PhysicsParams params, FmmOptions options = {});

    void compute(std::vector<Particle>& particles);
    void compute_targets(std::vector<Particle>& particles, std::size_t begin, std::size_t end);

    const FmmStats& stats() const { return stats_; }

private:
    struct Node {
        Vec2 center{};
        double half_width{1.0};
        double mass{0.0};
        Vec2 center_of_mass{};
        std::array<int, 4> children{{-1, -1, -1, -1}};
        std::vector<std::size_t> particle_indices{};
        std::vector<int> far_nodes{};
        std::vector<int> near_leaves{};
    };

    const std::vector<Particle>* particles_{nullptr};
    PhysicsParams params_{};
    FmmOptions options_{};
    std::vector<Node> nodes_{};
    std::vector<int> leaf_indices_{};
    FmmStats stats_{};

    void build(const std::vector<Particle>& particles);
    void insert_particle(int node_index, std::size_t particle_index, int depth);
    void subdivide(int node_index);
    double p2m_m2m(int node_index);
    void collect_leaves(int node_index);
    void build_interaction_lists();
    void build_leaf_interactions(int target_leaf_index, int source_node_index);
    Vec2 evaluate_particle(std::size_t target_index, const Node& target_leaf) const;

    bool is_leaf(const Node& node) const;
    bool well_separated(const Node& target, const Node& source) const;
    int child_index_for(const Node& node, const Vec2& position) const;
};

void compute_fmm_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    FmmOptions options = {}
);

void compute_fmm_accelerations_for_targets(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    std::size_t begin,
    std::size_t end,
    FmmOptions options = {}
);

}  // namespace fmmgalaxy
