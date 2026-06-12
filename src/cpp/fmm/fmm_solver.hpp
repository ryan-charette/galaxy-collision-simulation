#pragma once

#include "core/particle.hpp"
#include "core/simulation_params.hpp"
#include "fmm/flat_tree.hpp"
#include "fmm/multipole.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace fmmgalaxy {

/// Public configuration knobs for the FMM solver.
struct FmmOptions {
    /// Opening angle used when building far/near interaction lists.
    double theta{0.6};
    /// Maximum particle count in a leaf node before subdivision.
    std::size_t leaf_capacity{16};
    /// Maximum tree subdivision depth.
    int max_depth{32};
    /// Cartesian expansion order; supported values are normalized to implemented orders.
    int expansion_order{4};
};

/// Runtime counters describing the most recent FMM tree and interaction lists.
struct FmmStats {
    /// Number of tree nodes.
    std::size_t node_count{0};
    /// Number of leaf nodes.
    std::size_t leaf_count{0};
    /// Number of accepted far-cell interactions.
    std::size_t far_interactions{0};
    /// Number of near-cell or near-leaf interactions.
    std::size_t near_interactions{0};
};

/// @brief Fast multipole solver for softened gravitational acceleration.
///
/// The implementation builds an octree, constructs Cartesian multipole moments, propagates
/// local expansions, and evaluates near interactions by direct particle-particle summation.
class FastMultipoleSolver {
public:
    /// Construct an FMM solver with physics and algorithm options.
    FastMultipoleSolver(PhysicsParams params, FmmOptions options = {});

    /// Build the FMM hierarchy and update every particle acceleration.
    void compute(std::vector<Particle>& particles);
    /// Compute accelerations for targets in the half-open range `[begin, end)`.
    void compute_targets(std::vector<Particle>& particles, std::size_t begin, std::size_t end);
    /// Build and export flattened FMM interaction data for CUDA evaluation paths.
    FlatFmmData build_flat_fmm(const std::vector<Particle>& particles);

    /// Return counters from the most recent FMM build/evaluation.
    const FmmStats& stats() const { return stats_; }

private:
    struct Node {
        Vec2 center{};
        double half_width{1.0};
        double mass{0.0};
        Vec2 center_of_mass{};
        CartesianMoments moments{};
        LocalExpansion local{};
        std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
        int parent{-1};
        int depth{0};
        std::vector<std::size_t> particle_indices{};
        std::vector<int> far_nodes{};
        std::vector<int> near_nodes{};
        std::vector<int> near_leaves{};
    };

    const std::vector<Particle>* particles_{nullptr};
    PhysicsParams params_{};
    FmmOptions options_{};
    std::vector<Node> nodes_{};
    std::vector<int> leaf_indices_{};
    std::vector<std::vector<int>> levels_{};
    FmmStats stats_{};

    void build(const std::vector<Particle>& particles);
    void insert_particle(int node_index, std::size_t particle_index, int depth);
    void subdivide(int node_index);
    void build_levels();
    void compute_masses_and_moments();
    void collect_leaves(int node_index);
    void build_interaction_lists();
    void classify_node_interaction(int target_node_index, int source_node_index);
    void append_leaf_descendants(int source_node_index, std::vector<int>& leaves) const;
    void compute_local_expansions();
    Vec2 evaluate_particle(std::size_t target_index, const Node& target_leaf) const;
    FlatFmmData export_flat_fmm() const;

    bool is_leaf(const Node& node) const;
    bool well_separated(const Node& target, const Node& source) const;
};

/// Convenience wrapper that computes FMM accelerations for all particles.
void compute_fmm_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    FmmOptions options = {}
);

/// Convenience wrapper that computes FMM accelerations for a target range.
void compute_fmm_accelerations_for_targets(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    std::size_t begin,
    std::size_t end,
    FmmOptions options = {}
);

/// Build flattened FMM data without computing particle accelerations.
FlatFmmData build_flat_fmm(
    const std::vector<Particle>& particles,
    const PhysicsParams& params,
    FmmOptions options = {}
);

}  // namespace fmmgalaxy
