#pragma once

#include "core/vector2.hpp"
#include "fmm/multipole.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace fmmgalaxy {

/// Flattened tree node used to transfer CPU-built tree data to CUDA kernels.
struct FlatTreeNode {
    /// Cell center.
    Vec2 center{};
    /// Half-width of the cubic cell.
    double half_width{1.0};
    /// Total source mass in the cell.
    double mass{0.0};
    /// Mass-weighted center of mass.
    Vec2 center_of_mass{};
    /// Multipole coefficients for the cell.
    CartesianMoments moments{};
    /// Local expansion coefficients for FMM export.
    LocalExpansion local{};
    /// Child node indices, with `-1` for missing children.
    std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
    /// Start offset into `FlatTreeData::particle_indices`.
    std::size_t particle_begin{0};
    /// Number of particle indices belonging to this node.
    std::size_t particle_count{0};
    /// Whether this node is a leaf.
    bool is_leaf{true};
};

/// Contiguous tree representation for GPU upload and diagnostics.
struct FlatTreeData {
    /// Flattened tree nodes in CPU construction order.
    std::vector<FlatTreeNode> nodes{};
    /// Particle indices grouped by leaf/node ranges.
    std::vector<std::size_t> particle_indices{};
};

/// Flattened FMM leaf metadata with offsets into near-leaf interaction lists.
struct FlatFmmLeaf {
    /// Index of the corresponding tree node.
    int node_index{-1};
    /// Start offset into `FlatFmmData::near_leaf_node_indices`.
    std::size_t near_begin{0};
    /// Number of near leaves for this leaf.
    std::size_t near_count{0};
};

/// Flattened FMM tree, leaf ownership, and near-interaction data.
struct FlatFmmData {
    /// Shared flattened tree representation.
    FlatTreeData tree{};
    /// Leaf metadata records.
    std::vector<FlatFmmLeaf> leaves{};
    /// Concatenated near-leaf node indices.
    std::vector<int> near_leaf_node_indices{};
    /// Leaf index for each particle, used by target-evaluation kernels.
    std::vector<int> particle_leaf_indices{};
};

}  // namespace fmmgalaxy
