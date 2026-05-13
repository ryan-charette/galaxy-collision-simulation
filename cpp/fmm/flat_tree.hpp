#pragma once

#include "core/vector2.hpp"
#include "fmm/multipole.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace fmmgalaxy {

struct FlatTreeNode {
    Vec2 center{};
    double half_width{1.0};
    double mass{0.0};
    Vec2 center_of_mass{};
    CartesianMoments moments{};
    std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
    std::size_t particle_begin{0};
    std::size_t particle_count{0};
    bool is_leaf{true};
};

struct FlatTreeData {
    std::vector<FlatTreeNode> nodes{};
    std::vector<std::size_t> particle_indices{};
};

struct FlatFmmLeaf {
    int node_index{-1};
    std::size_t far_begin{0};
    std::size_t far_count{0};
    std::size_t near_begin{0};
    std::size_t near_count{0};
};

struct FlatFmmData {
    FlatTreeData tree{};
    std::vector<FlatFmmLeaf> leaves{};
    std::vector<int> far_node_indices{};
    std::vector<int> near_leaf_node_indices{};
    std::vector<int> particle_leaf_indices{};
};

}  // namespace fmmgalaxy
