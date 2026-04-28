#pragma once

#include "core/initial_conditions.hpp"
#include "core/simulation_params.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fmmgalaxy {

struct OutputConfig {
    std::filesystem::path directory{"experiments/validation/smoke_test"};
    std::string format{"csv"};
};

struct SimulationConfig {
    std::string name{"smoke_test"};
    std::string solver{"direct"};
    int dim{3};
    std::uint64_t seed{42};
    std::size_t n_particles{0};
    int steps{40};
    double dt{0.01};
    int snapshot_every{10};
    double tree_theta{0.6};
    std::size_t tree_leaf_capacity{16};
    int fmm_expansion_order{2};
    PhysicsParams physics{};
    std::vector<GalaxyConfig> galaxies{};
    OutputConfig output{};
};

SimulationConfig default_config();
SimulationConfig load_config(const std::filesystem::path& path);

}  // namespace fmmgalaxy
