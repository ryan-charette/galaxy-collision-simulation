#pragma once

#include "mpi/distributed_solver.hpp"

#include <filesystem>
#include <string>

namespace fmmgalaxy {

struct RunProvenance {
    std::string git_commit{"unavailable"};
    std::string git_branch{"unavailable"};
    bool git_dirty{false};
    std::string build_type{"unknown"};
    std::string compiler{"unknown"};
    std::string compiler_version{"unknown"};
    bool cmake_enable_mpi{false};
    bool cmake_enable_cuda{false};
    bool cuda_available{false};
    std::string cuda_device_name{};
    bool mpi_enabled{false};
    int rank_count{1};
    std::string hostname{"unknown"};
    std::string timestamp_utc{};
    std::string config_path{"builtin:default"};
    std::string config_sha256{};
};

RunProvenance collect_run_provenance(
    const std::filesystem::path* config_path,
    const MpiExecution& execution
);

std::string sha256_file(const std::filesystem::path& path);

}  // namespace fmmgalaxy
