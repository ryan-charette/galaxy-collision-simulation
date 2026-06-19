#pragma once

#include "mpi/distributed_solver.hpp"

#include <filesystem>
#include <string>

namespace fmmgalaxy {

/// @brief Build, source-control, hardware, and runtime context recorded with each run.
///
/// `SnapshotWriter` serializes this data into `metadata.json` so benchmark and validation
/// artifacts remain traceable to source revisions, compiler settings, MPI/CUDA state, and
/// the exact input configuration.
struct RunProvenance {
    /// Git commit hash or `unavailable` when not in a Git checkout.
    std::string git_commit{"unavailable"};
    /// Git branch name or `unavailable`.
    std::string git_branch{"unavailable"};
    /// Whether tracked source files had uncommitted changes.
    bool git_dirty{false};
    /// CMake build type.
    std::string build_type{"unknown"};
    /// Compiler identifier reported by CMake.
    std::string compiler{"unknown"};
    /// Compiler version reported by CMake.
    std::string compiler_version{"unknown"};
    /// Requested CMake MPI option.
    bool cmake_enable_mpi{false};
    /// Requested CMake CUDA option.
    bool cmake_enable_cuda{false};
    /// Whether a CUDA implementation is available at runtime.
    bool cuda_available{false};
    /// CUDA device name, when available.
    std::string cuda_device_name{};
    /// Whether this run is executing under MPI.
    bool mpi_enabled{false};
    /// Number of MPI ranks participating in the run.
    int rank_count{1};
    /// Hostname that produced the artifacts.
    std::string hostname{"unknown"};
    /// UTC timestamp when provenance was collected.
    std::string timestamp_utc{};
    /// Absolute path to the input config, or `builtin:default`.
    std::string config_path{"builtin:default"};
    /// SHA-256 hash of the input config when a config path is available.
    std::string config_sha256{};
};

/// Collect reproducibility metadata for the current process and optional config file.
RunProvenance collect_run_provenance(
    const std::filesystem::path* config_path,
    const MpiExecution& execution
);

/// Return the SHA-256 digest of a file as lowercase hexadecimal text.
std::string sha256_file(const std::filesystem::path& path);

}  // namespace fmmgalaxy
