#pragma once

#include "core/particle.hpp"

#include <cstddef>
#include <vector>

namespace fmmgalaxy {

/// MPI runtime state used by the simulation runner and provenance metadata.
struct MpiExecution {
    /// Whether MPI was initialized and should be used.
    bool enabled{false};
    /// Rank of the current process.
    int rank{0};
    /// Number of participating ranks.
    int size{1};
};

/// Half-open particle ownership range `[begin, end)` for one MPI rank.
struct OwnershipRange {
    /// First owned particle index.
    std::size_t begin{0};
    /// One-past-the-end owned particle index.
    std::size_t end{0};
};

/// Detect MPI execution context, falling back to a serial rank when MPI is unavailable.
MpiExecution mpi_execution();
/// Return the contiguous ownership range for a rank.
OwnershipRange ownership_for_rank(std::size_t particle_count, int rank, int size);
/// Synchronize particle state across MPI ranks after owned particles have been updated.
void mpi_synchronize_particles(std::vector<Particle>& particles, const OwnershipRange& owned);

}  // namespace fmmgalaxy
