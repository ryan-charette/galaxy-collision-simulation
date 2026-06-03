#pragma once

#include "core/particle.hpp"

#include <cstddef>
#include <vector>

namespace fmmgalaxy {

struct MpiExecution {
    bool enabled{false};
    int rank{0};
    int size{1};
};

struct OwnershipRange {
    std::size_t begin{0};
    std::size_t end{0};
};

MpiExecution mpi_execution();
OwnershipRange ownership_for_rank(std::size_t particle_count, int rank, int size);
void mpi_synchronize_particles(std::vector<Particle>& particles, const OwnershipRange& owned);

}  // namespace fmmgalaxy
