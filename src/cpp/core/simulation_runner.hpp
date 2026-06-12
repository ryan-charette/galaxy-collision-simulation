#pragma once

#include "core/config.hpp"
#include "core/provenance.hpp"
#include "mpi/distributed_solver.hpp"

namespace fmmgalaxy {

/// Run the configured simulation using the selected serial, MPI, CUDA, or CPU solver path.
void run_configured_simulation(
    const SimulationConfig& config,
    const MpiExecution& execution,
    const RunProvenance& provenance
);

}  // namespace fmmgalaxy
