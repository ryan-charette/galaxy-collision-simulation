#include "core/simulation_info.hpp"
#include "build_config.hpp"

#include <sstream>

namespace fmmgalaxy {

std::string build_summary() {
    std::ostringstream out;
    out << "Distributed FMM Galaxy Simulator v" << FMM_GALAXY_VERSION << '\n';
    out << "MPI:  " << (FMM_GALAXY_HAVE_MPI ? "enabled" : "disabled") << '\n';
    out << "CUDA: " << (FMM_GALAXY_HAVE_CUDA ? "enabled" : "disabled") << '\n';
    return out.str();
}

}  // namespace fmmgalaxy
