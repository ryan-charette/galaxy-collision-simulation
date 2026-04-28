#include "core/simulation_info.hpp"
#include "build_config.hpp"

#include <iostream>

#if FMM_GALAXY_HAVE_MPI
#include <mpi.h>
#endif

#if FMM_GALAXY_HAVE_CUDA
extern "C" int cuda_smoke_value();
#endif

int main(int argc, char** argv) {
#if FMM_GALAXY_HAVE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << fmmgalaxy::build_summary();
        std::cout << "MPI world size: " << size << '\n';
    }

    MPI_Finalize();
#else
    (void)argc;
    (void)argv;
    std::cout << fmmgalaxy::build_summary();
#endif

#if FMM_GALAXY_HAVE_CUDA
    std::cout << "CUDA smoke value: " << cuda_smoke_value() << '\n';
#endif

    return 0;
}
