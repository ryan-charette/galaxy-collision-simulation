#include "build_config.hpp"
#include "core/config.hpp"
#include "core/provenance.hpp"
#include "core/simulation_runner.hpp"
#include "mpi/distributed_solver.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#if FMM_GALAXY_HAVE_MPI
#include <mpi.h>
#endif

namespace {

struct CliOptions {
    std::filesystem::path config_path{};
    std::filesystem::path output_directory{};
    bool has_config{false};
    bool has_output_directory{false};
    bool show_help{false};
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [--config path] [--output directory]\n\n"
        << "Runs the 2D/3D softened-gravity galaxy collision simulator.\n"
        << "If --config is omitted, a small built-in two-galaxy smoke config is used.\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
        } else if (arg == "--config" && i + 1 < argc) {
            options.config_path = argv[++i];
            options.has_config = true;
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_directory = argv[++i];
            options.has_output_directory = true;
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
#if FMM_GALAXY_HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    const fmmgalaxy::MpiExecution mpi = fmmgalaxy::mpi_execution();

    int exit_code = 0;
    try {
        const CliOptions options = parse_args(argc, argv);
        if (options.show_help) {
            if (mpi.rank == 0) {
                print_usage(argv[0]);
            }
        } else {
            fmmgalaxy::SimulationConfig config =
                options.has_config ? fmmgalaxy::load_config(options.config_path) : fmmgalaxy::default_config();
            if (options.has_output_directory) {
                config.output.directory = options.output_directory;
            }
            const auto provenance = fmmgalaxy::collect_run_provenance(
                options.has_config ? &options.config_path : nullptr,
                mpi
            );
            fmmgalaxy::run_configured_simulation(config, mpi, provenance);
        }
    } catch (const std::exception& error) {
        if (mpi.rank == 0) {
            std::cerr << "error: " << error.what() << '\n';
        }
        exit_code = 1;
    }

#if FMM_GALAXY_HAVE_MPI
    MPI_Finalize();
#endif

    return exit_code;
}
