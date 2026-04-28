#include "build_config.hpp"
#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/simulation_info.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/quadtree.hpp"
#include "io/snapshot_writer.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
        << "Runs the 2D softened-gravity galaxy collision simulator.\n"
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

bool uses_tree_solver(const std::string& solver) {
    return solver == "tree" || solver == "treecode" || solver == "barnes-hut" || solver == "fmm";
}

void run_simulation(const fmmgalaxy::SimulationConfig& config) {
    std::cout << fmmgalaxy::build_summary();
    std::cout << "Simulation: " << config.name << '\n';
    std::cout << "Solver:     " << config.solver << '\n';

    std::vector<fmmgalaxy::Particle> particles =
        fmmgalaxy::generate_galaxies(config.galaxies, config.physics, config.seed);

    if (particles.empty()) {
        throw std::runtime_error("No particles were generated. Check galaxy n_particles values.");
    }

    fmmgalaxy::AccelerationFunction compute_accelerations;
    if (config.solver == "direct") {
        compute_accelerations = [&config](std::vector<fmmgalaxy::Particle>& state) {
            fmmgalaxy::compute_direct_accelerations(state, config.physics);
        };
    } else if (uses_tree_solver(config.solver)) {
        compute_accelerations = [&config](std::vector<fmmgalaxy::Particle>& state) {
            fmmgalaxy::compute_tree_accelerations(
                state,
                config.physics,
                config.tree_theta,
                config.tree_leaf_capacity
            );
        };
    } else {
        throw std::runtime_error("Unknown solver '" + config.solver + "'. Use direct or tree.");
    }

    fmmgalaxy::SnapshotWriter writer(config);
    writer.write_metadata(config, particles.size());

    compute_accelerations(particles);

    auto write_outputs = [&](int step, double time) {
        const auto diagnostics = fmmgalaxy::compute_diagnostics(particles, config.physics);
        writer.write_snapshot(step, time, particles);
        writer.write_diagnostics(step, time, diagnostics, particles.size());
        std::cout << "step " << step << " time " << time
                  << " total_energy " << diagnostics.total_energy << '\n';
    };

    write_outputs(0, 0.0);
    for (int step = 1; step <= config.steps; ++step) {
        fmmgalaxy::leapfrog_step(particles, config.dt, compute_accelerations);
        if (step % config.snapshot_every == 0 || step == config.steps) {
            write_outputs(step, static_cast<double>(step) * config.dt);
        }
    }

    std::cout << "Wrote snapshots to " << config.output.directory.string() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
#if FMM_GALAXY_HAVE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    int exit_code = 0;
    try {
        const CliOptions options = parse_args(argc, argv);
        if (options.show_help) {
            if (
#if FMM_GALAXY_HAVE_MPI
                rank == 0
#else
                true
#endif
            ) {
                print_usage(argv[0]);
            }
        } else {
#if FMM_GALAXY_HAVE_MPI
            if (size > 1 && rank != 0) {
                MPI_Finalize();
                return 0;
            }
            if (size > 1) {
                std::cout << "MPI is linked, but this MVP simulation runner executes on rank 0 only.\n";
            }
#endif
            fmmgalaxy::SimulationConfig config =
                options.has_config ? fmmgalaxy::load_config(options.config_path) : fmmgalaxy::default_config();
            if (options.has_output_directory) {
                config.output.directory = options.output_directory;
            }
            run_simulation(config);
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        exit_code = 1;
    }

#if FMM_GALAXY_HAVE_MPI
    MPI_Finalize();
#endif

    return exit_code;
}
