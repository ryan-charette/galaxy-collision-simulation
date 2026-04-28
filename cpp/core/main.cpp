#include "build_config.hpp"
#include "core/config.hpp"
#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/simulation_info.hpp"
#include "cuda/cuda_solver.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"
#include "io/snapshot_writer.hpp"
#include "mpi/distributed_solver.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
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

bool uses_tree_solver(const std::string& solver) {
    return solver == "tree" || solver == "treecode" || solver == "barnes-hut";
}

bool uses_fmm_solver(const std::string& solver) {
    return solver == "fmm" || solver == "monopole-fmm" || solver == "quadrupole-fmm" ||
           solver == "p4-fmm" || solver == "cartesian-fmm";
}

bool uses_cuda_direct_solver(const std::string& solver) {
    return solver == "cuda" || solver == "cuda-direct" || solver == "gpu-direct";
}

fmmgalaxy::FmmOptions fmm_options_from_config(const fmmgalaxy::SimulationConfig& config) {
    fmmgalaxy::FmmOptions options;
    options.theta = config.tree_theta;
    options.leaf_capacity = config.tree_leaf_capacity;
    options.expansion_order = config.fmm_expansion_order;
    return options;
}

void compute_serial_accelerations(
    std::vector<fmmgalaxy::Particle>& particles,
    const fmmgalaxy::SimulationConfig& config
) {
    if (config.solver == "direct") {
        fmmgalaxy::compute_direct_accelerations(particles, config.physics);
    } else if (uses_tree_solver(config.solver)) {
        fmmgalaxy::compute_tree_accelerations(
            particles,
            config.physics,
            config.tree_theta,
            config.tree_leaf_capacity,
            config.fmm_expansion_order
        );
    } else if (uses_fmm_solver(config.solver)) {
        fmmgalaxy::compute_fmm_accelerations(
            particles,
            config.physics,
            fmm_options_from_config(config)
        );
    } else if (uses_cuda_direct_solver(config.solver)) {
        fmmgalaxy::compute_cuda_direct_accelerations(particles, config.physics);
    } else {
        throw std::runtime_error(
            "Unknown solver '" + config.solver + "'. Use direct, tree, fmm, or cuda-direct."
        );
    }
}

void compute_owned_accelerations(
    std::vector<fmmgalaxy::Particle>& particles,
    const fmmgalaxy::SimulationConfig& config,
    const fmmgalaxy::OwnershipRange& owned
) {
    if (config.solver == "direct") {
        fmmgalaxy::compute_direct_accelerations_for_targets(
            particles,
            config.physics,
            owned.begin,
            owned.end
        );
    } else if (uses_tree_solver(config.solver) || uses_fmm_solver(config.solver)) {
        fmmgalaxy::compute_fmm_accelerations_for_targets(
            particles,
            config.physics,
            owned.begin,
            owned.end,
            fmm_options_from_config(config)
        );
    } else if (uses_cuda_direct_solver(config.solver)) {
        fmmgalaxy::compute_cuda_direct_accelerations(particles, config.physics);
    } else {
        throw std::runtime_error(
            "Unknown solver '" + config.solver + "'. Use direct, tree, fmm, or cuda-direct."
        );
    }
}

void run_simulation(const fmmgalaxy::SimulationConfig& config) {
    std::cout << fmmgalaxy::build_summary();
    std::cout << "Simulation: " << config.name << '\n';
    std::cout << "Solver:     " << config.solver << '\n';
    if (uses_cuda_direct_solver(config.solver)) {
        std::cout << "CUDA direct available: "
                  << (fmmgalaxy::cuda_solver_available() ? "yes" : "no, using CPU fallback")
                  << '\n';
    }

    std::vector<fmmgalaxy::Particle> particles =
        fmmgalaxy::generate_galaxies(config.galaxies, config.physics, config.seed);

    if (particles.empty()) {
        throw std::runtime_error("No particles were generated. Check galaxy n_particles values.");
    }

    fmmgalaxy::SnapshotWriter writer(config);
    writer.write_metadata(config, particles.size());

    compute_serial_accelerations(particles, config);

    auto write_outputs = [&](int step, double time) {
        const auto diagnostics = fmmgalaxy::compute_diagnostics(particles, config.physics);
        writer.write_snapshot(step, time, particles);
        writer.write_diagnostics(step, time, diagnostics, particles.size());
        std::cout << "step " << step << " time " << time
                  << " total_energy " << diagnostics.total_energy << '\n';
    };

    write_outputs(0, 0.0);
    for (int step = 1; step <= config.steps; ++step) {
        if (uses_cuda_direct_solver(config.solver)) {
            fmmgalaxy::cuda_direct_leapfrog_step(particles, config.dt, config.physics);
        } else {
            auto compute_accelerations = [&config](std::vector<fmmgalaxy::Particle>& state) {
                compute_serial_accelerations(state, config);
            };
            fmmgalaxy::leapfrog_step(particles, config.dt, compute_accelerations);
        }
        if (step % config.snapshot_every == 0 || step == config.steps) {
            write_outputs(step, static_cast<double>(step) * config.dt);
        }
    }

    std::cout << "Wrote snapshots to " << config.output.directory.string() << '\n';
}

void run_distributed_simulation(
    const fmmgalaxy::SimulationConfig& config,
    const fmmgalaxy::MpiExecution& execution
) {
    if (execution.rank == 0) {
        std::cout << fmmgalaxy::build_summary();
        std::cout << "Simulation: " << config.name << '\n';
        std::cout << "Solver:     " << config.solver << '\n';
        std::cout << "MPI ranks:  " << execution.size << '\n';
        if (uses_cuda_direct_solver(config.solver)) {
            std::cout << "CUDA direct available: "
                      << (fmmgalaxy::cuda_solver_available() ? "yes" : "no, using CPU fallback")
                      << '\n';
        }
    }

    std::vector<fmmgalaxy::Particle> particles =
        fmmgalaxy::generate_galaxies(config.galaxies, config.physics, config.seed);

    if (particles.empty()) {
        throw std::runtime_error("No particles were generated. Check galaxy n_particles values.");
    }

    const fmmgalaxy::OwnershipRange owned =
        fmmgalaxy::ownership_for_rank(particles.size(), execution.rank, execution.size);

    if (execution.rank == 0) {
        std::cout << "Particle count: " << particles.size() << '\n';
    }

    std::unique_ptr<fmmgalaxy::SnapshotWriter> writer;
    if (execution.rank == 0) {
        writer = std::make_unique<fmmgalaxy::SnapshotWriter>(config);
        writer->write_metadata(config, particles.size());
    }

    auto write_outputs = [&](int step, double time) {
        if (execution.rank != 0) {
            return;
        }
        const auto diagnostics = fmmgalaxy::compute_diagnostics(particles, config.physics);
        writer->write_snapshot(step, time, particles);
        writer->write_diagnostics(step, time, diagnostics, particles.size());
        std::cout << "step " << step << " time " << time
                  << " total_energy " << diagnostics.total_energy << '\n';
    };

    compute_owned_accelerations(particles, config, owned);
    fmmgalaxy::mpi_synchronize_particles(particles, owned);
    write_outputs(0, 0.0);

    for (int step = 1; step <= config.steps; ++step) {
        fmmgalaxy::kick(particles, owned.begin, owned.end, 0.5 * config.dt);
        fmmgalaxy::drift(particles, owned.begin, owned.end, config.dt);
        fmmgalaxy::mpi_synchronize_particles(particles, owned);

        compute_owned_accelerations(particles, config, owned);

        fmmgalaxy::kick(particles, owned.begin, owned.end, 0.5 * config.dt);
        fmmgalaxy::mpi_synchronize_particles(particles, owned);

        if (step % config.snapshot_every == 0 || step == config.steps) {
            write_outputs(step, static_cast<double>(step) * config.dt);
        }
    }

    if (execution.rank == 0) {
        std::cout << "Wrote snapshots to " << config.output.directory.string() << '\n';
    }
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
            if (mpi.enabled && mpi.size > 1) {
                run_distributed_simulation(config, mpi);
            } else {
                run_simulation(config);
            }
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
