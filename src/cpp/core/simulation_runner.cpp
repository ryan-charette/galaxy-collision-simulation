#include "core/simulation_runner.hpp"

#include "core/diagnostics.hpp"
#include "core/initial_conditions.hpp"
#include "core/integrator.hpp"
#include "core/simulation_info.hpp"
#include "cuda/cuda_solver.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"
#include "io/snapshot_writer.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fmmgalaxy {

namespace {

enum class SolverKind {
    Direct,
    Tree,
    Fmm,
    CudaDirect,
    CudaTree,
    CudaFmm,
};

std::string unknown_solver_message(const std::string& solver) {
    return "Unknown solver '" + solver +
           "'. Use direct, tree, fmm, cuda-direct, cuda-tree, or cuda-fmm.";
}

SolverKind classify_solver(const std::string& solver) {
    if (solver == "direct") {
        return SolverKind::Direct;
    }
    if (solver == "tree" || solver == "treecode" || solver == "barnes-hut") {
        return SolverKind::Tree;
    }
    if (solver == "fmm" || solver == "monopole-fmm" || solver == "quadrupole-fmm" ||
        solver == "p4-fmm" || solver == "cartesian-fmm") {
        return SolverKind::Fmm;
    }
    if (solver == "cuda" || solver == "cuda-direct" || solver == "gpu-direct") {
        return SolverKind::CudaDirect;
    }
    if (solver == "cuda-tree" || solver == "gpu-tree" || solver == "cuda-barnes-hut") {
        return SolverKind::CudaTree;
    }
    if (solver == "cuda-fmm" || solver == "gpu-fmm") {
        return SolverKind::CudaFmm;
    }
    throw std::runtime_error(unknown_solver_message(solver));
}

bool is_cuda_solver(SolverKind solver) {
    return solver == SolverKind::CudaDirect ||
           solver == SolverKind::CudaTree ||
           solver == SolverKind::CudaFmm;
}

FmmOptions fmm_options_from_config(const SimulationConfig& config) {
    FmmOptions options;
    options.theta = config.tree_theta;
    options.leaf_capacity = config.tree_leaf_capacity;
    options.expansion_order = config.fmm_expansion_order;
    return options;
}

CudaTreeOptions cuda_options_from_config(const SimulationConfig& config) {
    CudaTreeOptions options;
    options.theta = config.tree_theta;
    options.leaf_capacity = config.tree_leaf_capacity;
    options.expansion_order = config.fmm_expansion_order;
    return options;
}

void print_run_header(const SimulationConfig& config, SolverKind solver, int mpi_ranks = 0) {
    std::cout << build_summary();
    std::cout << "Simulation: " << config.name << '\n';
    std::cout << "Solver:     " << config.solver << '\n';
    if (mpi_ranks > 0) {
        std::cout << "MPI ranks:  " << mpi_ranks << '\n';
    }
    if (is_cuda_solver(solver)) {
        std::cout << "CUDA available: "
                  << (cuda_solver_available() ? "yes" : "no, using CPU fallback")
                  << '\n';
    }
}

void compute_serial_accelerations(
    std::vector<Particle>& particles,
    const SimulationConfig& config,
    SolverKind solver
) {
    switch (solver) {
    case SolverKind::Direct:
        compute_direct_accelerations(particles, config.physics);
        return;
    case SolverKind::Tree:
        compute_tree_accelerations(
            particles,
            config.physics,
            config.tree_theta,
            config.tree_leaf_capacity,
            config.fmm_expansion_order
        );
        return;
    case SolverKind::Fmm:
        compute_fmm_accelerations(particles, config.physics, fmm_options_from_config(config));
        return;
    case SolverKind::CudaDirect:
        compute_cuda_direct_accelerations(particles, config.physics);
        return;
    case SolverKind::CudaTree:
        compute_cuda_tree_accelerations(particles, config.physics, cuda_options_from_config(config));
        return;
    case SolverKind::CudaFmm:
        compute_cuda_fmm_accelerations(particles, config.physics, cuda_options_from_config(config));
        return;
    }
    throw std::logic_error("Unhandled solver kind");
}

void compute_owned_accelerations(
    std::vector<Particle>& particles,
    const SimulationConfig& config,
    const OwnershipRange& owned,
    SolverKind solver
) {
    switch (solver) {
    case SolverKind::Direct:
        compute_direct_accelerations_for_targets(particles, config.physics, owned.begin, owned.end);
        return;
    case SolverKind::Tree:
    case SolverKind::Fmm:
        compute_fmm_accelerations_for_targets(
            particles,
            config.physics,
            owned.begin,
            owned.end,
            fmm_options_from_config(config)
        );
        return;
    case SolverKind::CudaDirect:
        compute_cuda_direct_accelerations(particles, config.physics);
        return;
    case SolverKind::CudaTree:
        compute_cuda_tree_accelerations(particles, config.physics, cuda_options_from_config(config));
        return;
    case SolverKind::CudaFmm:
        compute_cuda_fmm_accelerations(particles, config.physics, cuda_options_from_config(config));
        return;
    }
    throw std::logic_error("Unhandled solver kind");
}

void write_step_outputs(
    SnapshotWriter& writer,
    const SimulationConfig& config,
    const std::vector<Particle>& particles,
    int step,
    double time
) {
    writer.write_accelerations(step, time, particles);
    if (config.output.format == OutputFormat::None) {
        std::cout << "step " << step << " time " << time << " output disabled\n";
        return;
    }
    const auto diagnostics = compute_diagnostics(particles, config.physics);
    writer.write_snapshot(step, time, particles);
    writer.write_diagnostics(step, time, diagnostics, particles.size());
    std::cout << "step " << step << " time " << time
              << " total_energy " << diagnostics.total_energy << '\n';
}

std::vector<Particle> generated_particles_or_throw(const SimulationConfig& config) {
    std::vector<Particle> particles = generate_galaxies(config.galaxies, config.physics, config.seed);
    if (particles.empty()) {
        throw std::runtime_error("No particles were generated. Check galaxy n_particles values.");
    }
    return particles;
}

void run_serial_simulation(
    const SimulationConfig& config,
    const RunProvenance& provenance,
    SolverKind solver
) {
    print_run_header(config, solver);
    std::vector<Particle> particles = generated_particles_or_throw(config);

    SnapshotWriter writer(config);
    writer.write_metadata(config, particles.size(), provenance);

    compute_serial_accelerations(particles, config, solver);
    write_step_outputs(writer, config, particles, 0, 0.0);

    for (int step = 1; step <= config.steps; ++step) {
        if (solver == SolverKind::CudaDirect) {
            cuda_direct_leapfrog_step(particles, config.dt, config.physics);
        } else if (solver == SolverKind::CudaTree) {
            cuda_tree_leapfrog_step(
                particles,
                config.dt,
                config.physics,
                cuda_options_from_config(config)
            );
        } else if (solver == SolverKind::CudaFmm) {
            cuda_fmm_leapfrog_step(
                particles,
                config.dt,
                config.physics,
                cuda_options_from_config(config)
            );
        } else {
            auto compute_accelerations = [&config, solver](std::vector<Particle>& state) {
                compute_serial_accelerations(state, config, solver);
            };
            leapfrog_step(particles, config.dt, compute_accelerations);
        }

        if (step % config.snapshot_every == 0 || step == config.steps) {
            write_step_outputs(writer, config, particles, step, static_cast<double>(step) * config.dt);
        }
    }

    if (config.output.format != OutputFormat::None) {
        std::cout << "Wrote snapshots to " << config.output.directory.string() << '\n';
    }
}

void run_distributed_simulation(
    const SimulationConfig& config,
    const MpiExecution& execution,
    const RunProvenance& provenance,
    SolverKind solver
) {
    if (execution.rank == 0) {
        print_run_header(config, solver, execution.size);
    }

    std::vector<Particle> particles = generated_particles_or_throw(config);
    const OwnershipRange owned = ownership_for_rank(particles.size(), execution.rank, execution.size);

    if (execution.rank == 0) {
        std::cout << "Particle count: " << particles.size() << '\n';
    }

    std::unique_ptr<SnapshotWriter> writer;
    if (execution.rank == 0) {
        writer = std::make_unique<SnapshotWriter>(config);
        writer->write_metadata(config, particles.size(), provenance);
    }

    auto write_outputs = [&](int step, double time) {
        if (execution.rank == 0) {
            write_step_outputs(*writer, config, particles, step, time);
        }
    };

    compute_owned_accelerations(particles, config, owned, solver);
    mpi_synchronize_particles(particles, owned);
    write_outputs(0, 0.0);

    for (int step = 1; step <= config.steps; ++step) {
        kick(particles, owned.begin, owned.end, 0.5 * config.dt);
        drift(particles, owned.begin, owned.end, config.dt);
        mpi_synchronize_particles(particles, owned);

        compute_owned_accelerations(particles, config, owned, solver);

        kick(particles, owned.begin, owned.end, 0.5 * config.dt);
        mpi_synchronize_particles(particles, owned);

        if (step % config.snapshot_every == 0 || step == config.steps) {
            write_outputs(step, static_cast<double>(step) * config.dt);
        }
    }

    if (execution.rank == 0 && config.output.format != OutputFormat::None) {
        std::cout << "Wrote snapshots to " << config.output.directory.string() << '\n';
    }
}

}  // namespace

void run_configured_simulation(
    const SimulationConfig& config,
    const MpiExecution& execution,
    const RunProvenance& provenance
) {
    const SolverKind solver = classify_solver(config.solver);
    if (execution.enabled && execution.size > 1) {
        run_distributed_simulation(config, execution, provenance, solver);
    } else {
        run_serial_simulation(config, provenance, solver);
    }
}

}  // namespace fmmgalaxy
