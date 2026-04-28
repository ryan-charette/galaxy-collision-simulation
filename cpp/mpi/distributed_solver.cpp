#include "mpi/distributed_solver.hpp"

#include "build_config.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#if FMM_GALAXY_HAVE_MPI
#include <mpi.h>
#endif

namespace fmmgalaxy {

namespace {

constexpr int fields_per_particle = 11;

void pack_particle(const Particle& particle, double* out) {
    out[0] = particle.position.x;
    out[1] = particle.position.y;
    out[2] = particle.position.z;
    out[3] = particle.velocity.x;
    out[4] = particle.velocity.y;
    out[5] = particle.velocity.z;
    out[6] = particle.acceleration.x;
    out[7] = particle.acceleration.y;
    out[8] = particle.acceleration.z;
    out[9] = particle.mass;
    out[10] = static_cast<double>(particle.group_id);
}

Particle unpack_particle(const double* data) {
    Particle particle;
    particle.position = {data[0], data[1], data[2]};
    particle.velocity = {data[3], data[4], data[5]};
    particle.acceleration = {data[6], data[7], data[8]};
    particle.mass = data[9];
    particle.group_id = static_cast<std::uint32_t>(data[10]);
    return particle;
}

int checked_int(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds MPI int count limit");
    }
    return static_cast<int>(value);
}

}  // namespace

MpiExecution mpi_execution() {
    MpiExecution execution;
#if FMM_GALAXY_HAVE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        execution.enabled = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &execution.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &execution.size);
    }
#endif
    return execution;
}

OwnershipRange ownership_for_rank(std::size_t particle_count, int rank, int size) {
    if (size <= 1) {
        return {0, particle_count};
    }

    const std::size_t base = particle_count / static_cast<std::size_t>(size);
    const std::size_t remainder = particle_count % static_cast<std::size_t>(size);
    const std::size_t rank_index = static_cast<std::size_t>(rank);
    const std::size_t begin = rank_index * base + std::min(rank_index, remainder);
    const std::size_t count = base + (rank_index < remainder ? 1 : 0);
    return {begin, begin + count};
}

void mpi_synchronize_particles(std::vector<Particle>& particles, const OwnershipRange& owned) {
#if FMM_GALAXY_HAVE_MPI
    const MpiExecution execution = mpi_execution();
    if (!execution.enabled || execution.size <= 1) {
        return;
    }

    const std::size_t local_count = owned.end > owned.begin ? owned.end - owned.begin : 0;
    std::vector<double> send_buffer(local_count * fields_per_particle);
    for (std::size_t local = 0; local < local_count; ++local) {
        pack_particle(particles[owned.begin + local], send_buffer.data() + local * fields_per_particle);
    }

    const int send_count = checked_int(send_buffer.size(), "local particle buffer");
    std::vector<int> receive_counts(static_cast<std::size_t>(execution.size), 0);
    MPI_Allgather(
        &send_count,
        1,
        MPI_INT,
        receive_counts.data(),
        1,
        MPI_INT,
        MPI_COMM_WORLD
    );

    std::vector<int> displacements(static_cast<std::size_t>(execution.size), 0);
    int total_receive_count = 0;
    for (int rank = 0; rank < execution.size; ++rank) {
        displacements[static_cast<std::size_t>(rank)] = total_receive_count;
        total_receive_count += receive_counts[static_cast<std::size_t>(rank)];
    }

    std::vector<double> receive_buffer(static_cast<std::size_t>(total_receive_count));
    MPI_Allgatherv(
        send_buffer.data(),
        send_count,
        MPI_DOUBLE,
        receive_buffer.data(),
        receive_counts.data(),
        displacements.data(),
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    std::size_t particle_index = 0;
    for (int rank = 0; rank < execution.size; ++rank) {
        const int offset = displacements[static_cast<std::size_t>(rank)];
        const int count = receive_counts[static_cast<std::size_t>(rank)];
        if (count % fields_per_particle != 0) {
            throw std::runtime_error("Received invalid MPI particle buffer size");
        }

        const int rank_particle_count = count / fields_per_particle;
        for (int local = 0; local < rank_particle_count; ++local) {
            if (particle_index >= particles.size()) {
                throw std::runtime_error("Received more MPI particles than expected");
            }
            particles[particle_index++] =
                unpack_particle(receive_buffer.data() + offset + local * fields_per_particle);
        }
    }

    if (particle_index != particles.size()) {
        throw std::runtime_error("Received fewer MPI particles than expected");
    }
#else
    (void)particles;
    (void)owned;
#endif
}

}  // namespace fmmgalaxy
