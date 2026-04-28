#include "cuda/cuda_solver.hpp"

#include "core/integrator.hpp"
#include "direct/direct_solver.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace fmmgalaxy {

namespace {

struct DeviceParticle {
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    double ax;
    double ay;
    double az;
    double mass;
    unsigned int group_id;
};

DeviceParticle pack_particle(const Particle& particle) {
    return DeviceParticle{
        particle.position.x,
        particle.position.y,
        particle.position.z,
        particle.velocity.x,
        particle.velocity.y,
        particle.velocity.z,
        particle.acceleration.x,
        particle.acceleration.y,
        particle.acceleration.z,
        particle.mass,
        particle.group_id,
    };
}

void unpack_particle(const DeviceParticle& device_particle, Particle& particle) {
    particle.position = {device_particle.x, device_particle.y, device_particle.z};
    particle.velocity = {device_particle.vx, device_particle.vy, device_particle.vz};
    particle.acceleration = {device_particle.ax, device_particle.ay, device_particle.az};
    particle.mass = device_particle.mass;
    particle.group_id = device_particle.group_id;
}

void throw_on_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(status)
        );
    }
}

__global__ void direct_acceleration_kernel(
    DeviceParticle* particles,
    int count,
    double gravitational_constant,
    double softening
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const double xi = particles[i].x;
    const double yi = particles[i].y;
    const double zi = particles[i].z;
    const double eps2 = softening * softening;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (int j = 0; j < count; ++j) {
        if (i == j) {
            continue;
        }

        const double dx = particles[j].x - xi;
        const double dy = particles[j].y - yi;
        const double dz = particles[j].z - zi;
        const double s2 = dx * dx + dy * dy + dz * dz + eps2;
        if (s2 == 0.0) {
            continue;
        }

        const double inv_r = 1.0 / sqrt(s2);
        const double inv_r3 = inv_r * inv_r * inv_r;
        const double scale = gravitational_constant * particles[j].mass * inv_r3;
        ax += dx * scale;
        ay += dy * scale;
        az += dz * scale;
    }

    particles[i].ax = ax;
    particles[i].ay = ay;
    particles[i].az = az;
}

__global__ void drift_kernel(DeviceParticle* particles, int count, double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    particles[i].x += particles[i].vx * dt;
    particles[i].y += particles[i].vy * dt;
    particles[i].z += particles[i].vz * dt;
}

__global__ void kick_kernel(DeviceParticle* particles, int count, double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    particles[i].vx += particles[i].ax * dt;
    particles[i].vy += particles[i].ay * dt;
    particles[i].vz += particles[i].az * dt;
}

void copy_back(DeviceParticle* device_particles, std::vector<Particle>& particles) {
    std::vector<DeviceParticle> host_particles(particles.size());
    throw_on_cuda(
        cudaMemcpy(
            host_particles.data(),
            device_particles,
            host_particles.size() * sizeof(DeviceParticle),
            cudaMemcpyDeviceToHost
        ),
        "copy particles from CUDA device"
    );

    for (std::size_t i = 0; i < particles.size(); ++i) {
        unpack_particle(host_particles[i], particles[i]);
    }
}

DeviceParticle* copy_to_device(const std::vector<Particle>& particles) {
    std::vector<DeviceParticle> host_particles;
    host_particles.reserve(particles.size());
    for (const auto& particle : particles) {
        host_particles.push_back(pack_particle(particle));
    }

    DeviceParticle* device_particles = nullptr;
    throw_on_cuda(
        cudaMalloc(&device_particles, host_particles.size() * sizeof(DeviceParticle)),
        "allocate CUDA particle buffer"
    );
    throw_on_cuda(
        cudaMemcpy(
            device_particles,
            host_particles.data(),
            host_particles.size() * sizeof(DeviceParticle),
            cudaMemcpyHostToDevice
        ),
        "copy particles to CUDA device"
    );
    return device_particles;
}

void launch_acceleration(DeviceParticle* device_particles, std::size_t count, const PhysicsParams& params) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    direct_acceleration_kernel<<<blocks, threads>>>(
        device_particles,
        static_cast<int>(count),
        params.gravitational_constant,
        params.softening
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA direct acceleration kernel");
}

}  // namespace

bool cuda_solver_available() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

void compute_cuda_direct_accelerations(std::vector<Particle>& particles, const PhysicsParams& params) {
    if (particles.empty()) {
        return;
    }
    if (!cuda_solver_available()) {
        compute_direct_accelerations(particles, params);
        return;
    }

    DeviceParticle* device_particles = copy_to_device(particles);
    try {
        launch_acceleration(device_particles, particles.size(), params);
        throw_on_cuda(cudaDeviceSynchronize(), "synchronize CUDA direct acceleration kernel");
        copy_back(device_particles, particles);
    } catch (...) {
        cudaFree(device_particles);
        throw;
    }
    throw_on_cuda(cudaFree(device_particles), "free CUDA particle buffer");
}

void cuda_direct_leapfrog_step(std::vector<Particle>& particles, double dt, const PhysicsParams& params) {
    if (particles.empty()) {
        return;
    }
    if (!cuda_solver_available()) {
        auto compute = [&params](std::vector<Particle>& state) {
            compute_direct_accelerations(state, params);
        };
        leapfrog_step(particles, dt, compute);
        return;
    }

    DeviceParticle* device_particles = copy_to_device(particles);
    try {
        const int threads = 256;
        const int blocks = static_cast<int>((particles.size() + threads - 1) / threads);
        kick_kernel<<<blocks, threads>>>(device_particles, static_cast<int>(particles.size()), 0.5 * dt);
        throw_on_cuda(cudaGetLastError(), "launch CUDA half kick kernel");

        drift_kernel<<<blocks, threads>>>(device_particles, static_cast<int>(particles.size()), dt);
        throw_on_cuda(cudaGetLastError(), "launch CUDA drift kernel");

        launch_acceleration(device_particles, particles.size(), params);

        kick_kernel<<<blocks, threads>>>(device_particles, static_cast<int>(particles.size()), 0.5 * dt);
        throw_on_cuda(cudaGetLastError(), "launch CUDA final kick kernel");
        throw_on_cuda(cudaDeviceSynchronize(), "synchronize CUDA leapfrog kernels");

        copy_back(device_particles, particles);
    } catch (...) {
        cudaFree(device_particles);
        throw;
    }
    throw_on_cuda(cudaFree(device_particles), "free CUDA particle buffer");
}

}  // namespace fmmgalaxy
