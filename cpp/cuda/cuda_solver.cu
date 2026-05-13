#include "cuda/cuda_solver.hpp"

#include "core/integrator.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

struct DeviceTreeNode {
    double center_x;
    double center_y;
    double center_z;
    double half_width;
    double mass;
    double com_x;
    double com_y;
    double com_z;
    double moments[35];
    int children[8];
    int particle_begin;
    int particle_count;
    int is_leaf;
};

struct DeviceFmmLeaf {
    int node_index;
    int far_begin;
    int far_count;
    int near_begin;
    int near_count;
};

struct DeviceVec3 {
    double x;
    double y;
    double z;
};

constexpr int tree_stack_capacity = 512;

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

int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds CUDA int indexing limit");
    }
    return static_cast<int>(value);
}

DeviceTreeNode pack_tree_node(const FlatTreeNode& node) {
    DeviceTreeNode device_node{};
    device_node.center_x = node.center.x;
    device_node.center_y = node.center.y;
    device_node.center_z = node.center.z;
    device_node.half_width = node.half_width;
    device_node.mass = node.mass;
    device_node.com_x = node.center_of_mass.x;
    device_node.com_y = node.center_of_mass.y;
    device_node.com_z = node.center_of_mass.z;
    for (std::size_t i = 0; i < node.moments.values.size(); ++i) {
        device_node.moments[i] = node.moments.values[i];
    }
    for (std::size_t i = 0; i < node.children.size(); ++i) {
        device_node.children[i] = node.children[i];
    }
    device_node.particle_begin = checked_int(node.particle_begin, "tree node particle offset");
    device_node.particle_count = checked_int(node.particle_count, "tree node particle count");
    device_node.is_leaf = node.is_leaf ? 1 : 0;
    return device_node;
}

DeviceFmmLeaf pack_fmm_leaf(const FlatFmmLeaf& leaf) {
    return DeviceFmmLeaf{
        leaf.node_index,
        checked_int(leaf.far_begin, "FMM far-list offset"),
        checked_int(leaf.far_count, "FMM far-list count"),
        checked_int(leaf.near_begin, "FMM near-list offset"),
        checked_int(leaf.near_count, "FMM near-list count"),
    };
}

std::vector<DeviceTreeNode> pack_tree_nodes(const FlatTreeData& tree) {
    std::vector<DeviceTreeNode> nodes;
    nodes.reserve(tree.nodes.size());
    for (const FlatTreeNode& node : tree.nodes) {
        nodes.push_back(pack_tree_node(node));
    }
    return nodes;
}

std::vector<int> pack_particle_indices(const std::vector<std::size_t>& indices) {
    std::vector<int> packed;
    packed.reserve(indices.size());
    for (const std::size_t index : indices) {
        packed.push_back(checked_int(index, "particle index"));
    }
    return packed;
}

template <typename T>
T* copy_vector_to_device(const std::vector<T>& values, const char* allocation_context, const char* copy_context) {
    if (values.empty()) {
        return nullptr;
    }

    T* device_values = nullptr;
    throw_on_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_values), values.size() * sizeof(T)),
        allocation_context
    );
    try {
        throw_on_cuda(
            cudaMemcpy(
                device_values,
                values.data(),
                values.size() * sizeof(T),
                cudaMemcpyHostToDevice
            ),
            copy_context
        );
    } catch (...) {
        cudaFree(device_values);
        throw;
    }
    return device_values;
}

__device__ __constant__ int device_exponents[35][3] = {
    {0, 0, 0},
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2},
    {3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2},
    {0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3},
    {4, 0, 0}, {3, 1, 0}, {3, 0, 1}, {2, 2, 0}, {2, 1, 1}, {2, 0, 2},
    {1, 3, 0}, {1, 2, 1}, {1, 1, 2}, {1, 0, 3}, {0, 4, 0}, {0, 3, 1},
    {0, 2, 2}, {0, 1, 3}, {0, 0, 4},
};

__device__ int device_degree(int index) {
    return device_exponents[index][0] + device_exponents[index][1] + device_exponents[index][2];
}

__device__ int device_index_of(int x, int y, int z) {
    for (int i = 0; i < 35; ++i) {
        if (device_exponents[i][0] == x && device_exponents[i][1] == y && device_exponents[i][2] == z) {
            return i;
        }
    }
    return -1;
}

__device__ double device_pow_int(double value, int exponent) {
    double result = 1.0;
    for (int i = 0; i < exponent; ++i) {
        result *= value;
    }
    return result;
}

__device__ void device_zero_polynomial(double* polynomial) {
    for (int i = 0; i < 35; ++i) {
        polynomial[i] = 0.0;
    }
}

__device__ void device_multiply_polynomial(const double* a, const double* b, double* result) {
    device_zero_polynomial(result);
    for (int i = 0; i < 35; ++i) {
        if (a[i] == 0.0) {
            continue;
        }
        for (int j = 0; j < 35; ++j) {
            if (b[j] == 0.0) {
                continue;
            }

            const int x = device_exponents[i][0] + device_exponents[j][0];
            const int y = device_exponents[i][1] + device_exponents[j][1];
            const int z = device_exponents[i][2] + device_exponents[j][2];
            if (x + y + z > 4) {
                continue;
            }

            const int index = device_index_of(x, y, z);
            if (index >= 0) {
                result[index] += a[i] * b[j];
            }
        }
    }
}

__device__ void device_add_scaled_polynomial(double* result, const double* source, double scale) {
    for (int i = 0; i < 35; ++i) {
        result[i] += source[i] * scale;
    }
}

__device__ void device_scale_polynomial(const double* source, double scale, double* result) {
    for (int i = 0; i < 35; ++i) {
        result[i] = source[i] * scale;
    }
}

__device__ DeviceVec3 device_softened_acceleration(
    double tx,
    double ty,
    double tz,
    double sx,
    double sy,
    double sz,
    double source_mass,
    double gravitational_constant,
    double softening
) {
    const double dx = sx - tx;
    const double dy = sy - ty;
    const double dz = sz - tz;
    const double s2 = dx * dx + dy * dy + dz * dz + softening * softening;
    if (s2 == 0.0) {
        return {0.0, 0.0, 0.0};
    }

    const double inv_r = 1.0 / sqrt(s2);
    const double inv_r3 = inv_r * inv_r * inv_r;
    const double scale = gravitational_constant * source_mass * inv_r3;
    return {dx * scale, dy * scale, dz * scale};
}

__device__ void device_inv_r3_polynomial(DeviceVec3 delta, double softening, double* result) {
    const double h0 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z + softening * softening;
    const double base = pow(h0, -1.5);

    double q[35];
    device_zero_polynomial(q);
    q[device_index_of(1, 0, 0)] = 2.0 * delta.x / h0;
    q[device_index_of(0, 1, 0)] = 2.0 * delta.y / h0;
    q[device_index_of(0, 0, 1)] = 2.0 * delta.z / h0;
    q[device_index_of(2, 0, 0)] = 1.0 / h0;
    q[device_index_of(0, 2, 0)] = 1.0 / h0;
    q[device_index_of(0, 0, 2)] = 1.0 / h0;

    const double coefficients[5] = {1.0, -1.5, 1.875, -2.1875, 2.4609375};
    double series[35];
    double power[35];
    device_zero_polynomial(series);
    device_zero_polynomial(power);
    power[0] = 1.0;

    for (int n = 0; n <= 4; ++n) {
        device_add_scaled_polynomial(series, power, coefficients[n]);
        double next_power[35];
        device_multiply_polynomial(power, q, next_power);
        for (int i = 0; i < 35; ++i) {
            power[i] = next_power[i];
        }
    }

    device_scale_polynomial(series, base, result);
}

__device__ void device_component_polynomial(
    const double* inv_r3,
    int component,
    double component_value,
    double* result
) {
    device_scale_polynomial(inv_r3, component_value, result);

    double linear[35];
    double product[35];
    device_zero_polynomial(linear);
    linear[device_index_of(component == 0 ? 1 : 0, component == 1 ? 1 : 0, component == 2 ? 1 : 0)] = 1.0;
    device_multiply_polynomial(linear, inv_r3, product);
    for (int i = 0; i < 35; ++i) {
        result[i] += product[i];
    }
}

__device__ double device_expansion_moment_value(
    const double* moments,
    int exponent_index,
    double mass,
    int expansion_order
) {
    const int degree = device_degree(exponent_index);
    if (degree == 0) {
        return mass;
    }
    if (degree == 1 || degree > expansion_order) {
        return 0.0;
    }
    return moments[exponent_index];
}

__device__ double device_evaluate_component(
    const double* polynomial,
    const double* moments,
    double mass,
    int expansion_order
) {
    double value = 0.0;
    for (int i = 0; i < 35; ++i) {
        if (device_degree(i) <= expansion_order) {
            value += polynomial[i] * device_expansion_moment_value(moments, i, mass, expansion_order);
        }
    }
    return value;
}

__device__ DeviceVec3 device_multipole_acceleration(
    double tx,
    double ty,
    double tz,
    const DeviceTreeNode& source,
    double gravitational_constant,
    double softening,
    int expansion_order
) {
    if (source.mass <= 0.0) {
        return {0.0, 0.0, 0.0};
    }

    if (expansion_order < 0) {
        expansion_order = 0;
    }
    if (expansion_order > 4) {
        expansion_order = 4;
    }
    if (expansion_order == 1 || expansion_order == 3) {
        ++expansion_order;
    }

    if (expansion_order == 0) {
        return device_softened_acceleration(
            tx,
            ty,
            tz,
            source.com_x,
            source.com_y,
            source.com_z,
            source.mass,
            gravitational_constant,
            softening
        );
    }

    const DeviceVec3 delta{source.com_x - tx, source.com_y - ty, source.com_z - tz};
    const double h0 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z + softening * softening;
    if (h0 == 0.0) {
        return {0.0, 0.0, 0.0};
    }

    double inv[35];
    double gx[35];
    double gy[35];
    double gz[35];
    device_inv_r3_polynomial(delta, softening, inv);
    device_component_polynomial(inv, 0, delta.x, gx);
    device_component_polynomial(inv, 1, delta.y, gy);
    device_component_polynomial(inv, 2, delta.z, gz);

    return {
        gravitational_constant * device_evaluate_component(gx, source.moments, source.mass, expansion_order),
        gravitational_constant * device_evaluate_component(gy, source.moments, source.mass, expansion_order),
        gravitational_constant * device_evaluate_component(gz, source.moments, source.mass, expansion_order),
    };
}

__device__ DeviceVec3 device_direct_acceleration_for_target(
    const DeviceParticle* particles,
    int count,
    int target_index,
    double gravitational_constant,
    double softening
) {
    const DeviceParticle& target = particles[target_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};
    for (int j = 0; j < count; ++j) {
        if (j == target_index) {
            continue;
        }
        const DeviceVec3 contribution = device_softened_acceleration(
            target.x,
            target.y,
            target.z,
            particles[j].x,
            particles[j].y,
            particles[j].z,
            particles[j].mass,
            gravitational_constant,
            softening
        );
        acceleration.x += contribution.x;
        acceleration.y += contribution.y;
        acceleration.z += contribution.z;
    }
    return acceleration;
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

__global__ void tree_acceleration_kernel(
    DeviceParticle* particles,
    int count,
    const DeviceTreeNode* nodes,
    int node_count,
    const int* particle_indices,
    double gravitational_constant,
    double softening,
    double theta,
    int expansion_order
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count || node_count <= 0) {
        return;
    }

    const DeviceParticle& target = particles[target_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};
    int stack[tree_stack_capacity];
    int top = 0;
    bool stack_overflow = false;
    stack[top++] = 0;

    while (top > 0 && !stack_overflow) {
        const int node_index = stack[--top];
        if (node_index < 0 || node_index >= node_count) {
            continue;
        }

        const DeviceTreeNode& node = nodes[node_index];
        if (node.mass <= 0.0) {
            continue;
        }

        if (node.is_leaf != 0) {
            for (int offset = 0; offset < node.particle_count; ++offset) {
                const int source_index = particle_indices[node.particle_begin + offset];
                if (source_index == target_index) {
                    continue;
                }
                const DeviceParticle& source = particles[source_index];
                const DeviceVec3 contribution = device_softened_acceleration(
                    target.x,
                    target.y,
                    target.z,
                    source.x,
                    source.y,
                    source.z,
                    source.mass,
                    gravitational_constant,
                    softening
                );
                acceleration.x += contribution.x;
                acceleration.y += contribution.y;
                acceleration.z += contribution.z;
            }
            continue;
        }

        const double dx = node.com_x - target.x;
        const double dy = node.com_y - target.y;
        const double dz = node.com_z - target.z;
        const double distance = sqrt(dx * dx + dy * dy + dz * dz);
        const double node_width = 2.0 * node.half_width;
        const bool target_inside_node =
            fabs(target.x - node.center_x) <= node.half_width &&
            fabs(target.y - node.center_y) <= node.half_width &&
            fabs(target.z - node.center_z) <= node.half_width;

        if (!target_inside_node && distance > 0.0 && node_width / distance < theta) {
            const DeviceVec3 contribution = device_multipole_acceleration(
                target.x,
                target.y,
                target.z,
                node,
                gravitational_constant,
                softening,
                expansion_order
            );
            acceleration.x += contribution.x;
            acceleration.y += contribution.y;
            acceleration.z += contribution.z;
            continue;
        }

        for (int child = 0; child < 8; ++child) {
            const int child_index = node.children[child];
            if (child_index < 0) {
                continue;
            }
            if (top >= tree_stack_capacity) {
                stack_overflow = true;
                break;
            }
            stack[top++] = child_index;
        }
    }

    if (stack_overflow) {
        acceleration = device_direct_acceleration_for_target(
            particles,
            count,
            target_index,
            gravitational_constant,
            softening
        );
    }

    particles[target_index].ax = acceleration.x;
    particles[target_index].ay = acceleration.y;
    particles[target_index].az = acceleration.z;
}

__global__ void fmm_acceleration_kernel(
    DeviceParticle* particles,
    int count,
    const DeviceTreeNode* nodes,
    const int* particle_indices,
    const DeviceFmmLeaf* leaves,
    const int* far_node_indices,
    const int* near_leaf_node_indices,
    const int* particle_leaf_indices,
    double gravitational_constant,
    double softening,
    int expansion_order
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count) {
        return;
    }

    const int leaf_index = particle_leaf_indices[target_index];
    if (leaf_index < 0) {
        particles[target_index].ax = 0.0;
        particles[target_index].ay = 0.0;
        particles[target_index].az = 0.0;
        return;
    }

    const DeviceParticle& target = particles[target_index];
    const DeviceFmmLeaf& leaf = leaves[leaf_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};

    for (int offset = 0; offset < leaf.far_count; ++offset) {
        const int source_node_index = far_node_indices[leaf.far_begin + offset];
        const DeviceVec3 contribution = device_multipole_acceleration(
            target.x,
            target.y,
            target.z,
            nodes[source_node_index],
            gravitational_constant,
            softening,
            expansion_order
        );
        acceleration.x += contribution.x;
        acceleration.y += contribution.y;
        acceleration.z += contribution.z;
    }

    for (int near_offset = 0; near_offset < leaf.near_count; ++near_offset) {
        const int source_leaf_node_index = near_leaf_node_indices[leaf.near_begin + near_offset];
        const DeviceTreeNode& source_leaf = nodes[source_leaf_node_index];
        for (int offset = 0; offset < source_leaf.particle_count; ++offset) {
            const int source_index = particle_indices[source_leaf.particle_begin + offset];
            if (source_index == target_index) {
                continue;
            }
            const DeviceParticle& source = particles[source_index];
            const DeviceVec3 contribution = device_softened_acceleration(
                target.x,
                target.y,
                target.z,
                source.x,
                source.y,
                source.z,
                source.mass,
                gravitational_constant,
                softening
            );
            acceleration.x += contribution.x;
            acceleration.y += contribution.y;
            acceleration.z += contribution.z;
        }
    }

    particles[target_index].ax = acceleration.x;
    particles[target_index].ay = acceleration.y;
    particles[target_index].az = acceleration.z;
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
        cudaMalloc(reinterpret_cast<void**>(&device_particles), host_particles.size() * sizeof(DeviceParticle)),
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
        checked_int(count, "particle count"),
        params.gravitational_constant,
        params.softening
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA direct acceleration kernel");
}

void launch_tree_acceleration(
    DeviceParticle* device_particles,
    std::size_t count,
    const DeviceTreeNode* device_nodes,
    std::size_t node_count,
    const int* device_particle_indices,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    tree_acceleration_kernel<<<blocks, threads>>>(
        device_particles,
        checked_int(count, "particle count"),
        device_nodes,
        checked_int(node_count, "tree node count"),
        device_particle_indices,
        params.gravitational_constant,
        params.softening,
        options.theta,
        options.expansion_order
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA tree acceleration kernel");
}

void launch_fmm_acceleration(
    DeviceParticle* device_particles,
    std::size_t count,
    const DeviceTreeNode* device_nodes,
    const int* device_particle_indices,
    const DeviceFmmLeaf* device_leaves,
    const int* device_far_node_indices,
    const int* device_near_leaf_node_indices,
    const int* device_particle_leaf_indices,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    fmm_acceleration_kernel<<<blocks, threads>>>(
        device_particles,
        checked_int(count, "particle count"),
        device_nodes,
        device_particle_indices,
        device_leaves,
        device_far_node_indices,
        device_near_leaf_node_indices,
        device_particle_leaf_indices,
        params.gravitational_constant,
        params.softening,
        options.expansion_order
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA FMM acceleration kernel");
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

void compute_cuda_tree_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    if (particles.empty()) {
        return;
    }
    if (!cuda_solver_available()) {
        compute_tree_accelerations(
            particles,
            params,
            options.theta,
            options.leaf_capacity,
            options.expansion_order
        );
        return;
    }

    checked_int(particles.size(), "particle count");
    const FlatTreeData tree = build_flat_tree(
        particles,
        params,
        options.theta,
        options.leaf_capacity,
        options.max_depth,
        options.expansion_order
    );
    if (tree.nodes.empty()) {
        return;
    }

    const std::vector<DeviceTreeNode> host_nodes = pack_tree_nodes(tree);
    const std::vector<int> host_particle_indices = pack_particle_indices(tree.particle_indices);

    DeviceParticle* device_particles = copy_to_device(particles);
    DeviceTreeNode* device_nodes = copy_vector_to_device(
        host_nodes,
        "allocate CUDA tree nodes",
        "copy CUDA tree nodes"
    );
    int* device_particle_indices = copy_vector_to_device(
        host_particle_indices,
        "allocate CUDA tree particle indices",
        "copy CUDA tree particle indices"
    );

    try {
        launch_tree_acceleration(
            device_particles,
            particles.size(),
            device_nodes,
            host_nodes.size(),
            device_particle_indices,
            params,
            options
        );
        throw_on_cuda(cudaDeviceSynchronize(), "synchronize CUDA tree acceleration kernel");
        copy_back(device_particles, particles);
    } catch (...) {
        cudaFree(device_particle_indices);
        cudaFree(device_nodes);
        cudaFree(device_particles);
        throw;
    }

    throw_on_cuda(cudaFree(device_particle_indices), "free CUDA tree particle indices");
    throw_on_cuda(cudaFree(device_nodes), "free CUDA tree nodes");
    throw_on_cuda(cudaFree(device_particles), "free CUDA particle buffer");
}

void compute_cuda_fmm_accelerations(
    std::vector<Particle>& particles,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    if (particles.empty()) {
        return;
    }
    if (!cuda_solver_available()) {
        FmmOptions fmm_options;
        fmm_options.theta = options.theta;
        fmm_options.leaf_capacity = options.leaf_capacity;
        fmm_options.max_depth = options.max_depth;
        fmm_options.expansion_order = options.expansion_order;
        compute_fmm_accelerations(particles, params, fmm_options);
        return;
    }

    checked_int(particles.size(), "particle count");
    FmmOptions fmm_options;
    fmm_options.theta = options.theta;
    fmm_options.leaf_capacity = options.leaf_capacity;
    fmm_options.max_depth = options.max_depth;
    fmm_options.expansion_order = options.expansion_order;
    const FlatFmmData fmm = build_flat_fmm(particles, params, fmm_options);
    if (fmm.tree.nodes.empty() || fmm.leaves.empty()) {
        return;
    }

    const std::vector<DeviceTreeNode> host_nodes = pack_tree_nodes(fmm.tree);
    const std::vector<int> host_particle_indices = pack_particle_indices(fmm.tree.particle_indices);
    std::vector<DeviceFmmLeaf> host_leaves;
    host_leaves.reserve(fmm.leaves.size());
    for (const FlatFmmLeaf& leaf : fmm.leaves) {
        host_leaves.push_back(pack_fmm_leaf(leaf));
    }

    DeviceParticle* device_particles = copy_to_device(particles);
    DeviceTreeNode* device_nodes = copy_vector_to_device(
        host_nodes,
        "allocate CUDA FMM nodes",
        "copy CUDA FMM nodes"
    );
    int* device_particle_indices = copy_vector_to_device(
        host_particle_indices,
        "allocate CUDA FMM particle indices",
        "copy CUDA FMM particle indices"
    );
    DeviceFmmLeaf* device_leaves = copy_vector_to_device(
        host_leaves,
        "allocate CUDA FMM leaves",
        "copy CUDA FMM leaves"
    );
    int* device_far_node_indices = copy_vector_to_device(
        fmm.far_node_indices,
        "allocate CUDA FMM far-list indices",
        "copy CUDA FMM far-list indices"
    );
    int* device_near_leaf_node_indices = copy_vector_to_device(
        fmm.near_leaf_node_indices,
        "allocate CUDA FMM near-list indices",
        "copy CUDA FMM near-list indices"
    );
    int* device_particle_leaf_indices = copy_vector_to_device(
        fmm.particle_leaf_indices,
        "allocate CUDA FMM particle-leaf indices",
        "copy CUDA FMM particle-leaf indices"
    );

    try {
        launch_fmm_acceleration(
            device_particles,
            particles.size(),
            device_nodes,
            device_particle_indices,
            device_leaves,
            device_far_node_indices,
            device_near_leaf_node_indices,
            device_particle_leaf_indices,
            params,
            options
        );
        throw_on_cuda(cudaDeviceSynchronize(), "synchronize CUDA FMM acceleration kernel");
        copy_back(device_particles, particles);
    } catch (...) {
        cudaFree(device_particle_leaf_indices);
        cudaFree(device_near_leaf_node_indices);
        cudaFree(device_far_node_indices);
        cudaFree(device_leaves);
        cudaFree(device_particle_indices);
        cudaFree(device_nodes);
        cudaFree(device_particles);
        throw;
    }

    throw_on_cuda(cudaFree(device_particle_leaf_indices), "free CUDA FMM particle-leaf indices");
    if (device_near_leaf_node_indices != nullptr) {
        throw_on_cuda(cudaFree(device_near_leaf_node_indices), "free CUDA FMM near-list indices");
    }
    if (device_far_node_indices != nullptr) {
        throw_on_cuda(cudaFree(device_far_node_indices), "free CUDA FMM far-list indices");
    }
    throw_on_cuda(cudaFree(device_leaves), "free CUDA FMM leaves");
    throw_on_cuda(cudaFree(device_particle_indices), "free CUDA FMM particle indices");
    throw_on_cuda(cudaFree(device_nodes), "free CUDA FMM nodes");
    throw_on_cuda(cudaFree(device_particles), "free CUDA particle buffer");
}

void cuda_tree_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    auto compute = [&params, options](std::vector<Particle>& state) {
        compute_cuda_tree_accelerations(state, params, options);
    };
    leapfrog_step(particles, dt, compute);
}

void cuda_fmm_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    auto compute = [&params, options](std::vector<Particle>& state) {
        compute_cuda_fmm_accelerations(state, params, options);
    };
    leapfrog_step(particles, dt, compute);
}

}  // namespace fmmgalaxy
