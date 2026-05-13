#include "cuda/cuda_solver.hpp"

#include "core/integrator.hpp"
#include "direct/direct_solver.hpp"
#include "fmm/fmm_solver.hpp"
#include "fmm/quadtree.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

struct DeviceBody {
    double x;
    double y;
    double z;
    double mass;
};

struct DeviceAcceleration {
    double ax;
    double ay;
    double az;
};

struct DeviceBodySoA {
    const double* x;
    const double* y;
    const double* z;
    const double* mass;
};

struct DeviceParticleSoA {
    double* x;
    double* y;
    double* z;
    double* vx;
    double* vy;
    double* vz;
    double* ax;
    double* ay;
    double* az;
    double* mass;
    std::uint32_t* group_id;
};

struct DeviceAccelerationSoA {
    double* ax;
    double* ay;
    double* az;
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

struct DeviceMonopoleNode {
    double center_x;
    double center_y;
    double center_z;
    double half_width;
    double mass;
    double com_x;
    double com_y;
    double com_z;
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
constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

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

DeviceBody pack_body(const Particle& particle) {
    return DeviceBody{
        particle.position.x,
        particle.position.y,
        particle.position.z,
        particle.mass,
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

std::uint64_t mix_hash(std::uint64_t hash, std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & 0xffu;
        hash *= fnv_prime;
    }
    return hash;
}

std::uint64_t double_bits(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
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

DeviceMonopoleNode pack_monopole_node(const FlatTreeNode& node) {
    DeviceMonopoleNode device_node{};
    device_node.center_x = node.center.x;
    device_node.center_y = node.center.y;
    device_node.center_z = node.center.z;
    device_node.half_width = node.half_width;
    device_node.mass = node.mass;
    device_node.com_x = node.center_of_mass.x;
    device_node.com_y = node.center_of_mass.y;
    device_node.com_z = node.center_of_mass.z;
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

std::vector<DeviceMonopoleNode> pack_monopole_nodes(const FlatTreeData& tree) {
    std::vector<DeviceMonopoleNode> nodes;
    nodes.reserve(tree.nodes.size());
    for (const FlatTreeNode& node : tree.nodes) {
        nodes.push_back(pack_monopole_node(node));
    }
    return nodes;
}

std::vector<DeviceBody> pack_bodies(const std::vector<Particle>& particles) {
    std::vector<DeviceBody> bodies;
    bodies.reserve(particles.size());
    for (const Particle& particle : particles) {
        bodies.push_back(pack_body(particle));
    }
    return bodies;
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

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        release();
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    void ensure(std::size_t count, const char* context) {
        if (count <= capacity_) {
            return;
        }

        T* next = nullptr;
        throw_on_cuda(
            cudaMalloc(reinterpret_cast<void**>(&next), count * sizeof(T)),
            context
        );
        release();
        data_ = next;
        capacity_ = count;
    }

    void upload(const T* host, std::size_t count, cudaStream_t stream, const char* allocation_context, const char* copy_context) {
        if (count == 0) {
            return;
        }
        ensure(count, allocation_context);
        throw_on_cuda(
            cudaMemcpyAsync(data_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream),
            copy_context
        );
    }

    void download(T* host, std::size_t count, cudaStream_t stream, const char* copy_context) {
        if (count == 0) {
            return;
        }
        throw_on_cuda(
            cudaMemcpyAsync(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
            copy_context
        );
    }

private:
    void release() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
    }

    T* data_{nullptr};
    std::size_t capacity_{0};
};

template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    ~PinnedHostBuffer() {
        release();
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    void ensure(std::size_t count, const char* context) {
        if (count <= capacity_) {
            return;
        }

        T* next = nullptr;
        throw_on_cuda(
            cudaHostAlloc(reinterpret_cast<void**>(&next), count * sizeof(T), cudaHostAllocPortable),
            context
        );
        release();
        data_ = next;
        capacity_ = count;
    }

private:
    void release() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
    }

    T* data_{nullptr};
    std::size_t capacity_{0};
};

struct CudaWorkspace {
    ~CudaWorkspace() {
        if (stream_created_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    cudaStream_t stream() {
        if (!stream_created_) {
            throw_on_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create CUDA stream");
            stream_created_ = true;
        }
        return stream_;
    }

    DeviceParticleSoA particle_arrays() {
        return {
            x.data(),
            y.data(),
            z.data(),
            vx.data(),
            vy.data(),
            vz.data(),
            ax.data(),
            ay.data(),
            az.data(),
            mass.data(),
            group_id.data(),
        };
    }

    DeviceBodySoA body_arrays() const {
        return {x.data(), y.data(), z.data(), mass.data()};
    }

    DeviceAccelerationSoA acceleration_arrays() {
        return {ax.data(), ay.data(), az.data()};
    }

    DeviceBuffer<double> x{};
    DeviceBuffer<double> y{};
    DeviceBuffer<double> z{};
    DeviceBuffer<double> vx{};
    DeviceBuffer<double> vy{};
    DeviceBuffer<double> vz{};
    DeviceBuffer<double> ax{};
    DeviceBuffer<double> ay{};
    DeviceBuffer<double> az{};
    DeviceBuffer<double> mass{};
    DeviceBuffer<std::uint32_t> group_id{};
    DeviceBuffer<DeviceTreeNode> tree_nodes{};
    DeviceBuffer<DeviceMonopoleNode> monopole_nodes{};
    DeviceBuffer<DeviceFmmLeaf> leaves{};
    DeviceBuffer<int> particle_indices{};
    DeviceBuffer<int> far_node_indices{};
    DeviceBuffer<int> near_leaf_node_indices{};
    DeviceBuffer<int> particle_leaf_indices{};

    PinnedHostBuffer<double> host_x{};
    PinnedHostBuffer<double> host_y{};
    PinnedHostBuffer<double> host_z{};
    PinnedHostBuffer<double> host_vx{};
    PinnedHostBuffer<double> host_vy{};
    PinnedHostBuffer<double> host_vz{};
    PinnedHostBuffer<double> host_ax{};
    PinnedHostBuffer<double> host_ay{};
    PinnedHostBuffer<double> host_az{};
    PinnedHostBuffer<double> host_mass{};
    PinnedHostBuffer<std::uint32_t> host_group_id{};
    PinnedHostBuffer<DeviceTreeNode> host_tree_nodes{};
    PinnedHostBuffer<DeviceMonopoleNode> host_monopole_nodes{};
    PinnedHostBuffer<DeviceFmmLeaf> host_leaves{};
    PinnedHostBuffer<int> host_particle_indices{};
    PinnedHostBuffer<int> host_far_node_indices{};
    PinnedHostBuffer<int> host_near_leaf_node_indices{};
    PinnedHostBuffer<int> host_particle_leaf_indices{};

    std::size_t cached_mass_count{0};
    std::uint64_t cached_mass_hash{0};
    bool mass_cache_valid{false};
    std::size_t cached_group_count{0};
    std::uint64_t cached_group_hash{0};
    bool group_cache_valid{false};

private:
    cudaStream_t stream_{};
    bool stream_created_{false};
};

CudaWorkspace& cuda_workspace() {
    static CudaWorkspace workspace;
    return workspace;
}

template <typename T>
void upload_vector(
    const std::vector<T>& values,
    PinnedHostBuffer<T>& host_buffer,
    DeviceBuffer<T>& device_buffer,
    cudaStream_t stream,
    const char* host_context,
    const char* device_context,
    const char* copy_context
) {
    if (values.empty()) {
        return;
    }
    host_buffer.ensure(values.size(), host_context);
    std::copy(values.begin(), values.end(), host_buffer.data());
    device_buffer.upload(host_buffer.data(), values.size(), stream, device_context, copy_context);
}

void ensure_acceleration_arrays(CudaWorkspace& workspace, std::size_t count) {
    workspace.ax.ensure(count, "allocate CUDA acceleration x buffer");
    workspace.ay.ensure(count, "allocate CUDA acceleration y buffer");
    workspace.az.ensure(count, "allocate CUDA acceleration z buffer");
}

void upload_static_mass_array(
    CudaWorkspace& workspace,
    const std::vector<Particle>& particles,
    std::uint64_t mass_hash,
    cudaStream_t stream
) {
    const std::size_t count = particles.size();
    if (workspace.mass_cache_valid &&
        workspace.cached_mass_count == count &&
        workspace.cached_mass_hash == mass_hash) {
        return;
    }

    workspace.host_mass.ensure(count, "allocate pinned host mass buffer");
    for (std::size_t i = 0; i < count; ++i) {
        workspace.host_mass.data()[i] = particles[i].mass;
    }
    workspace.mass.upload(workspace.host_mass.data(), count, stream, "allocate CUDA mass buffer", "copy mass to CUDA device");
    workspace.cached_mass_count = count;
    workspace.cached_mass_hash = mass_hash;
    workspace.mass_cache_valid = true;
}

void upload_static_group_array(
    CudaWorkspace& workspace,
    const std::vector<Particle>& particles,
    std::uint64_t group_hash,
    cudaStream_t stream
) {
    const std::size_t count = particles.size();
    if (workspace.group_cache_valid &&
        workspace.cached_group_count == count &&
        workspace.cached_group_hash == group_hash) {
        return;
    }

    workspace.host_group_id.ensure(count, "allocate pinned host particle group buffer");
    for (std::size_t i = 0; i < count; ++i) {
        workspace.host_group_id.data()[i] = particles[i].group_id;
    }
    workspace.group_id.upload(
        workspace.host_group_id.data(),
        count,
        stream,
        "allocate CUDA particle group buffer",
        "copy particle group to CUDA device"
    );
    workspace.cached_group_count = count;
    workspace.cached_group_hash = group_hash;
    workspace.group_cache_valid = true;
}

void upload_body_arrays(CudaWorkspace& workspace, const std::vector<Particle>& particles, cudaStream_t stream) {
    const std::size_t count = particles.size();
    workspace.host_x.ensure(count, "allocate pinned host body x buffer");
    workspace.host_y.ensure(count, "allocate pinned host body y buffer");
    workspace.host_z.ensure(count, "allocate pinned host body z buffer");

    std::uint64_t mass_hash = mix_hash(fnv_offset_basis, static_cast<std::uint64_t>(count));
    for (std::size_t i = 0; i < count; ++i) {
        workspace.host_x.data()[i] = particles[i].position.x;
        workspace.host_y.data()[i] = particles[i].position.y;
        workspace.host_z.data()[i] = particles[i].position.z;
        mass_hash = mix_hash(mass_hash, double_bits(particles[i].mass));
    }

    workspace.x.upload(workspace.host_x.data(), count, stream, "allocate CUDA body x buffer", "copy body x to CUDA device");
    workspace.y.upload(workspace.host_y.data(), count, stream, "allocate CUDA body y buffer", "copy body y to CUDA device");
    workspace.z.upload(workspace.host_z.data(), count, stream, "allocate CUDA body z buffer", "copy body z to CUDA device");
    upload_static_mass_array(workspace, particles, mass_hash, stream);
    ensure_acceleration_arrays(workspace, count);
}

void upload_particle_arrays(CudaWorkspace& workspace, const std::vector<Particle>& particles, cudaStream_t stream) {
    const std::size_t count = particles.size();
    workspace.host_x.ensure(count, "allocate pinned host particle x buffer");
    workspace.host_y.ensure(count, "allocate pinned host particle y buffer");
    workspace.host_z.ensure(count, "allocate pinned host particle z buffer");
    workspace.host_vx.ensure(count, "allocate pinned host particle vx buffer");
    workspace.host_vy.ensure(count, "allocate pinned host particle vy buffer");
    workspace.host_vz.ensure(count, "allocate pinned host particle vz buffer");
    workspace.host_ax.ensure(count, "allocate pinned host particle ax buffer");
    workspace.host_ay.ensure(count, "allocate pinned host particle ay buffer");
    workspace.host_az.ensure(count, "allocate pinned host particle az buffer");

    std::uint64_t mass_hash = mix_hash(fnv_offset_basis, static_cast<std::uint64_t>(count));
    std::uint64_t group_hash = mix_hash(fnv_offset_basis, static_cast<std::uint64_t>(count));
    for (std::size_t i = 0; i < count; ++i) {
        workspace.host_x.data()[i] = particles[i].position.x;
        workspace.host_y.data()[i] = particles[i].position.y;
        workspace.host_z.data()[i] = particles[i].position.z;
        workspace.host_vx.data()[i] = particles[i].velocity.x;
        workspace.host_vy.data()[i] = particles[i].velocity.y;
        workspace.host_vz.data()[i] = particles[i].velocity.z;
        workspace.host_ax.data()[i] = particles[i].acceleration.x;
        workspace.host_ay.data()[i] = particles[i].acceleration.y;
        workspace.host_az.data()[i] = particles[i].acceleration.z;
        mass_hash = mix_hash(mass_hash, double_bits(particles[i].mass));
        group_hash = mix_hash(group_hash, static_cast<std::uint64_t>(particles[i].group_id));
    }

    workspace.x.upload(workspace.host_x.data(), count, stream, "allocate CUDA particle x buffer", "copy particle x to CUDA device");
    workspace.y.upload(workspace.host_y.data(), count, stream, "allocate CUDA particle y buffer", "copy particle y to CUDA device");
    workspace.z.upload(workspace.host_z.data(), count, stream, "allocate CUDA particle z buffer", "copy particle z to CUDA device");
    workspace.vx.upload(workspace.host_vx.data(), count, stream, "allocate CUDA particle vx buffer", "copy particle vx to CUDA device");
    workspace.vy.upload(workspace.host_vy.data(), count, stream, "allocate CUDA particle vy buffer", "copy particle vy to CUDA device");
    workspace.vz.upload(workspace.host_vz.data(), count, stream, "allocate CUDA particle vz buffer", "copy particle vz to CUDA device");
    workspace.ax.upload(workspace.host_ax.data(), count, stream, "allocate CUDA particle ax buffer", "copy particle ax to CUDA device");
    workspace.ay.upload(workspace.host_ay.data(), count, stream, "allocate CUDA particle ay buffer", "copy particle ay to CUDA device");
    workspace.az.upload(workspace.host_az.data(), count, stream, "allocate CUDA particle az buffer", "copy particle az to CUDA device");
    upload_static_mass_array(workspace, particles, mass_hash, stream);
    upload_static_group_array(workspace, particles, group_hash, stream);
}

void download_acceleration_arrays(CudaWorkspace& workspace, std::vector<Particle>& particles, cudaStream_t stream) {
    const std::size_t count = particles.size();
    workspace.host_ax.ensure(count, "allocate pinned host acceleration x buffer");
    workspace.host_ay.ensure(count, "allocate pinned host acceleration y buffer");
    workspace.host_az.ensure(count, "allocate pinned host acceleration z buffer");
    workspace.ax.download(workspace.host_ax.data(), count, stream, "copy acceleration x from CUDA device");
    workspace.ay.download(workspace.host_ay.data(), count, stream, "copy acceleration y from CUDA device");
    workspace.az.download(workspace.host_az.data(), count, stream, "copy acceleration z from CUDA device");
    throw_on_cuda(cudaStreamSynchronize(stream), "synchronize CUDA acceleration downloads");

    for (std::size_t i = 0; i < count; ++i) {
        particles[i].acceleration = {
            workspace.host_ax.data()[i],
            workspace.host_ay.data()[i],
            workspace.host_az.data()[i],
        };
    }
}

void download_particle_arrays(CudaWorkspace& workspace, std::vector<Particle>& particles, cudaStream_t stream) {
    const std::size_t count = particles.size();
    workspace.host_x.ensure(count, "allocate pinned host particle x buffer");
    workspace.host_y.ensure(count, "allocate pinned host particle y buffer");
    workspace.host_z.ensure(count, "allocate pinned host particle z buffer");
    workspace.host_vx.ensure(count, "allocate pinned host particle vx buffer");
    workspace.host_vy.ensure(count, "allocate pinned host particle vy buffer");
    workspace.host_vz.ensure(count, "allocate pinned host particle vz buffer");
    workspace.host_ax.ensure(count, "allocate pinned host particle ax buffer");
    workspace.host_ay.ensure(count, "allocate pinned host particle ay buffer");
    workspace.host_az.ensure(count, "allocate pinned host particle az buffer");

    workspace.x.download(workspace.host_x.data(), count, stream, "copy particle x from CUDA device");
    workspace.y.download(workspace.host_y.data(), count, stream, "copy particle y from CUDA device");
    workspace.z.download(workspace.host_z.data(), count, stream, "copy particle z from CUDA device");
    workspace.vx.download(workspace.host_vx.data(), count, stream, "copy particle vx from CUDA device");
    workspace.vy.download(workspace.host_vy.data(), count, stream, "copy particle vy from CUDA device");
    workspace.vz.download(workspace.host_vz.data(), count, stream, "copy particle vz from CUDA device");
    workspace.ax.download(workspace.host_ax.data(), count, stream, "copy particle ax from CUDA device");
    workspace.ay.download(workspace.host_ay.data(), count, stream, "copy particle ay from CUDA device");
    workspace.az.download(workspace.host_az.data(), count, stream, "copy particle az from CUDA device");
    throw_on_cuda(cudaStreamSynchronize(stream), "synchronize CUDA particle downloads");

    for (std::size_t i = 0; i < count; ++i) {
        particles[i].position = {
            workspace.host_x.data()[i],
            workspace.host_y.data()[i],
            workspace.host_z.data()[i],
        };
        particles[i].velocity = {
            workspace.host_vx.data()[i],
            workspace.host_vy.data()[i],
            workspace.host_vz.data()[i],
        };
        particles[i].acceleration = {
            workspace.host_ax.data()[i],
            workspace.host_ay.data()[i],
            workspace.host_az.data()[i],
        };
    }
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

__device__ DeviceVec3 device_monopole_acceleration(
    double tx,
    double ty,
    double tz,
    const DeviceMonopoleNode& source,
    double gravitational_constant,
    double softening
) {
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

__device__ DeviceVec3 device_direct_acceleration_for_target(
    const DeviceBody* bodies,
    int count,
    int target_index,
    double gravitational_constant,
    double softening
) {
    const DeviceBody& target = bodies[target_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};
    for (int j = 0; j < count; ++j) {
        if (j == target_index) {
            continue;
        }
        const DeviceBody& source = bodies[j];
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

__global__ void tree_monopole_acceleration_kernel(
    const DeviceBody* bodies,
    DeviceAcceleration* accelerations,
    int count,
    const DeviceMonopoleNode* nodes,
    int node_count,
    const int* particle_indices,
    double gravitational_constant,
    double softening,
    double theta
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count || node_count <= 0) {
        return;
    }

    const DeviceBody& target = bodies[target_index];
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

        const DeviceMonopoleNode& node = nodes[node_index];
        if (node.mass <= 0.0) {
            continue;
        }

        if (node.is_leaf != 0) {
            for (int offset = 0; offset < node.particle_count; ++offset) {
                const int source_index = particle_indices[node.particle_begin + offset];
                if (source_index == target_index) {
                    continue;
                }
                const DeviceBody& source = bodies[source_index];
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
            const DeviceVec3 contribution = device_monopole_acceleration(
                target.x,
                target.y,
                target.z,
                node,
                gravitational_constant,
                softening
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
            bodies,
            count,
            target_index,
            gravitational_constant,
            softening
        );
    }

    accelerations[target_index].ax = acceleration.x;
    accelerations[target_index].ay = acceleration.y;
    accelerations[target_index].az = acceleration.z;
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

__global__ void fmm_monopole_acceleration_kernel(
    const DeviceBody* bodies,
    DeviceAcceleration* accelerations,
    int count,
    const DeviceMonopoleNode* nodes,
    const int* particle_indices,
    const DeviceFmmLeaf* leaves,
    const int* far_node_indices,
    const int* near_leaf_node_indices,
    const int* particle_leaf_indices,
    double gravitational_constant,
    double softening
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count) {
        return;
    }

    const int leaf_index = particle_leaf_indices[target_index];
    if (leaf_index < 0) {
        accelerations[target_index].ax = 0.0;
        accelerations[target_index].ay = 0.0;
        accelerations[target_index].az = 0.0;
        return;
    }

    const DeviceBody& target = bodies[target_index];
    const DeviceFmmLeaf& leaf = leaves[leaf_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};

    for (int offset = 0; offset < leaf.far_count; ++offset) {
        const int source_node_index = far_node_indices[leaf.far_begin + offset];
        const DeviceVec3 contribution = device_monopole_acceleration(
            target.x,
            target.y,
            target.z,
            nodes[source_node_index],
            gravitational_constant,
            softening
        );
        acceleration.x += contribution.x;
        acceleration.y += contribution.y;
        acceleration.z += contribution.z;
    }

    for (int near_offset = 0; near_offset < leaf.near_count; ++near_offset) {
        const int source_leaf_node_index = near_leaf_node_indices[leaf.near_begin + near_offset];
        const DeviceMonopoleNode& source_leaf = nodes[source_leaf_node_index];
        for (int offset = 0; offset < source_leaf.particle_count; ++offset) {
            const int source_index = particle_indices[source_leaf.particle_begin + offset];
            if (source_index == target_index) {
                continue;
            }
            const DeviceBody& source = bodies[source_index];
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

    accelerations[target_index].ax = acceleration.x;
    accelerations[target_index].ay = acceleration.y;
    accelerations[target_index].az = acceleration.z;
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

__device__ DeviceVec3 device_direct_acceleration_for_target(
    DeviceBodySoA bodies,
    int count,
    int target_index,
    double gravitational_constant,
    double softening
) {
    const double tx = bodies.x[target_index];
    const double ty = bodies.y[target_index];
    const double tz = bodies.z[target_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};
    for (int j = 0; j < count; ++j) {
        if (j == target_index) {
            continue;
        }
        const DeviceVec3 contribution = device_softened_acceleration(
            tx,
            ty,
            tz,
            bodies.x[j],
            bodies.y[j],
            bodies.z[j],
            bodies.mass[j],
            gravitational_constant,
            softening
        );
        acceleration.x += contribution.x;
        acceleration.y += contribution.y;
        acceleration.z += contribution.z;
    }
    return acceleration;
}

template <int ExpansionOrder>
__device__ void device_inv_r3_polynomial_order(DeviceVec3 delta, double softening, double* result) {
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

    for (int n = 0; n <= ExpansionOrder; ++n) {
        device_add_scaled_polynomial(series, power, coefficients[n]);
        if (n == ExpansionOrder) {
            break;
        }
        double next_power[35];
        device_multiply_polynomial(power, q, next_power);
        for (int i = 0; i < 35; ++i) {
            power[i] = next_power[i];
        }
    }

    device_scale_polynomial(series, base, result);
}

template <int ExpansionOrder>
__device__ double device_expansion_moment_value_order(
    const double* moments,
    int exponent_index,
    double mass
) {
    const int degree = device_degree(exponent_index);
    if (degree == 0) {
        return mass;
    }
    if (degree == 1 || degree > ExpansionOrder) {
        return 0.0;
    }
    return moments[exponent_index];
}

template <int ExpansionOrder>
__device__ double device_evaluate_component_order(
    const double* polynomial,
    const double* moments,
    double mass
) {
    double value = 0.0;
    for (int i = 0; i < 35; ++i) {
        if (device_degree(i) <= ExpansionOrder) {
            value += polynomial[i] * device_expansion_moment_value_order<ExpansionOrder>(moments, i, mass);
        }
    }
    return value;
}

template <int ExpansionOrder>
__device__ DeviceVec3 device_multipole_acceleration_order(
    double tx,
    double ty,
    double tz,
    const DeviceTreeNode& source,
    double gravitational_constant,
    double softening
) {
    if (source.mass <= 0.0) {
        return {0.0, 0.0, 0.0};
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
    device_inv_r3_polynomial_order<ExpansionOrder>(delta, softening, inv);
    device_component_polynomial(inv, 0, delta.x, gx);
    device_component_polynomial(inv, 1, delta.y, gy);
    device_component_polynomial(inv, 2, delta.z, gz);

    return {
        gravitational_constant * device_evaluate_component_order<ExpansionOrder>(gx, source.moments, source.mass),
        gravitational_constant * device_evaluate_component_order<ExpansionOrder>(gy, source.moments, source.mass),
        gravitational_constant * device_evaluate_component_order<ExpansionOrder>(gz, source.moments, source.mass),
    };
}

__global__ void direct_tiled_acceleration_kernel(
    DeviceBodySoA bodies,
    DeviceAccelerationSoA accelerations,
    int count,
    double gravitational_constant,
    double softening
) {
    extern __shared__ double shared_body_values[];
    double* shared_x = shared_body_values;
    double* shared_y = shared_x + blockDim.x;
    double* shared_z = shared_y + blockDim.x;
    double* shared_mass = shared_z + blockDim.x;

    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = target_index < count;
    const double tx = active ? bodies.x[target_index] : 0.0;
    const double ty = active ? bodies.y[target_index] : 0.0;
    const double tz = active ? bodies.z[target_index] : 0.0;
    const double eps2 = softening * softening;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (int tile_begin = 0; tile_begin < count; tile_begin += blockDim.x) {
        const int source_index = tile_begin + threadIdx.x;
        if (source_index < count) {
            shared_x[threadIdx.x] = bodies.x[source_index];
            shared_y[threadIdx.x] = bodies.y[source_index];
            shared_z[threadIdx.x] = bodies.z[source_index];
            shared_mass[threadIdx.x] = bodies.mass[source_index];
        } else {
            shared_x[threadIdx.x] = 0.0;
            shared_y[threadIdx.x] = 0.0;
            shared_z[threadIdx.x] = 0.0;
            shared_mass[threadIdx.x] = 0.0;
        }
        __syncthreads();

        const int remaining = count - tile_begin;
        const int tile_count = remaining < blockDim.x ? remaining : blockDim.x;
        if (active) {
            for (int j = 0; j < tile_count; ++j) {
                if (tile_begin + j == target_index) {
                    continue;
                }
                const double dx = shared_x[j] - tx;
                const double dy = shared_y[j] - ty;
                const double dz = shared_z[j] - tz;
                const double s2 = dx * dx + dy * dy + dz * dz + eps2;
                if (s2 == 0.0) {
                    continue;
                }
                const double inv_r = 1.0 / sqrt(s2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                const double scale = gravitational_constant * shared_mass[j] * inv_r3;
                ax += dx * scale;
                ay += dy * scale;
                az += dz * scale;
            }
        }
        __syncthreads();
    }

    if (!active) {
        return;
    }

    accelerations.ax[target_index] = ax;
    accelerations.ay[target_index] = ay;
    accelerations.az[target_index] = az;
}

template <int ExpansionOrder>
__global__ void tree_acceleration_order_soa_kernel(
    DeviceBodySoA bodies,
    DeviceAccelerationSoA accelerations,
    int count,
    const DeviceTreeNode* nodes,
    int node_count,
    const int* particle_indices,
    double gravitational_constant,
    double softening,
    double theta
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count || node_count <= 0) {
        return;
    }

    const double tx = bodies.x[target_index];
    const double ty = bodies.y[target_index];
    const double tz = bodies.z[target_index];
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
                const DeviceVec3 contribution = device_softened_acceleration(
                    tx,
                    ty,
                    tz,
                    bodies.x[source_index],
                    bodies.y[source_index],
                    bodies.z[source_index],
                    bodies.mass[source_index],
                    gravitational_constant,
                    softening
                );
                acceleration.x += contribution.x;
                acceleration.y += contribution.y;
                acceleration.z += contribution.z;
            }
            continue;
        }

        const double dx = node.com_x - tx;
        const double dy = node.com_y - ty;
        const double dz = node.com_z - tz;
        const double distance = sqrt(dx * dx + dy * dy + dz * dz);
        const double node_width = 2.0 * node.half_width;
        const bool target_inside_node =
            fabs(tx - node.center_x) <= node.half_width &&
            fabs(ty - node.center_y) <= node.half_width &&
            fabs(tz - node.center_z) <= node.half_width;

        if (!target_inside_node && distance > 0.0 && node_width / distance < theta) {
            const DeviceVec3 contribution = device_multipole_acceleration_order<ExpansionOrder>(
                tx,
                ty,
                tz,
                node,
                gravitational_constant,
                softening
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
            bodies,
            count,
            target_index,
            gravitational_constant,
            softening
        );
    }

    accelerations.ax[target_index] = acceleration.x;
    accelerations.ay[target_index] = acceleration.y;
    accelerations.az[target_index] = acceleration.z;
}

__global__ void tree_monopole_acceleration_soa_kernel(
    DeviceBodySoA bodies,
    DeviceAccelerationSoA accelerations,
    int count,
    const DeviceMonopoleNode* nodes,
    int node_count,
    const int* particle_indices,
    double gravitational_constant,
    double softening,
    double theta
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count || node_count <= 0) {
        return;
    }

    const double tx = bodies.x[target_index];
    const double ty = bodies.y[target_index];
    const double tz = bodies.z[target_index];
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

        const DeviceMonopoleNode& node = nodes[node_index];
        if (node.mass <= 0.0) {
            continue;
        }

        if (node.is_leaf != 0) {
            for (int offset = 0; offset < node.particle_count; ++offset) {
                const int source_index = particle_indices[node.particle_begin + offset];
                if (source_index == target_index) {
                    continue;
                }
                const DeviceVec3 contribution = device_softened_acceleration(
                    tx,
                    ty,
                    tz,
                    bodies.x[source_index],
                    bodies.y[source_index],
                    bodies.z[source_index],
                    bodies.mass[source_index],
                    gravitational_constant,
                    softening
                );
                acceleration.x += contribution.x;
                acceleration.y += contribution.y;
                acceleration.z += contribution.z;
            }
            continue;
        }

        const double dx = node.com_x - tx;
        const double dy = node.com_y - ty;
        const double dz = node.com_z - tz;
        const double distance = sqrt(dx * dx + dy * dy + dz * dz);
        const double node_width = 2.0 * node.half_width;
        const bool target_inside_node =
            fabs(tx - node.center_x) <= node.half_width &&
            fabs(ty - node.center_y) <= node.half_width &&
            fabs(tz - node.center_z) <= node.half_width;

        if (!target_inside_node && distance > 0.0 && node_width / distance < theta) {
            const DeviceVec3 contribution = device_monopole_acceleration(
                tx,
                ty,
                tz,
                node,
                gravitational_constant,
                softening
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
            bodies,
            count,
            target_index,
            gravitational_constant,
            softening
        );
    }

    accelerations.ax[target_index] = acceleration.x;
    accelerations.ay[target_index] = acceleration.y;
    accelerations.az[target_index] = acceleration.z;
}

template <int ExpansionOrder>
__global__ void fmm_acceleration_order_soa_kernel(
    DeviceBodySoA bodies,
    DeviceAccelerationSoA accelerations,
    int count,
    const DeviceTreeNode* nodes,
    const int* particle_indices,
    const DeviceFmmLeaf* leaves,
    const int* far_node_indices,
    const int* near_leaf_node_indices,
    const int* particle_leaf_indices,
    double gravitational_constant,
    double softening
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count) {
        return;
    }

    const int leaf_index = particle_leaf_indices[target_index];
    if (leaf_index < 0) {
        accelerations.ax[target_index] = 0.0;
        accelerations.ay[target_index] = 0.0;
        accelerations.az[target_index] = 0.0;
        return;
    }

    const double tx = bodies.x[target_index];
    const double ty = bodies.y[target_index];
    const double tz = bodies.z[target_index];
    const DeviceFmmLeaf& leaf = leaves[leaf_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};

    for (int offset = 0; offset < leaf.far_count; ++offset) {
        const int source_node_index = far_node_indices[leaf.far_begin + offset];
        const DeviceVec3 contribution = device_multipole_acceleration_order<ExpansionOrder>(
            tx,
            ty,
            tz,
            nodes[source_node_index],
            gravitational_constant,
            softening
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
            const DeviceVec3 contribution = device_softened_acceleration(
                tx,
                ty,
                tz,
                bodies.x[source_index],
                bodies.y[source_index],
                bodies.z[source_index],
                bodies.mass[source_index],
                gravitational_constant,
                softening
            );
            acceleration.x += contribution.x;
            acceleration.y += contribution.y;
            acceleration.z += contribution.z;
        }
    }

    accelerations.ax[target_index] = acceleration.x;
    accelerations.ay[target_index] = acceleration.y;
    accelerations.az[target_index] = acceleration.z;
}

__global__ void fmm_monopole_acceleration_soa_kernel(
    DeviceBodySoA bodies,
    DeviceAccelerationSoA accelerations,
    int count,
    const DeviceMonopoleNode* nodes,
    const int* particle_indices,
    const DeviceFmmLeaf* leaves,
    const int* far_node_indices,
    const int* near_leaf_node_indices,
    const int* particle_leaf_indices,
    double gravitational_constant,
    double softening
) {
    const int target_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_index >= count) {
        return;
    }

    const int leaf_index = particle_leaf_indices[target_index];
    if (leaf_index < 0) {
        accelerations.ax[target_index] = 0.0;
        accelerations.ay[target_index] = 0.0;
        accelerations.az[target_index] = 0.0;
        return;
    }

    const double tx = bodies.x[target_index];
    const double ty = bodies.y[target_index];
    const double tz = bodies.z[target_index];
    const DeviceFmmLeaf& leaf = leaves[leaf_index];
    DeviceVec3 acceleration{0.0, 0.0, 0.0};

    for (int offset = 0; offset < leaf.far_count; ++offset) {
        const int source_node_index = far_node_indices[leaf.far_begin + offset];
        const DeviceVec3 contribution = device_monopole_acceleration(
            tx,
            ty,
            tz,
            nodes[source_node_index],
            gravitational_constant,
            softening
        );
        acceleration.x += contribution.x;
        acceleration.y += contribution.y;
        acceleration.z += contribution.z;
    }

    for (int near_offset = 0; near_offset < leaf.near_count; ++near_offset) {
        const int source_leaf_node_index = near_leaf_node_indices[leaf.near_begin + near_offset];
        const DeviceMonopoleNode& source_leaf = nodes[source_leaf_node_index];
        for (int offset = 0; offset < source_leaf.particle_count; ++offset) {
            const int source_index = particle_indices[source_leaf.particle_begin + offset];
            if (source_index == target_index) {
                continue;
            }
            const DeviceVec3 contribution = device_softened_acceleration(
                tx,
                ty,
                tz,
                bodies.x[source_index],
                bodies.y[source_index],
                bodies.z[source_index],
                bodies.mass[source_index],
                gravitational_constant,
                softening
            );
            acceleration.x += contribution.x;
            acceleration.y += contribution.y;
            acceleration.z += contribution.z;
        }
    }

    accelerations.ax[target_index] = acceleration.x;
    accelerations.ay[target_index] = acceleration.y;
    accelerations.az[target_index] = acceleration.z;
}

__global__ void drift_soa_kernel(DeviceParticleSoA particles, int count, double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    particles.x[i] += particles.vx[i] * dt;
    particles.y[i] += particles.vy[i] * dt;
    particles.z[i] += particles.vz[i] * dt;
}

__global__ void kick_soa_kernel(DeviceParticleSoA particles, int count, double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    particles.vx[i] += particles.ax[i] * dt;
    particles.vy[i] += particles.ay[i] * dt;
    particles.vz[i] += particles.az[i] * dt;
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

void copy_accelerations_back(DeviceAcceleration* device_accelerations, std::vector<Particle>& particles) {
    std::vector<DeviceAcceleration> host_accelerations(particles.size());
    throw_on_cuda(
        cudaMemcpy(
            host_accelerations.data(),
            device_accelerations,
            host_accelerations.size() * sizeof(DeviceAcceleration),
            cudaMemcpyDeviceToHost
        ),
        "copy accelerations from CUDA device"
    );

    for (std::size_t i = 0; i < particles.size(); ++i) {
        particles[i].acceleration = {
            host_accelerations[i].ax,
            host_accelerations[i].ay,
            host_accelerations[i].az,
        };
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

DeviceAcceleration* allocate_device_accelerations(std::size_t count) {
    DeviceAcceleration* device_accelerations = nullptr;
    throw_on_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_accelerations), count * sizeof(DeviceAcceleration)),
        "allocate CUDA acceleration buffer"
    );
    return device_accelerations;
}

void launch_acceleration(
    DeviceBodySoA device_bodies,
    DeviceAccelerationSoA device_accelerations,
    std::size_t count,
    const PhysicsParams& params,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    const std::size_t shared_memory_bytes = 4 * threads * sizeof(double);
    direct_tiled_acceleration_kernel<<<blocks, threads, shared_memory_bytes, stream>>>(
        device_bodies,
        device_accelerations,
        checked_int(count, "particle count"),
        params.gravitational_constant,
        params.softening
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA direct acceleration kernel");
}

void launch_tree_acceleration(
    DeviceBodySoA device_bodies,
    DeviceAccelerationSoA device_accelerations,
    std::size_t count,
    const DeviceTreeNode* device_nodes,
    std::size_t node_count,
    const int* device_particle_indices,
    const PhysicsParams& params,
    CudaTreeOptions options,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (options.expansion_order <= 2) {
        tree_acceleration_order_soa_kernel<2><<<blocks, threads, 0, stream>>>(
            device_bodies,
            device_accelerations,
            checked_int(count, "particle count"),
            device_nodes,
            checked_int(node_count, "tree node count"),
            device_particle_indices,
            params.gravitational_constant,
            params.softening,
            options.theta
        );
    } else {
        tree_acceleration_order_soa_kernel<4><<<blocks, threads, 0, stream>>>(
            device_bodies,
            device_accelerations,
            checked_int(count, "particle count"),
            device_nodes,
            checked_int(node_count, "tree node count"),
            device_particle_indices,
            params.gravitational_constant,
            params.softening,
            options.theta
        );
    }
    throw_on_cuda(cudaGetLastError(), "launch CUDA tree acceleration kernel");
}

void launch_tree_monopole_acceleration(
    DeviceBodySoA device_bodies,
    DeviceAccelerationSoA device_accelerations,
    std::size_t count,
    const DeviceMonopoleNode* device_nodes,
    std::size_t node_count,
    const int* device_particle_indices,
    const PhysicsParams& params,
    CudaTreeOptions options,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    tree_monopole_acceleration_soa_kernel<<<blocks, threads, 0, stream>>>(
        device_bodies,
        device_accelerations,
        checked_int(count, "particle count"),
        device_nodes,
        checked_int(node_count, "tree node count"),
        device_particle_indices,
        params.gravitational_constant,
        params.softening,
        options.theta
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA monopole tree acceleration kernel");
}

void launch_fmm_acceleration(
    DeviceBodySoA device_bodies,
    DeviceAccelerationSoA device_accelerations,
    std::size_t count,
    const DeviceTreeNode* device_nodes,
    const int* device_particle_indices,
    const DeviceFmmLeaf* device_leaves,
    const int* device_far_node_indices,
    const int* device_near_leaf_node_indices,
    const int* device_particle_leaf_indices,
    const PhysicsParams& params,
    CudaTreeOptions options,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (options.expansion_order <= 2) {
        fmm_acceleration_order_soa_kernel<2><<<blocks, threads, 0, stream>>>(
            device_bodies,
            device_accelerations,
            checked_int(count, "particle count"),
            device_nodes,
            device_particle_indices,
            device_leaves,
            device_far_node_indices,
            device_near_leaf_node_indices,
            device_particle_leaf_indices,
            params.gravitational_constant,
            params.softening
        );
    } else {
        fmm_acceleration_order_soa_kernel<4><<<blocks, threads, 0, stream>>>(
            device_bodies,
            device_accelerations,
            checked_int(count, "particle count"),
            device_nodes,
            device_particle_indices,
            device_leaves,
            device_far_node_indices,
            device_near_leaf_node_indices,
            device_particle_leaf_indices,
            params.gravitational_constant,
            params.softening
        );
    }
    throw_on_cuda(cudaGetLastError(), "launch CUDA FMM acceleration kernel");
}

void launch_fmm_monopole_acceleration(
    DeviceBodySoA device_bodies,
    DeviceAccelerationSoA device_accelerations,
    std::size_t count,
    const DeviceMonopoleNode* device_nodes,
    const int* device_particle_indices,
    const DeviceFmmLeaf* device_leaves,
    const int* device_far_node_indices,
    const int* device_near_leaf_node_indices,
    const int* device_particle_leaf_indices,
    const PhysicsParams& params,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    fmm_monopole_acceleration_soa_kernel<<<blocks, threads, 0, stream>>>(
        device_bodies,
        device_accelerations,
        checked_int(count, "particle count"),
        device_nodes,
        device_particle_indices,
        device_leaves,
        device_far_node_indices,
        device_near_leaf_node_indices,
        device_particle_leaf_indices,
        params.gravitational_constant,
        params.softening
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA monopole FMM acceleration kernel");
}

void launch_kick(DeviceParticleSoA device_particles, std::size_t count, double dt, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    kick_soa_kernel<<<blocks, threads, 0, stream>>>(
        device_particles,
        checked_int(count, "particle count"),
        dt
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA kick kernel");
}

void launch_drift(DeviceParticleSoA device_particles, std::size_t count, double dt, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    drift_soa_kernel<<<blocks, threads, 0, stream>>>(
        device_particles,
        checked_int(count, "particle count"),
        dt
    );
    throw_on_cuda(cudaGetLastError(), "launch CUDA drift kernel");
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

    CudaWorkspace& workspace = cuda_workspace();
    cudaStream_t stream = workspace.stream();
    upload_body_arrays(workspace, particles, stream);
    launch_acceleration(
        workspace.body_arrays(),
        workspace.acceleration_arrays(),
        particles.size(),
        params,
        stream
    );
    download_acceleration_arrays(workspace, particles, stream);
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

    CudaWorkspace& workspace = cuda_workspace();
    cudaStream_t stream = workspace.stream();
    upload_particle_arrays(workspace, particles, stream);
    launch_kick(workspace.particle_arrays(), particles.size(), 0.5 * dt, stream);
    launch_drift(workspace.particle_arrays(), particles.size(), dt, stream);
    launch_acceleration(
        workspace.body_arrays(),
        workspace.acceleration_arrays(),
        particles.size(),
        params,
        stream
    );
    launch_kick(workspace.particle_arrays(), particles.size(), 0.5 * dt, stream);
    download_particle_arrays(workspace, particles, stream);
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

    CudaWorkspace& workspace = cuda_workspace();
    cudaStream_t stream = workspace.stream();
    upload_body_arrays(workspace, particles, stream);

    const std::vector<int> host_particle_indices = pack_particle_indices(tree.particle_indices);
    upload_vector(
        host_particle_indices,
        workspace.host_particle_indices,
        workspace.particle_indices,
        stream,
        "allocate pinned host tree particle indices",
        "allocate CUDA tree particle indices",
        "copy CUDA tree particle indices"
    );

    if (options.expansion_order <= 0) {
        const std::vector<DeviceMonopoleNode> host_nodes = pack_monopole_nodes(tree);
        upload_vector(
            host_nodes,
            workspace.host_monopole_nodes,
            workspace.monopole_nodes,
            stream,
            "allocate pinned host monopole tree nodes",
            "allocate CUDA monopole tree nodes",
            "copy CUDA monopole tree nodes"
        );

        launch_tree_monopole_acceleration(
            workspace.body_arrays(),
            workspace.acceleration_arrays(),
            particles.size(),
            workspace.monopole_nodes.data(),
            host_nodes.size(),
            workspace.particle_indices.data(),
            params,
            options,
            stream
        );
        download_acceleration_arrays(workspace, particles, stream);
        return;
    }

    const std::vector<DeviceTreeNode> host_nodes = pack_tree_nodes(tree);
    upload_vector(
        host_nodes,
        workspace.host_tree_nodes,
        workspace.tree_nodes,
        stream,
        "allocate pinned host tree nodes",
        "allocate CUDA tree nodes",
        "copy CUDA tree nodes"
    );

    launch_tree_acceleration(
        workspace.body_arrays(),
        workspace.acceleration_arrays(),
        particles.size(),
        workspace.tree_nodes.data(),
        host_nodes.size(),
        workspace.particle_indices.data(),
        params,
        options,
        stream
    );
    download_acceleration_arrays(workspace, particles, stream);
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

    CudaWorkspace& workspace = cuda_workspace();
    cudaStream_t stream = workspace.stream();
    upload_body_arrays(workspace, particles, stream);

    const std::vector<int> host_particle_indices = pack_particle_indices(fmm.tree.particle_indices);
    std::vector<DeviceFmmLeaf> host_leaves;
    host_leaves.reserve(fmm.leaves.size());
    for (const FlatFmmLeaf& leaf : fmm.leaves) {
        host_leaves.push_back(pack_fmm_leaf(leaf));
    }

    upload_vector(
        host_particle_indices,
        workspace.host_particle_indices,
        workspace.particle_indices,
        stream,
        "allocate pinned host FMM particle indices",
        "allocate CUDA FMM particle indices",
        "copy CUDA FMM particle indices"
    );
    upload_vector(
        host_leaves,
        workspace.host_leaves,
        workspace.leaves,
        stream,
        "allocate pinned host FMM leaves",
        "allocate CUDA FMM leaves",
        "copy CUDA FMM leaves"
    );
    upload_vector(
        fmm.far_node_indices,
        workspace.host_far_node_indices,
        workspace.far_node_indices,
        stream,
        "allocate pinned host FMM far-list indices",
        "allocate CUDA FMM far-list indices",
        "copy CUDA FMM far-list indices"
    );
    upload_vector(
        fmm.near_leaf_node_indices,
        workspace.host_near_leaf_node_indices,
        workspace.near_leaf_node_indices,
        stream,
        "allocate pinned host FMM near-list indices",
        "allocate CUDA FMM near-list indices",
        "copy CUDA FMM near-list indices"
    );
    upload_vector(
        fmm.particle_leaf_indices,
        workspace.host_particle_leaf_indices,
        workspace.particle_leaf_indices,
        stream,
        "allocate pinned host FMM particle-leaf indices",
        "allocate CUDA FMM particle-leaf indices",
        "copy CUDA FMM particle-leaf indices"
    );

    if (options.expansion_order <= 0) {
        const std::vector<DeviceMonopoleNode> host_nodes = pack_monopole_nodes(fmm.tree);
        upload_vector(
            host_nodes,
            workspace.host_monopole_nodes,
            workspace.monopole_nodes,
            stream,
            "allocate pinned host monopole FMM nodes",
            "allocate CUDA monopole FMM nodes",
            "copy CUDA monopole FMM nodes"
        );

        launch_fmm_monopole_acceleration(
            workspace.body_arrays(),
            workspace.acceleration_arrays(),
            particles.size(),
            workspace.monopole_nodes.data(),
            workspace.particle_indices.data(),
            workspace.leaves.data(),
            workspace.far_node_indices.data(),
            workspace.near_leaf_node_indices.data(),
            workspace.particle_leaf_indices.data(),
            params,
            stream
        );
        download_acceleration_arrays(workspace, particles, stream);
        return;
    }

    const std::vector<DeviceTreeNode> host_nodes = pack_tree_nodes(fmm.tree);
    upload_vector(
        host_nodes,
        workspace.host_tree_nodes,
        workspace.tree_nodes,
        stream,
        "allocate pinned host FMM nodes",
        "allocate CUDA FMM nodes",
        "copy CUDA FMM nodes"
    );

    launch_fmm_acceleration(
        workspace.body_arrays(),
        workspace.acceleration_arrays(),
        particles.size(),
        workspace.tree_nodes.data(),
        workspace.particle_indices.data(),
        workspace.leaves.data(),
        workspace.far_node_indices.data(),
        workspace.near_leaf_node_indices.data(),
        workspace.particle_leaf_indices.data(),
        params,
        options,
        stream
    );
    download_acceleration_arrays(workspace, particles, stream);
}

void cuda_tree_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
    const PhysicsParams& params,
    CudaTreeOptions options
) {
    if (particles.empty()) {
        return;
    }
    if (!cuda_solver_available()) {
        auto compute = [&params, options](std::vector<Particle>& state) {
            compute_tree_accelerations(
                state,
                params,
                options.theta,
                options.leaf_capacity,
                options.expansion_order
            );
        };
        leapfrog_step(particles, dt, compute);
        return;
    }

    checked_int(particles.size(), "particle count");
    CudaWorkspace& workspace = cuda_workspace();
    cudaStream_t stream = workspace.stream();
    upload_particle_arrays(workspace, particles, stream);
    launch_kick(workspace.particle_arrays(), particles.size(), 0.5 * dt, stream);
    launch_drift(workspace.particle_arrays(), particles.size(), dt, stream);
    download_particle_arrays(workspace, particles, stream);

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

    const std::vector<int> host_particle_indices = pack_particle_indices(tree.particle_indices);
    upload_vector(
        host_particle_indices,
        workspace.host_particle_indices,
        workspace.particle_indices,
        stream,
        "allocate pinned host tree particle indices",
        "allocate CUDA tree particle indices",
        "copy CUDA tree particle indices"
    );

    if (options.expansion_order <= 0) {
        const std::vector<DeviceMonopoleNode> host_nodes = pack_monopole_nodes(tree);
        upload_vector(
            host_nodes,
            workspace.host_monopole_nodes,
            workspace.monopole_nodes,
            stream,
            "allocate pinned host monopole tree nodes",
            "allocate CUDA monopole tree nodes",
            "copy CUDA monopole tree nodes"
        );
        launch_tree_monopole_acceleration(
            workspace.body_arrays(),
            workspace.acceleration_arrays(),
            particles.size(),
            workspace.monopole_nodes.data(),
            host_nodes.size(),
            workspace.particle_indices.data(),
            params,
            options,
            stream
        );
    } else {
        const std::vector<DeviceTreeNode> host_nodes = pack_tree_nodes(tree);
        upload_vector(
            host_nodes,
            workspace.host_tree_nodes,
            workspace.tree_nodes,
            stream,
            "allocate pinned host tree nodes",
            "allocate CUDA tree nodes",
            "copy CUDA tree nodes"
        );
        launch_tree_acceleration(
            workspace.body_arrays(),
            workspace.acceleration_arrays(),
            particles.size(),
            workspace.tree_nodes.data(),
            host_nodes.size(),
            workspace.particle_indices.data(),
            params,
            options,
            stream
        );
    }

    launch_kick(workspace.particle_arrays(), particles.size(), 0.5 * dt, stream);
    download_particle_arrays(workspace, particles, stream);
}

void cuda_fmm_leapfrog_step(
    std::vector<Particle>& particles,
    double dt,
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
        auto compute = [&params, fmm_options](std::vector<Particle>& state) {
            compute_fmm_accelerations(state, params, fmm_options);
        };
        leapfrog_step(particles, dt, compute);
        return;
    }

    checked_int(particles.size(), "particle count");
    CudaWorkspace& workspace = cuda_workspace();
    cudaStream_t stream = workspace.stream();
    upload_particle_arrays(workspace, particles, stream);
    launch_kick(workspace.particle_arrays(), particles.size(), 0.5 * dt, stream);
    launch_drift(workspace.particle_arrays(), particles.size(), dt, stream);
    download_particle_arrays(workspace, particles, stream);

    FmmOptions fmm_options;
    fmm_options.theta = options.theta;
    fmm_options.leaf_capacity = options.leaf_capacity;
    fmm_options.max_depth = options.max_depth;
    fmm_options.expansion_order = options.expansion_order;
    const FlatFmmData fmm = build_flat_fmm(particles, params, fmm_options);
    if (fmm.tree.nodes.empty() || fmm.leaves.empty()) {
        return;
    }

    const std::vector<int> host_particle_indices = pack_particle_indices(fmm.tree.particle_indices);
    std::vector<DeviceFmmLeaf> host_leaves;
    host_leaves.reserve(fmm.leaves.size());
    for (const FlatFmmLeaf& leaf : fmm.leaves) {
        host_leaves.push_back(pack_fmm_leaf(leaf));
    }

    upload_vector(
        host_particle_indices,
        workspace.host_particle_indices,
        workspace.particle_indices,
        stream,
        "allocate pinned host FMM particle indices",
        "allocate CUDA FMM particle indices",
        "copy CUDA FMM particle indices"
    );
    upload_vector(
        host_leaves,
        workspace.host_leaves,
        workspace.leaves,
        stream,
        "allocate pinned host FMM leaves",
        "allocate CUDA FMM leaves",
        "copy CUDA FMM leaves"
    );
    upload_vector(
        fmm.far_node_indices,
        workspace.host_far_node_indices,
        workspace.far_node_indices,
        stream,
        "allocate pinned host FMM far-list indices",
        "allocate CUDA FMM far-list indices",
        "copy CUDA FMM far-list indices"
    );
    upload_vector(
        fmm.near_leaf_node_indices,
        workspace.host_near_leaf_node_indices,
        workspace.near_leaf_node_indices,
        stream,
        "allocate pinned host FMM near-list indices",
        "allocate CUDA FMM near-list indices",
        "copy CUDA FMM near-list indices"
    );
    upload_vector(
        fmm.particle_leaf_indices,
        workspace.host_particle_leaf_indices,
        workspace.particle_leaf_indices,
        stream,
        "allocate pinned host FMM particle-leaf indices",
        "allocate CUDA FMM particle-leaf indices",
        "copy CUDA FMM particle-leaf indices"
    );

    if (options.expansion_order <= 0) {
        const std::vector<DeviceMonopoleNode> host_nodes = pack_monopole_nodes(fmm.tree);
        upload_vector(
            host_nodes,
            workspace.host_monopole_nodes,
            workspace.monopole_nodes,
            stream,
            "allocate pinned host monopole FMM nodes",
            "allocate CUDA monopole FMM nodes",
            "copy CUDA monopole FMM nodes"
        );
        launch_fmm_monopole_acceleration(
            workspace.body_arrays(),
            workspace.acceleration_arrays(),
            particles.size(),
            workspace.monopole_nodes.data(),
            workspace.particle_indices.data(),
            workspace.leaves.data(),
            workspace.far_node_indices.data(),
            workspace.near_leaf_node_indices.data(),
            workspace.particle_leaf_indices.data(),
            params,
            stream
        );
    } else {
        const std::vector<DeviceTreeNode> host_nodes = pack_tree_nodes(fmm.tree);
        upload_vector(
            host_nodes,
            workspace.host_tree_nodes,
            workspace.tree_nodes,
            stream,
            "allocate pinned host FMM nodes",
            "allocate CUDA FMM nodes",
            "copy CUDA FMM nodes"
        );
        launch_fmm_acceleration(
            workspace.body_arrays(),
            workspace.acceleration_arrays(),
            particles.size(),
            workspace.tree_nodes.data(),
            workspace.particle_indices.data(),
            workspace.leaves.data(),
            workspace.far_node_indices.data(),
            workspace.near_leaf_node_indices.data(),
            workspace.particle_leaf_indices.data(),
            params,
            options,
            stream
        );
    }

    launch_kick(workspace.particle_arrays(), particles.size(), 0.5 * dt, stream);
    download_particle_arrays(workspace, particles, stream);
}

}  // namespace fmmgalaxy
