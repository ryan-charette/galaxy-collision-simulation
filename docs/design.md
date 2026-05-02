# Technical Design: Distributed FMM Galaxy Collision Simulator

## 1. Objective

This project will implement a high-performance gravitational N-body simulator for galaxy collision experiments. The implementation will progress from a direct-sum baseline to a single-node FMM solver, then to distributed MPI execution and selective CUDA acceleration.

The project is designed around four goals:

1. correctness against direct-sum baselines,
2. scalable long-range force evaluation,
3. reproducible galaxy collision experiments,
4. high-quality Python visualization and benchmarking.

## 2. Simulation model

### 2.1 Particle state

Each particle stores:

- position `x`, `y`, optionally `z` later,
- velocity `vx`, `vy`, optionally `vz` later,
- acceleration `ax`, `ay`, optionally `az` later,
- mass `m`,
- group ID, e.g. source galaxy ID,
- optional rendering attributes.

The current simulator stores 3D position, velocity, and acceleration. Planar 2D experiments remain available as the special case where `z = 0`.

### 2.2 Units

Use nondimensional code units initially:

- `G = 1`,
- characteristic length `R = 1`,
- characteristic mass `M = 1`,
- characteristic time derived from the chosen velocity scale.

A later galaxy-facing config layer can map these to physical units such as kpc and Myr.

### 2.3 Softened gravity

Use Plummer-style softening to avoid numerical instability during close encounters.

For particles `i` and `j`:

```text
r_ij = x_j - x_i
s2   = dot(r_ij, r_ij) + eps^2
a_i += G * m_j * r_ij / s2^(3/2)
```

Softening parameter `eps` is configurable per run.

### 2.4 Time integration

Use leapfrog integration as the initial default because it is simple and symplectic.

Kick-drift-kick form:

```text
v_{n+1/2} = v_n       + 0.5 * dt * a(x_n)
x_{n+1}   = x_n       + dt * v_{n+1/2}
a_{n+1}   = a(x_{n+1})
v_{n+1}   = v_{n+1/2} + 0.5 * dt * a_{n+1}
```

Future extension:

- adaptive timestep during close encounters,
- block timestepping,
- higher-order symplectic schemes.

## 3. Solver roadmap

### 3.1 Direct-sum baseline

The direct solver computes all pairwise interactions in `O(N^2)` time. This is required for:

- correctness validation,
- force-error measurement,
- small-system debugging,
- benchmark comparison.

### 3.2 Tree infrastructure

The hierarchical data structure is now a 3D octree.

Each node stores:

- bounding box center and half-width,
- total mass,
- center of mass,
- child indices,
- particle range or particle indices for leaves,
- multipole coefficients,
- local expansion coefficients,
- interaction lists.

### 3.3 Transitional Barnes-Hut/treecode mode

Before the full FMM pass is complete, the tree can support a Barnes-Hut-style approximation:

```text
if node_size / distance < theta:
    approximate node by aggregate mass
else:
    recurse into children
```

This stage is useful for debugging tree construction, center-of-mass calculations, and performance comparisons.

### 3.4 FMM pass structure

The intended FMM pipeline is:

1. **P2M:** convert particles in a leaf node into multipole coefficients,
2. **M2M:** aggregate child multipoles upward into parent multipoles,
3. **M2L:** convert well-separated source node multipoles into target node local expansions,
4. **L2L:** propagate local expansions downward from parent to child,
5. **L2P:** evaluate local expansions at particle positions,
6. **P2P:** compute direct interactions for near-field particles/nodes.

The current implementation supports `p=0` monopole, `p=2` quadrupole, and `p=4` fourth-order Cartesian moments. Orders above `p=4` are intentionally out of scope for this project.

### 3.5 Near/far criteria

The MVP will use a geometric well-separated test based on node size and node separation. Later versions can implement level-restricted trees and standard FMM interaction lists.

## 4. MPI distributed design

### 4.1 Initial decomposition

Start with particle-count decomposition:

```text
rank k owns particles [start_k, end_k)
```

This is easier to implement and validate.

### 4.2 Later decomposition

Upgrade to spatial decomposition using octree partitions or space-filling curves, such as Morton/Z-order keys.

### 4.3 Required MPI operations

- global particle count synchronization,
- global bounding box reduction,
- rank-local tree construction,
- exchange of far-field summaries,
- exchange of ghost or boundary particles for near-field calculations,
- reduction of diagnostic quantities,
- coordinated snapshot output.

### 4.4 MPI metrics

Measure:

- compute time,
- communication time,
- load imbalance,
- strong scaling,
- weak scaling,
- parallel efficiency.

## 5. CUDA design

CUDA acceleration should be introduced only after CPU profiling identifies bottlenecks.

Likely first CUDA targets:

1. near-field P2P interactions,
2. particle integration,
3. local expansion evaluation,
4. particle-to-multipole accumulation.

### 5.1 Data layout

Use structure-of-arrays layout for GPU-friendly memory access:

```cpp
struct ParticleArrays {
    double* x;
    double* y;
    double* vx;
    double* vy;
    double* ax;
    double* ay;
    double* mass;
};
```

Avoid an array-of-structs layout in CUDA kernels unless profiling justifies it.

### 5.2 CUDA metrics

Measure:

- kernel time,
- host-device transfer time,
- achieved speedup vs CPU,
- occupancy and memory bandwidth where practical.

## 6. Snapshot I/O

### 6.1 Preferred format

Use HDF5 if available. If HDF5 creates build friction, use a portable binary format plus JSON metadata first.

Target schema:

```text
/snapshots/step_000000/positions     shape [N, dim]
/snapshots/step_000000/velocities    shape [N, dim]
/snapshots/step_000000/accelerations shape [N, dim], optional
/snapshots/step_000000/masses        shape [N]
/snapshots/step_000000/group_id      shape [N]
/snapshots/step_000000/time          scalar
/metadata/config                      JSON string
/metadata/git_commit                  string, optional
```

### 6.2 Output cadence

Simulation timestep and snapshot cadence should be independent:

```text
steps = 10000
snapshot_every = 25
```

## 7. Galaxy initial conditions

The galaxy generator should support:

- exponential disk profile,
- approximate circular velocities,
- optional bulge component,
- optional halo approximation,
- galaxy origin position,
- galaxy bulk velocity,
- orientation/inclination,
- group ID assignment.

Collision configs should expose:

- mass ratio,
- impact parameter,
- relative velocity,
- orientation,
- initial separation,
- softening,
- timestep,
- simulation duration.

## 8. Validation plan

### 8.1 Physics validation

- two-body orbit stability,
- circular disk stability,
- total momentum conservation,
- angular momentum tracking,
- energy drift over time.

### 8.2 Solver validation

Compare approximate solvers to direct sum:

- mean relative force error,
- median relative force error,
- max relative force error,
- runtime vs particle count,
- memory usage.

### 8.3 Distributed validation

Compare single-process and MPI runs with identical seeds/configs.

## 9. Final demo plan

The final project should produce:

- one cinematic galaxy collision MP4,
- one shorter README GIF,
- benchmark plots,
- direct/Barnes-Hut/FMM comparison table,
- architecture diagram,
- reproducible configs.

## 10. Implementation priorities

1. Direct-sum correctness.
2. Snapshot output and Python loader.
3. Octree construction.
4. Treecode approximation.
5. FMM passes.
6. Galaxy initial conditions.
7. MPI distribution.
8. CUDA hot kernels.
9. Polished visualization.
