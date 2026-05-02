# Testing and Validation Plan

## Unit tests

Initial C++ unit tests should cover:

- vector arithmetic,
- particle initialization through generated galaxies,
- pairwise force symmetry,
- finite acceleration under softening,
- leapfrog update consistency,
- octree Barnes-Hut force agreement against direct summation,
- `p=4` FMM force agreement against direct summation,
- CUDA direct solver agreement with the CPU direct solver or CPU fallback,
- MPI ownership range decomposition,
- diagnostics sanity,
- config parsing,
- snapshot writing.

These are covered by `cpp/tests/smoke_tests.cpp` and registered with CTest as `smoke_tests`.

## Physics sanity tests

### Two-body orbit

Expected behavior:

- approximately closed orbit for small timestep,
- bounded energy drift,
- total momentum conservation.

### Isolated disk

Expected behavior:

- disk remains coherent for a reasonable number of dynamical times,
- inner particles are more timestep-sensitive,
- no immediate numerical explosion under default softening.

### Head-on collision

Expected behavior:

- symmetric morphology for equal-mass identical disks,
- conservation diagnostics remain interpretable,
- close encounter remains stable due to softening.

## Solver validation

Compare direct sum and approximate solver accelerations.

Metrics:

```text
relative_error_i = ||a_approx_i - a_direct_i|| / max(||a_direct_i||, tiny)
mean_relative_error
median_relative_error
p95_relative_error
max_relative_error
```

## Parallel validation

Compare serial and MPI outputs using identical seeds/configs.

Expected:

- small floating-point differences are acceptable,
- statistical diagnostics should match,
- aggregate mass, momentum, and total particle count should match exactly or within strict tolerance.

## CUDA validation

Compare CPU and GPU kernels for:

- accelerations,
- integrated positions,
- integrated velocities,
- diagnostics.

Use tolerance-based tests rather than exact equality.
