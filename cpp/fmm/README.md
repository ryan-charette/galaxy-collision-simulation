# FMM Solver

This module contains both the earlier Barnes-Hut treecode and a monopole FMM solver. The FMM is intentionally low order (`p=0`) today, but it uses the explicit pass structure needed for later high-order expansions.

Implemented:

- quadtree node storage,
- tree construction,
- P2M/M2M mass and center-of-mass upward aggregation,
- M2L-style far-cell interaction lists,
- P2P direct near-field leaf interactions,
- target-range evaluation for MPI-owned particles,
- configurable opening angle `theta`,
- smoke-test comparison against the direct solver.

Roadmap:

- high-order P2M/M2M/M2L/L2L/L2P coefficients,
- runtime/error benchmark plots.
