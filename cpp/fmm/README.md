# FMM Solver

This module contains both the Barnes-Hut treecode and a 3D octree FMM solver. The FMM supports monopole (`p=0`) and quadrupole (`p=2`) far-field coefficients.

Implemented:

- octree node storage,
- tree construction,
- P2M/M2M mass, center-of-mass, and quadrupole upward aggregation,
- M2L-style far-cell interaction lists,
- P2P direct near-field leaf interactions,
- target-range evaluation for MPI-owned particles,
- configurable opening angle `theta`,
- smoke-test comparison against the direct solver.

Roadmap:

- orders above quadrupole,
- runtime/error benchmark plots.
