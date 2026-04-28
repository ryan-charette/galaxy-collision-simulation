# FMM Solver

This module contains a Barnes-Hut quadtree solver as the first hierarchical force approximation. It is intentionally placed in the FMM module because it shares the same tree construction and center-of-mass aggregation path that later high-order FMM passes will extend.

Implemented:

- quadtree node storage,
- tree construction,
- mass and center-of-mass upward aggregation,
- configurable opening angle `theta`,
- direct near-field leaf interactions,
- smoke-test comparison against the direct solver.

Roadmap:

- P2M, M2M, M2L, L2L, L2P, and P2P FMM passes,
- approximation-order configuration,
- runtime/error benchmark plots.
