# MPI Layer

This module implements the first distributed execution path.

Implemented:

- rank-local particle ownership,
- contiguous particle-count decomposition,
- all-rank particle synchronization through `MPI_Allgatherv`,
- owned-particle direct and FMM acceleration evaluation,
- rank-0 coordinated snapshot and diagnostics output.

Current tradeoff:

- Every rank keeps a full synchronized particle array. This keeps the first MPI implementation simple and deterministic. The next scaling step is spatial decomposition with tree-summary exchange and ghost/near-field exchange.
