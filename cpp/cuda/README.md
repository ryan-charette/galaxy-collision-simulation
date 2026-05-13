# CUDA Layer

This module implements optional CUDA acceleration for the direct/P2P force path.

Implemented:

- GPU direct acceleration kernel,
- GPU kick/drift/kick leapfrog step for `cuda-direct`,
- GPU per-particle evaluation kernels for `cuda-tree` and `cuda-fmm`,
- CPU fallback symbols when CUDA is unavailable,
- runtime availability reporting from the CLI.

Performance notes:

- The tree/FMM builders still run on CPU, then flattened node and interaction-list data is copied to the GPU for per-particle force evaluation.
- `fmm_expansion_order = 0` selects a specialized monopole CUDA path with compact node buffers, compact position/mass input, acceleration-only output copies, and no CPU multipole-moment construction. This is the preferred high-scale benchmark mode.
- `[output] format = "none"` avoids CSV and diagnostic overhead dominating large runs.
