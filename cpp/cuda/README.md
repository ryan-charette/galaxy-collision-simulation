# CUDA Layer

This module implements optional CUDA acceleration for direct, Barnes-Hut tree,
and FMM force evaluation paths.

Implemented:

- GPU direct acceleration kernel,
- GPU kick/drift/kick leapfrog step for `cuda-direct`,
- GPU per-particle evaluation kernels for `cuda-tree` and `cuda-fmm`,
- reusable CUDA workspace buffers to avoid per-step allocation churn,
- pinned host staging with stream-ordered asynchronous transfers,
- SoA position/mass force inputs for coalesced reads,
- shared-memory tiling for `cuda-direct`,
- specialized p=0, p=2, and p=4 tree/FMM CUDA kernels,
- GPU kick/drift/final-kick wrappers for `cuda-tree` and `cuda-fmm`,
- CPU fallback symbols when CUDA is unavailable,
- runtime availability reporting from the CLI.

Performance notes:

- The tree/FMM builders still run on CPU, then flattened node and interaction-list data is copied to the GPU for per-particle force evaluation.
- `fmm_expansion_order = 0` selects the lowest-cost monopole path. Higher-order runs use specialized p=2 or p=4 kernels instead of a runtime-polymorphic multipole loop.
- `cuda-tree` and `cuda-fmm` keep kick/drift/final-kick work on the GPU; positions are copied back after drift because tree construction still happens on the CPU.
- `[output] format = "none"` avoids CSV and diagnostic overhead dominating large runs.
