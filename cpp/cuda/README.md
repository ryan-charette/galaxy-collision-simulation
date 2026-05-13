# CUDA Layer

This module implements optional CUDA acceleration for the direct/P2P force path.

Implemented:

- GPU direct acceleration kernel,
- GPU kick/drift/kick leapfrog step for `cuda-direct`,
- GPU per-particle evaluation kernels for `cuda-tree` and `cuda-fmm`,
- CPU fallback symbols when CUDA is unavailable,
- runtime availability reporting from the CLI.

Current tradeoff:

- The tree/FMM builders still run on CPU, then flattened node and interaction-list data is copied to the GPU for per-particle force evaluation. For high-scale benchmarks, use `fmm_expansion_order = 0` and `[output] format = "none"` to avoid p=4 expansion and CSV/diagnostic overhead dominating the run.
