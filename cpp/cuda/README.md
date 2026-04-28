# CUDA Layer

This module implements optional CUDA acceleration for the direct/P2P force path.

Implemented:

- GPU direct acceleration kernel,
- GPU kick/drift/kick leapfrog step for `cuda-direct`,
- CPU fallback symbols when CUDA is unavailable,
- runtime availability reporting from the CLI.

Current tradeoff:

- The FMM far-field pass runs on CPU. The CUDA kernel covers the direct all-pairs solver and provides the kernel surface for future FMM near-field leaf-pair acceleration.
