# GPU Optimization Code Samples

These minimal CUDA kernels demonstrate core GPU performance optimization
techniques used in high-performance libraries such as CUTLASS, cuBLAS,
Triton, and FlashAttention.

Each sample isolates a single technique for clarity.
The kernels are intentionally simplified and not tuned for peak performance.

## Contents

1. Thread Block Tiling
2. Register Blocking
3. Double Buffering
4. Vectorized Memory Loads
5. Warp-Level Primitives
6. Persistent Kernels

All examples target NVIDIA GPUs and CUDA.
