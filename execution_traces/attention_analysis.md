# FlashAttention: Execution Trace and Optimization Analysis

## 1. Background and Motivation

The standard scaled dot-product attention mechanism computes:

QKᵀ → Softmax → V

For a sequence length N and head dimension d, this results in:
- O(N²) memory reads/writes
- Large intermediate matrices (attention scores)
- Severe pressure on GPU memory bandwidth

On modern GPUs, **memory bandwidth**, not compute throughput,
becomes the dominant bottleneck for attention.

FlashAttention addresses this by redesigning the attention computation
to minimize global memory traffic while maintaining numerical correctness.

---

## 2. Naive Attention Execution on GPU

### Execution Steps

1. Load Q and K from global memory
2. Compute QKᵀ using GEMM
3. Write attention scores to global memory
4. Read attention scores for Softmax
5. Write Softmax output to global memory
6. Read Softmax output
7. Multiply by V
8. Write final output

### Key Issues

- Attention score matrix (N×N) does not fit in cache
- Multiple full reads/writes of the same data
- Low arithmetic intensity
- Memory-bound execution

### Bottleneck Classification

| Component | Limitation |
|--------|-----------|
Global Memory | Bandwidth saturation |
L2 Cache | Thrashing |
Tensor Cores | Underutilized |
Latency Hiding | Ineffective due to memory stalls |

---

## 3. FlashAttention High-Level Strategy

FlashAttention fuses:
- QKᵀ
- Softmax
- V multiplication

into a **single kernel**, operating on **small tiles** that fit
entirely in on-chip memory.

### Core Idea

> Never materialize the full attention matrix in global memory.

Instead:
- Stream blocks of K and V
- Keep partial softmax statistics in registers
- Accumulate output incrementally

---

## 4. FlashAttention Execution Trace (GPU Perspective)

### Step-by-Step Execution

1. Load a tile of Q into registers
2. Load a tile of K into shared memory (async)
3. Compute partial QKᵀ using Tensor Cores
4. Update running softmax (max + sum) in registers
5. Load corresponding V tile
6. Accumulate output tile in registers
7. Advance to next K/V tile
8. Write final output once

### Key Observations

- No global writes of attention scores
- Shared memory reused aggressively
- Registers hold softmax state
- Tensor Cores operate near peak throughput

---

## 5. Mapping to GPU Optimization Techniques

| Technique | Usage in FlashAttention |
|--------|------------------------|
Thread Block Tiling | Attention computed in small Q×K tiles |
Register Blocking | Softmax stats and output accumulation |
Double Buffering | Overlap K/V loads with computation |
Vectorized Loads | Efficient global memory access |
Warp Shuffles | Softmax reductions |
Persistent Kernel Behavior | Long-lived blocks over sequence tiles |

(See `optimizations/techniques.yaml` and `optimizations/code_samples/`)

---

## 6. Quantitative Impact

### Memory Traffic Reduction

- Standard Attention: O(N²) global memory traffic
- FlashAttention: O(N) global memory traffic

### Performance Effects

| Metric | Standard | FlashAttention |
|-----|--------|---------------|
DRAM Reads/Writes | High | Minimal |
Tensor Core Utilization | Low | High |
L2 Cache Hit Rate | Low | High |
Kernel Launches | Multiple | Single |

Empirical results show:
- 2–4× speedup for long sequences
- Near-peak Tensor Core utilization on Ampere/Hopper

(Source: FlashAttention Paper)

---

## 7. Architectural Implications

FlashAttention demonstrates that:
- Compute-bound kernels can be recovered from memory-bound formulations
- On-chip memory (registers + shared memory) is the critical resource
- Kernel fusion is essential for modern ML workloads

This aligns with trends seen in:
- CUTLASS
- Triton
- Fused Transformer kernels

---

## 8. CPU Implications

FlashAttention highlights challenges for CPUs:

- Lack of explicit on-chip memory management
- Limited async memory primitives
- Lower tolerance for fine-grained tiling

Potential CPU directions:
- Software-managed scratchpads
- Wider SIMD with explicit reductions
- Fused attention kernels in inference runtimes

---

## 9. Summary

FlashAttention is not merely an algorithmic improvement;
it is a **hardware-conscious reformulation** of attention.

Its success underscores the importance of:
- Memory hierarchy awareness
- Latency hiding
- Kernel fusion

These principles are broadly applicable beyond attention,
informing future GPU and CPU inference system design.
