# Workload Bottleneck Analysis

## Objective

This document classifies representative machine learning workloads
based on their primary GPU performance bottlenecks.

The analysis combines:
- Architectural understanding
- Execution tracing
- Known profiling behavior from literature

Measured profiler data (NCU/rocprof) will later be used to validate
these classifications.

---

## Bottleneck Classification Categories

### Compute-Bound
- Tensor/Core pipelines saturated
- Memory system underutilized
- Performance scales with clock and compute units

### Memory-Bound
- DRAM bandwidth saturated
- Low arithmetic intensity
- Poor cache reuse

### Mixed
- Alternates between compute- and memory-bound phases
- Common in end-to-end ML workloads

---

## 1. BERT (Encoder Inference)

### Dominant Kernels
- GEMM (QKV projections)
- Attention
- LayerNorm

### Bottleneck Classification
**Mixed**

### Primary Bottlenecks
- Attention: memory-bound without fusion
- GEMMs: compute-bound (Tensor Cores)
- LayerNorm: memory latency-bound

### Evidence
- Large intermediate tensors
- Frequent global memory access
- Partial kernel fusion in optimized libraries

### Optimization Levers
- FlashAttention
- Fused LayerNorm
- Tensor Core utilization

---

## 2. LLaMA – Prefill Phase

### Dominant Kernels
- Large GEMMs
- Attention over long sequence length

### Bottleneck Classification
**Memory-Bound**

### Primary Bottlenecks
- Attention score memory traffic
- Limited cache reuse for long sequences

### Evidence
- O(N²) attention cost
- DRAM bandwidth saturation reported in literature

### Optimization Levers
- FlashAttention
- Sequence tiling
- KV cache blocking

---

## 3. LLaMA – Decode Phase

### Dominant Kernels
- Small GEMMs
- KV cache reads

### Bottleneck Classification
**Latency-Bound / Memory-Bound**

### Primary Bottlenecks
- Poor GPU occupancy
- Memory latency dominates due to small batch size

### Evidence
- Low arithmetic intensity
- Underutilized SMs

### Optimization Levers
- Persistent kernels
- Batching
- CPU offload for small workloads

---

## 4. ResNet-50 Inference

### Dominant Kernels
- Convolutions
- BatchNorm
- ReLU

### Bottleneck Classification
**Compute-Bound**

### Primary Bottlenecks
- Tensor Core throughput
- Instruction throughput

### Evidence
- High arithmetic intensity
- Well-optimized convolution kernels

### Optimization Levers
- Tensor Core utilization
- Kernel fusion

---

## 5. Stable Diffusion – UNet

### Dominant Kernels
- Convolutions
- Attention
- Elementwise ops

### Bottleneck Classification
**Mixed**

### Primary Bottlenecks
- Attention layers (memory-bound)
- Convolutions (compute-bound)
- Elementwise ops (memory latency)

### Evidence
- Heterogeneous kernel mix
- Variable tensor sizes

### Optimization Levers
- Operator fusion
- FlashAttention
- Persistent execution

---

## 6. Large GEMM (Square Matrix)

### Dominant Kernels
- Matrix Multiply

### Bottleneck Classification
**Compute-Bound**

### Primary Bottlenecks
- Tensor Core utilization
- Register pressure

### Evidence
- Near-peak TFLOPS achievable
- High arithmetic intensity

### Optimization Levers
- Register blocking
- Double buffering
- Instruction pipelining

---

## 7. Elementwise Operations

### Dominant Kernels
- Add, Multiply, Activation

### Bottleneck Classification
**Memory-Bound**

### Primary Bottlenecks
- DRAM bandwidth
- Kernel launch overhead

### Evidence
- Very low arithmetic intensity
- High memory traffic per FLOP

### Optimization Levers
- Kernel fusion
- Vectorized loads
- CPU execution for small tensors

---

## 8. Embedding Lookup

### Dominant Kernels
- Sparse memory access

### Bottleneck Classification
**Memory-Latency-Bound**

### Primary Bottlenecks
- Random memory access
- Poor cache locality

### Evidence
- Low L2 hit rates
- High memory stall cycles

### Optimization Levers
- Cache-aware batching
- CPU offload
- Prefetching

---

## Cross-Workload Observations

| Workload Type | Common Bottleneck |
|----|----|
Attention | Memory bandwidth |
Small-batch inference | Latency |
Dense linear algebra | Compute |
Elementwise ops | Memory |

---

## Summary

This analysis highlights that:
- Most modern ML workloads are **not purely compute-bound**
- Memory behavior dominates performance in attention and decoding
- Kernel fusion and tiling are decisive optimization strategies

These findings directly inform:
- GPU kernel design
- Inference system architecture
- CPU–GPU workload partitioning
