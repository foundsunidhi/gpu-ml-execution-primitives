# GEMM Execution Trace and Bottleneck Analysis

## Objective

This document provides a step-by-step execution trace of a General
Matrix Multiply (GEMM) operation on modern GPUs. It focuses on resource
utilization, memory movement, and bottleneck identification across
NVIDIA Volta, Ampere, and Hopper architectures.

The analysis serves as a foundation for understanding attention kernels,
FlashAttention, and CPU inference implications.

---

## 1. GEMM Definition

We consider the canonical GEMM operation:

C = A × B + C

Where:
- A ∈ ℝ(M×K)
- B ∈ ℝ(K×N)
- C ∈ ℝ(M×N)

On GPUs, this operation is decomposed into many small matrix tiles,
each computed by a warp using matrix units.

---

## 2. High-Level GEMM Execution Pipeline

A GPU GEMM proceeds through the following **8 stages**:

1. Host → Device transfer (if required)
2. Kernel launch and grid scheduling
3. Global memory tile loads
4. Shared memory staging
5. Register fragment loading
6. Tensor Core MMA execution
7. Accumulation and epilogue
8. Device → Host transfer (if required)

Each stage stresses different hardware resources.

---

## 3. Step-by-Step Execution Trace

### Step 1: Host → Device Transfer

- Performed via PCIe or NVLink
- Typically amortized over many GEMMs
- Not part of kernel execution time in steady-state training/inference

**Bottleneck:** Interconnect bandwidth (outside SMs)

---

### Step 2: Kernel Launch and Scheduling

- Grid of thread blocks launched
- Each block mapped to an SM
- Multiple blocks may reside on one SM (occupancy)

**Key resources:**
- SM availability
- Register file
- Shared memory capacity

**Profiler signals:**
- Low occupancy → register/shared memory pressure

---

### Step 3: Global Memory Tile Loads

- Tiles of A and B loaded from HBM
- Vectorized and coalesced loads
- Often performed by all threads in a block

**Architectural notes:**
- Volta: synchronous loads
- Ampere: `cp.async`
- Hopper: Tensor Memory Accelerator (TMA)

**Bottleneck:** HBM bandwidth and L2 cache misses

---

### Step 4: Shared Memory Staging

- Loaded tiles stored in shared memory
- Layout transformed for tensor core consumption
- Bank conflicts must be avoided

**Shared memory acts as:**
- Reuse buffer
- Latency hiding mechanism

**Profiler signals:**
- Shared memory bank conflict metrics
- Stall reasons: `smem_dependency`

---

### Step 5: Register Fragment Loading

- Each warp loads its matrix fragment from shared memory
- Fragments mapped to registers
- High register pressure introduced

**Tradeoff:**
- Larger tiles → more registers → lower occupancy

---

### Step 6: Tensor Core MMA Execution

- MMA instructions issued by warps
- Registers feed tensor cores directly
- Compute is highly pipelined

**Architecture differences:**

| Architecture | MMA Feature |
|------------|------------|
| Volta | FP16 MMA only |
| Ampere | TF32, INT8, sparsity |
| Hopper | FP8, Transformer Engine |

**Profiler signals:**
- Tensor Core utilization
- Warp issue efficiency

---

### Step 7: Accumulation and Epilogue

- Partial sums accumulated in registers
- Optional operations:
  - Bias add
  - Activation
  - Scaling

**Epilogue often becomes:**
- Memory-bound
- Scalar instruction heavy

---

### Step 8: Device → Host Transfer

- Typically excluded from kernel timing
- Relevant only for small, isolated GEMMs

---

## 4. Tiling and Blocking Strategy

A typical GEMM uses **three levels of tiling**:

| Level | Tile Scope |
|----|----|
| Grid | Output matrix tiles |
| Thread Block | Cooperative A/B tiles |
| Warp | Tensor Core tile |

This hierarchy:
- Maximizes data reuse
- Reduces HBM traffic
- Matches hardware structure

---

## 5. Bottleneck Classification

### Compute-Bound GEMM

Indicators:
- High Tensor Core utilization
- Low memory stall metrics
- Achieved TFLOPS near theoretical peak

Occurs when:
- Large matrices
- High arithmetic intensity
- FP16/FP8 precision

---

### Memory-Bound GEMM

Indicators:
- Low compute utilization
- High memory stall cycles
- Achieved bandwidth near peak

Occurs when:
- Small batch sizes
- Thin matrices
- Poor tile reuse

---

### Mixed Bottleneck

Common in:
- Transformer prefill
- Attention score computation

Both memory and compute compete.

---

## 6. Architecture Evolution Impact

### Volta

- Tensor Cores underutilized without careful tuning
- Memory latency harder to hide
- Lower arithmetic intensity ceiling

### Ampere

- `cp.async` enables deep pipelining
- TF32 boosts FP32 GEMMs
- Larger L2 reduces HBM pressure

### Hopper

- FP8 increases arithmetic intensity
- TMA reduces instruction overhead
- Thread block clusters enable cross-SM cooperation

---

## 7. Profiling Methodology

Recommended tools:
- NVIDIA Nsight Compute
- NVIDIA Nsight Systems

Key metrics:
- Tensor Core utilization
- Achieved FLOPS vs peak
- Global memory throughput
- L1/L2 cache hit rates
- Stall reasons

Example kernels:
- cuBLAS GEMM
- CUTLASS reference GEMM

---

## 8. Relevance to Attention and FlashAttention

Attention kernels:
- Reuse tiles across Q, K, V
- Are sensitive to memory bandwidth
- Benefit from shared memory reuse

FlashAttention:
- Eliminates intermediate memory writes
- Keeps computation on-chip
- Maximizes tensor core utilization

GEMM analysis explains why FlashAttention scales better than naïve attention.

---

## 9. CPU Implications (Preview)

Key lessons transferable to CPU inference:
- Blocking and tiling dominate performance
- Register pressure vs occupancy tradeoff exists on CPUs too
- Memory movement often dominates arithmetic

These implications are expanded in the CPU synthesis section.

---

## Sources

- NVIDIA CUDA Programming Guide
- NVIDIA Volta, Ampere, Hopper Architecture Whitepapers
- CUTLASS Documentation
- Nsight Compute Profiling Guide
