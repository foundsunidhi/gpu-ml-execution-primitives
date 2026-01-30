# Latency Hiding in Modern GPUs

## 1. Introduction

Memory latency is one of the primary performance bottlenecks in GPU workloads. 
Global memory accesses can take hundreds of cycles to complete, far exceeding the execution latency of arithmetic instructions. 
Rather than reducing memory latency directly, GPUs are designed to *hide* this latency using massive hardware multithreading.

This document analyzes the mechanisms by which NVIDIA GPUs hide latency, the architectural limits of these mechanisms, and the conditions under which latency hiding breaks down.

---

## 2. Sources of Latency in GPU Execution

### 2.1 Memory Latency Breakdown

Typical latency ranges on modern NVIDIA GPUs:

| Memory Level | Approximate Latency (cycles) |
|-------------|-----------------------------|
| Register    | 1–2                         |
| Shared Mem  | 20–30                       |
| L1 Cache    | 30–40                       |
| L2 Cache    | 200–300                     |
| HBM (DRAM)  | 400–800                     |

These values are aggregated from NVIDIA architecture whitepapers and microbenchmark studies.

**Key observation:** Arithmetic pipelines cannot stall for hundreds of cycles without losing throughput.

---

## 3. Warp-Based Latency Hiding

### 3.1 Warp Scheduling Model

NVIDIA GPUs execute threads in groups of 32 called *warps*.  
Each Streaming Multiprocessor (SM) maintains a pool of *active warps*.

At every cycle:
- The warp scheduler selects one **ready warp**
- Issues one instruction per scheduler
- If a warp stalls (e.g., memory access), it is skipped

Latency hiding is achieved by **switching to another warp** whose data is ready.

---

### 3.2 Occupancy and Its Role

**Occupancy** is defined as:

Occupancy = Active Warps per SM / Maximum Warps per SM


Higher occupancy increases the probability that:
- At least one warp is ready to execute
- Memory latency can be overlapped with computation

However, occupancy alone does not guarantee latency hiding.

---

## 4. Quantitative Model of Latency Hiding

To fully hide memory latency:
Number of Independent Warps ≥ Memory Latency / Instruction Issue Interval


Example:
- HBM latency ≈ 600 cycles
- One instruction issued per cycle

Required:
≥ 600 independent warp instructions


In practice:
- Dependencies
- Execution divergence
- Instruction mix

reduce effective overlap.

---

## 5. When Latency Hiding Fails

Latency hiding breaks down under the following conditions:

### 5.1 Low Occupancy
- Excessive register usage
- Large shared memory allocations
- Small grid sizes

### 5.2 Memory-Level Parallelism Limits
- Too many warps waiting on the same cache line
- Serialized memory transactions

### 5.3 Instruction-Level Dependencies
- Pointer chasing
- Dependent memory loads
- Sequential address computation

### 5.4 Bandwidth Saturation
Even with sufficient warps:
- DRAM bandwidth may become the bottleneck
- Latency is hidden, but throughput is capped

---

## 6. Architectural Evolution

### 6.1 Volta
- Independent thread scheduling
- Improved warp issue flexibility
- Reduced stall penalties

### 6.2 Ampere
- Asynchronous memory operations (`cp.async`)
- Better overlap of memory and computation
- Increased L2 cache size

### 6.3 Hopper
- Thread Block Clusters
- Distributed shared memory
- Increased memory-level parallelism across SMs

Each generation increases the *scope* at which latency can be hidden.

---

## 7. Interaction with Memory Access Patterns

Latency hiding is effective only when:
- Accesses are coalesced
- Cache reuse exists
- Memory-level parallelism is high

Random or strided access patterns reduce the number of warps that can make forward progress simultaneously.

This directly connects to benchmarks defined in `memory/benchmarks.csv`.

---

## 8. Practical Implications for Kernel Design

To maximize latency hiding:
- Increase independent work per thread
- Reduce register pressure
- Avoid dependent memory loads
- Use shared memory for reuse
- Use asynchronous copies where available

Latency hiding is not automatic—it must be *enabled* by kernel structure.

---

## 9. Summary

GPUs tolerate high memory latency not by speeding up memory, but by:
- Maintaining thousands of threads
- Rapidly switching between warps
- Exploiting memory-level parallelism

Latency hiding is bounded by:
- Occupancy
- Dependency structure
- Memory system limits

Understanding these limits is essential for both performance optimization and architecture research.

---

## 10. References

- NVIDIA CUDA Programming Guide  
- NVIDIA Volta, Ampere, and Hopper Architecture Whitepapers  
- Nsight Compute Profiling Guide  
- GPU Performance Modeling Literature
