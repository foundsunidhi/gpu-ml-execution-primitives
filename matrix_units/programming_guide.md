# Matrix Unit Programming Guide

## Objective

This document explains how specialized matrix multiplication hardware
(Tensor Cores and equivalents) is programmed on modern GPUs. It focuses
on data layout requirements, execution flow, and architectural evolution
from NVIDIA Volta through Hopper.

The goal is to bridge **hardware specifications** with **real kernel behavior**
for GEMM and attention workloads.

---

## 1. Matrix Units in the GPU Execution Model

Matrix units do not operate independently. They are:

- Issued by warps
- Fed by registers
- Staged through shared memory
- Scheduled alongside scalar/vector instructions

A single warp cooperatively computes one or more matrix tiles using
matrix multiply-accumulate (MMA) instructions.

---

## 2. Programming Interfaces Overview

### 2.1 CUDA WMMA (Volta → Ampere)

WMMA (Warp Matrix Multiply Accumulate) is a high-level API that abstracts
tensor core usage.

Key properties:
- One warp computes one matrix tile
- Tile shape fixed by hardware generation
- Limited control over instruction scheduling

Example conceptual flow:

```cpp
load_matrix_sync(A_frag, A_shared);
load_matrix_sync(B_frag, B_shared);
mma_sync(C_frag, A_frag, B_frag, C_frag);
store_matrix_sync(C_shared, C_frag);
