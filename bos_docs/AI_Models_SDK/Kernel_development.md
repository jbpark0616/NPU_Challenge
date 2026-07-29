---
title: Untitled
source: https://developers.bos-semi.com/docs/AI_Models_SDK/Kernel_development/
---

# Kernel Development

Here are examples of Kernel development

## Basic Usage[](#basic-usage)

### Step 1: DRAM Loopback[](#step-1-dram-loopback)

Learn the basic structure of a Metalium application by implementing a [DRAM loopback](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/dram_loopback.html#dram-loopback-example)  that copies data from one DRAM buffer to another.

**What you’ll learn:**

- Basic host and kernel structure
- Buffer management
- Data transfer

### Step 2: Eltwise Binary Kernel[](#step-2-eltwise-binary-kernel)

Build on the loopback example by implementing an [Elwise binary kernel](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/eltwise_binary.html#eltwise-binary-example) that performs addition on two buffers. This introduces computation using the matrix engine (FPU) and data passing within a Tensix core.

**What you’ll learn:**

- Circular buffers for data passing
- Compute kernels
- Using the matrix engine (FPU)

### Step 3: Eltwise SFPU[](#step-3-eltwise-sfpu)

Extend the previous example to implement an [Eltwise SFPU](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/eltwise_sfpu.html#eltwise-sfpu-example) kernel that performs element-wise addition using the SFPU (vector engine, Special Function Processing Unit). This will introduce you to the SFPU and how to use it for vectorized operations

**What you’ll learn:**

- Vectorized operations using the SFPU

## Intermediate Usage[](#intermediate-usage)

### Step 4: Single-core Matrix Multiplication[](#step-4-single-core-matrix-multiplication)

Implement a [matrix multiplication](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html#matmul-single-core-example)  on a single Tensix core using the matrix engine.

**What you’ll learn:**

- Complex dataflow
- Tiled operations
- Matrix engine utilization

## Advanced Usage[](#advanced-usage)

### Step 5: Multi-core Matrix Multiplication[](#step-5-multi-core-matrix-multiplication)

Extend to a [multi-core implementation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core.html#matmul-multi-core-example) by distributing computation across multiple Tensix cores.

**What you’ll learn:**

- Parallel processing
- Workload distribution across cores

### Step 6: Optimized Multi-core Matrix Multiplication[](#step-6-optimized-multi-core-matrix-multiplication)

[Optimize the multi-core implementation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core_optimized.html#matmul-multi-core-example) by leveraging the processor grid, reducing redundant DRAM access, and minimizing NoC congestion through data sharing.

**What you’ll learn:**

- Performance optimization techniques
- Efficient data movement and reuse

Reference: [https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html)

## 이미지

![BOS_Logo_4E_RGB.avif](../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
