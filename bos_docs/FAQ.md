---
title: Untitled
source: https://developers.bos-semi.com/docs/FAQ/
---

# FAQ

This page contains frequently asked questions about the BOS AI Models SDK, covering topics like supported frameworks, graph construction, memory management, debugging, and hardware architecture. Find answers to common questions about model development, compilation, and deployment on Eagle-N NPU hardware.

## Model Definition[](#model-definition)

**Which AI frameworks do you support (e.g., PyTorch, ONNX, TensorFlow)?**

PyTorch, ONNX, TensorFlow.

## Graph Construction[](#graph-construction)

**Do you support a hardware-agnostic Intermediate Representation (IR)?**

TT-forge compiler has a hardware-agnostic IR layer.

**Can the IR be exported or imported from other formats (e.g., MLIR, TVM)?**

MLIR is supported through TT-mlir.

## Memory Management[](#memory-management)

**How can users configure or control memory layout (e.g., SRAM pinning, DMA)?**

TT-metal software stack supports memory layout configuration, such as tiled/row-major, sharded/interleaved, DRAM/SRAM.

## Low-Level Programming Interface[](#low-level-programming-interface)

**How can developers debug or trace execution at the kernel or instruction level?**

- Tracy shows processing time details through the GUI interface.
- L1 visualizer shows memory occupancy during operation.
- Watcher assists with tracking data at specific addresses.

## Memory Hierarchy (Transformer Execution)[](#memory-hierarchy-transformer-execution)

**Please describe the memory hierarchy (e.g., scratchpad, SRAM, L0–L2, DRAM): size, bandwidth, latency.**

- Register
- 36MB L1
- 96GB device DRAM
- Host DRAM over PCIe

## KV-Cache[](#kv-cache)

**How is the KV-cache implemented and managed? Where is it stored and how is it accessed per token?**

Stored in DRAM.

## Compute Units[](#compute-units)

**For each compute unit type, what are the core specifications — supported data types, native bit widths, operating frequency, and number of instances?**

### 1) Matrix / FPU[](#1-matrix--fpu)

- Compute unit type: INT8 / Low-fidelity — 2k MACs/clk
- Supported data types: INT8, FP4, BF16, FP16, MXFP6, MXFP8, MXINT8, MXINT4, MXINT2

### 2) Vector / SFPU[](#2-vector--sfpu)

- Compute unit type: 32 FP32 MACs per clk (per core)
- Supported data types: FP16, FP32

**Number of instances:** 48 (4 × 3 clusters, 4 cores per cluster)

## ARM Support[](#arm-support)

**Does BOS provide Arm-compatible releases?**

Yes. BOS can provide Arm-compatible releases on demand. Contact BOS support or your BOS representative to discuss your target environment and availability.

## Board suddenly not detected[](#board-suddenly-not-detected)

## Unfold details

**Symptom**

The host terminal may suddenly report that no NPU chip is detected:

```
Info:
No chips detected in the cluster
```
A device check may still show that the PCIe device is visible, but the BOS kernel driver is not loaded:

```
[FAIL] Expected kernel driver 'bos', got: '<none>'
```
**Cause**

This can happen after an Ubuntu automatic Linux kernel update on the host PC.

The BOS kernel module is built against the Linux kernel headers available at installation time. If Ubuntu updates the host kernel, the installed BOS kernel module may no longer match the currently running kernel version. In that case, Linux does not load the BOS kernel driver during boot.

As a result, the PCIe device may still appear in lspci, but the BOS driver is not attached, so TT-Metal cannot detect the NPU chip.

**Solution**

Rebuild and reinstall the BOS kernel driver against the currently running kernel.

From the bos-kmd folder, run:

```
./set_dkms.sh
```
Then reboot the host PC.

After reboot, confirm that the BOS driver is loaded and that the NPU is detected correctly.

**Optional preventive action**

It is possible to prevent Ubuntu from automatically upgrading the kernel packages:

sudo apt-mark hold linux-generic sudo apt-mark hold linux-image-generic sudo apt-mark hold linux-headers-generic

However, this is not generally recommended, because it may also block important kernel security updates.

## 이미지

![BOS_Logo_4E_RGB.avif](_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
