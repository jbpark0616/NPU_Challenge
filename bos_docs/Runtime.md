---
title: Untitled
source: https://developers.bos-semi.com/docs/Runtime/
---

# Runtime

Discover the Tenstorrent based runtime operations and APIs compatible with the BOS Neural Network accelerators.

## TT-NN[](#tt-nn)

TT-NN is an open-source C++ and Python library of neural network operations, built on top of the TT-Metalium programming model. It provides a PyTorch-like interface for running machine learning workloads on Tenstorrent AI accelerators, and serves as the primary high-level API for developing and optimizing ML models on Tenstorrent hardware.

### TT-NN APIs[](#tt-nn-apis)

Reference guide for all TT-NN neural network operations and APIs. Comprehensive documentation covering available operations, their parameters, execution semantics, and performance characteristics for building efficient AI models on BOS AI accelerators.

[https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html)

### Purpose[](#purpose)

#### Enable ML frameworks targeting Tenstorrent hardware[](#enable-ml-frameworks-targeting-tenstorrent-hardware)

Developers have access to a vast library of existing models implemented in PyTorch, JAX, TensorFlow, and other frameworks. TT-NN offers a collection of operations and reusable building blocks to support the development of ML compilers and framework backends targeting Tenstorrent hardware.

→ See framework integrations: [Forge Compiler](https://github.com/tenstorrent/tt-forge) [PyTorch 2.0 TT-NN Backend](https://github.com/tenstorrent/pytorch2.0_ttnn)

#### Manual bringup and optimization of ML models[](#manual-bringup-and-optimization-of-ml-models)

When performance is critical, developers need fine-grained control. Existing ML frameworks and compilers don’t fully expose the capabilities of our hardware. TT-NN lets developers work with familiar high-level operations while tapping into hardware-specific optimizations when necessary. This includes options to specify data format for mixed-precision, tensor layout and distribution, op fusion and other operation specific settings.

→ See production-ready model examples: [Model Zoo](https://tenstorrent.com/developers) 

→ See some model bringup guides: [General](https://github.com/tenstorrent/tt-metal/blob/0c0528ef1b395127c535edc0152a3ba2558fcfa4/tech_reports/ttnn/TTNN-model-bringup.md) | [LLMs](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) | [CNNs](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md)

### Key Features[](#key-features)

- **High-Level Neural Network Operations:**
 Optimized implementations of key neural net components: matrix multiplication, convolution, attention mechanisms, data movement, collective communications (CCLs), element-wise ops, reductions, losses, pooling, and more. APIs are PyTorch-style but expose hardware-specific options.
- **Tensor Library:**
 A flexible tensor abstraction for managing multidimensional arrays across host and device. Developers can precisely control data layout across a cluster of Tenstorrent accelerators via Tensor APIs.
- **Native Multi-Device Support:**
 TT-NN seamlessly virtualizes multiple Tenstorrent devices into a single logical unit, enabling effortless scaling across device clusters.

### Getting Started[](#getting-started)

Once the hardware and drivers are installed, you can install TT-NN directly from PyPI.

```
pip install ttnn
```
To check that installation was successful, run a simple script

```
import ttnn
device = ttnn.open_device(device_id=0)
a = ttnn.full([5, 5, 5], fill_value=1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.full([1], fill_value=2.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
c = a * b
print(c)
```
There is a collection of tutorials written with [Jupyter Notebooks](https://jupyter.org/install) to help you ramp up your skillset for using TT-NN.
These notebooks can be found under [ttnn/tutorials](https://github.com/tenstorrent/tt-metal/tree/main/ttnn/tutorials).

From within the `ttnn/tutorials` directory, launch the notebooks with:

```
jupyter lab --no-browser --port=8888
```
### Tools and Instruments[](#tools-and-instruments)

[TT-Triage](https://github.com/tenstorrent/tt-metal/blob/main/tools/triage/tt-triage.md)[](#tt-triage)

The TT-Triage tool diagnoses failures by performing comprehensive health checks on ARC processors, NOC connectivity, L1 memory, and RISC-V cores. It identifies running kernels and provides callstack data for troubleshooting.

[TT-NN Visualizer](https://github.com/tenstorrent/ttnn-visualizer)[](#tt-nn-visualizer)

A comprehensive tool for visualizing and analyzing model execution, offering interactive graphs, memory plots, tensor details, buffer overviews, operation flow graphs, and multi-instance support with file or SSH-based report loading.

[TT-Exalens](https://github.com/tenstorrent/tt-exalens)[](#tt-exalens)

The TT-Exalens repository describes TT-Lensium, a low-level debugging tool for Tenstorrent hardware. It allows developers to access and communicate with Wormhole and Blackhole devices.

[TT-SMI](https://github.com/tenstorrent/tt-smi)[](#tt-smi)

The TT-SMI repository describes the Tenstorrent System Management Interface. This command line utility can interact with Tenstorrent devices on host side. TT-SMI provides an easy-to-use interface displaying device, telemetry, and firmware information.

[Model Explorer](https://github.com/tenstorrent/model-explorer)[](#model-explorer)

The Model Explorer is an intuitive and hierarchical visualization tool using model graphs. It organizes model operations into nested layers and provides features for model exploration and debugging.

[Tracy Profiler](https://github.com/tenstorrent/tracy)[](#tracy-profiler)

The Tracy Profiler is a real-time nanosecond resolution, remote telemetry, hybrid frame, and sampling tool. Tracy supports profiling CPU, GPU, memory allocation, locks, context switches, and more.

[Kernel Print Debug](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_print.html)[](#kernel-print-debug)

DPRINT can print variables, addresses, and circular buffer data from kernels to the host terminal or log file. This feature is useful for debugging issues with kernels.

[Watcher](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/watcher.rst)[](#watcher)

Watcher monitors firmware and kernels for common programming errors, and overall device status. If an error or hang occurs, Watcher displays log detailing theoccurrence.

[Inspector](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/inspector.rst)[](#inspector)

Inspector provides insights into host runtime. It logs necessary data for investigation and allows queries to host runtime data.

## TT-Metalium[](#tt-metalium)

TT-metalium is the core Python and C++ neural network operator library built on MLIR. This section covers installation, featured model implementations, performance benchmarks, and links to the official repository and API documentation.

### Runtime Host APIs[](#runtime-host-apis)

Please refer to the official TT-metalium Host APIs reference.

Reference:

Key topics covered there:

- API entry points for host-side initialization and shutdown
- Device discovery, selection, and configuration
- Memory allocation, buffer management, and data movement
- Command submission, synchronization, and runtime control

For full details and examples, use the official reference link above.

### Runtime Kernel APIs[](#runtime-kernel-apis)

Detailed reference documentation for TT-metalium kernel APIs that enable low-level programming on Tensix cores. Includes API specifications, kernel development patterns, and execution models for creating custom operations on BOS AI accelerators hardware.

### Getting started[](#getting-started-1)

Get started with [simple kernels](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html).

### TT-Metalium Tech Reports[](#tt-metalium-tech-reports)

- [Matrix Engine](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md)(updated Sept 6th, 2024)
- [Data Formats](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/data_formats/data_formats.md)(updated Sept 7th, 2024)
- [Reconfiguring Data Formats](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/data_formats/reconfig_data_format.md)(updated Oct 17th, 2024)
- [Handling special floating-point numbers](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Handling_Special_Value/special_values.md)(updated Oct 5th, 2024)
- [Allocator](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/memory/allocator.md)(Updated Dec 19th, 2024)
- [Tensor Layouts](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md)(updated Sept 6th, 2024)
- [Saturating DRAM Bandwidth](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md)(updated Sept 6th, 2024)
- [Flash Attention on Wormhole](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashAttention.md)(updated Sept 6th, 2024)
- [CNNs on TT Architectures](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/ttcnn.md)(updated Sept 6th, 2024)
- [Ethernet and Multichip Basics](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/EthernetMultichip/BasicEthernetGuide.md)(Updated Sept 20th, 2024)
- [Blackhole Bring-Up Programming Guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md)(Updated Dec 18th, 2024)
- [Sub-Devices](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/SubDevices/SubDevices.md)(Updated Jan 7th, 2025)

### Scaleout Tech Reports[](#scaleout-tech-reports)

- [Programming Mesh of Devices (Scale-Up)](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md)(updated Jan 6th, 2026)
- [Programming Multiple Meshes (Scale-Out)](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md)(updated Jan 19th, 2026)
- [TT-Fabric Architecture](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/TT-Fabric/TT-Fabric-Architecture.md)(updated Dec 1st, 2025)
- [TT-Distributed Architecture](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/TT-Distributed/TT-Distributed-Architecture-1219.md)(updated Oct 20th, 2025)

### TT-Metalium Programming Examples[](#tt-metalium-programming-examples)

#### Hello World[](#hello-world)

#### Add Integers[](#add-integers)

#### Simple Tensor Manipulation[](#simple-tensor-manipulation)

#### DRAM Data Movement[](#dram-data-movement)

#### Eltwise[](#eltwise)

#### Matmul[](#matmul)

- [Matmul OP on a Single_core](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.md)
- [Matmul OP on Multi_core (Basic)](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_multi_core/matmul_multi_core.md)
- [Matmul Multi_core Reuse (Optimized)](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/prog_examples/matmul_multi_core_optimized/data_reuse.md)
- [Matmul Multi_core Multi-Cast (Optimized)](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/prog_examples/matmul_multi_core_optimized/data_mcast.md)

#### Tools and Instruments[](#tools-and-instruments-1)

[TT-NN Visualizer](https://github.com/tenstorrent/ttnn-visualizer)[](#tt-nn-visualizer-1)

A comprehensive tool for visualizing and analyzing model execution, offering interactive graphs, memory plots, tensor details, buffer overviews, operation flow graphs, and multi-instance support with file or SSH-based report loading.

[TT-Exalens](https://github.com/tenstorrent/tt-exalens)[](#tt-exalens-1)

The TT-Exalens repository describes TT-Lensium, a low-level debugging tool for Tenstorrent hardware. It allows developers to access and communicate with Wormhole and Blackhole devices.

[TT-SMI](https://github.com/tenstorrent/tt-smi)[](#tt-smi-1)

The TT-SMI repository describes the Tenstorrent System Management Interface. This command line utility can interact with Tenstorrent devices on host. TT-SMI provides an easy to use interface displaying device, telemetry, and firmware information.

[Model Explorer](https://github.com/tenstorrent/model-explorer)[](#model-explorer-1)

The Model Explorer is an intuitive and hierarchical visualization tool using model graphs. It organizes model operations into nested layers and provides features for model exploration and debugging.

[Tracy Profiler](https://github.com/tenstorrent/tracy)[](#tracy-profiler-1)

The Tracy Profiler is a real-time nanosecond resolution, remote telemetry, hybrid frame, and sampling tool. Tracy supports profiling CPU, GPU, memory allocation, locks, context switches, and more.

[Kernel Print Debug](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_print.html)[](#kernel-print-debug-1)

DPRINT can print variables, addresses, and circular buffer data from kernels to the host terminal or log file. This feature is useful for debugging issues with kernels.

[Watcher](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/watcher.rst)[](#watcher-1)

Watcher monitors firmware and kernels for common programming errors, and overall device status. If an error or hang occurs, Watcher displays log data of the occurrence.

[Inspector](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/inspector.rst)[](#inspector-1)

Inspector provides insights into host runtime. It logs necessary data for investigation and allows queries on host runtime data.

### Latest Releases[](#latest-releases)

| Release | Release Date | FW Version | KMD Version | SMI Version | 
|---|---|---|---|---|
| 0.66.0 | ETA Jan 30, 2026 | 19.2.0 | 2.5.0 | 3.0.38 | 
| [0.65.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.65.0) | Dec 15, 2025 | 19.2.0 | 2.5.0 | 3.0.38 | 
| [0.64.5](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.5) | Dec 1, 2025 | 18.12.0 | 2.4.1 | 3.0.32 | 
| [0.64.4](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.4) | Nov 24, 2025 | 18.12.0 | 2.4.1 | 3.0.32 | 
| [0.64.3](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.3) | Nov 14, 2025 | 18.12.0 | 2.4.1 | 3.0.32 | 
| [0.64.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.0) | Oct 29, 2025 | 18.12.0 | 2.4.1 | 3.0.32 | 
| [0.63.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.63.0) | Sep 22, 2025 | 18.8.0 | 2.3.0 | 3.0.28 | 
| [0.62.2](https://github.com/tenstorrent/tt-metal/releases/tag/v0.62.2) | Aug 20, 2025 | 18.6.0 | 2.0.0 | 3.0.20 | 
| 0.61.0 | Skipped | - | - | - | 
| [0.60.1](https://github.com/tenstorrent/tt-metal/releases/tag/v0.60.1) | Jul 22, 2025 | 18.6.0 | 2.0.0 | 3.0.20 | 
| [0.59.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.59.0) | Jun 18, 2025 | - | - | - | 
| [0.58.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.58.0) | May 13, 2025 | - | - | - | 
| [0.57.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.57.0) | Apr 15, 2025 | - | - | - | 
| [0.56.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.56.0) | Mar 7, 2025 | - | - | - | 

Visit the [releases](https://github.com/tenstorrent/tt-metal/tree/main/releases) folder for details on releases, release notes, and estimated release dates.

## TT-LLk[](#tt-llk)

TT-llk is Tenstorrent's low-level kernel library for AI chips such as Wormhole and Blackhole. It provides header-only compute primitives that serve as the foundation for higher-level ML software stacks. The repository also includes a test environment for validating LLK APIs and kernel behavior. This page summarizes installation, dependencies, supporting documentation, and contribution guidance.

### Overview[](#overview)

This repository contains header-only low-level kernels (LLK) for Tenstorrent AI chips, including Wormhole, and Blackhole.

These kernels serve as foundational compute primitives, acting as building blocks for higher-level software stacks that implement machine learning (ML) operations.

Additionally, the repository includes a test environment designed to validate LLK APIs.

### Install[](#install)

- 
**Clone the repository**Clone the [llk](https://github.com/bos-semi-release/tt-llk)repository to your local computer.
- 
**Set up the test environment**Follow the instructions in the [testing README](https://github.com/tenstorrent/tt-llk/blob/main/tests/README.md)to set up the test environment.

### Software dependencies[](#software-dependencies)

Test environment requires SFPI compiler for building, which is automatically ingested from [sfpi](https://github.com/tenstorrent/sfpi)

### Documentation[](#documentation)

The following documentation is available to help you understand and use low-level kernels:

- 
[Intro](https://github.com/tenstorrent/tt-llk/blob/main/docs/llk/l1/intro.md)
- 
[Top-level Overview](https://github.com/tenstorrent/tt-llk/blob/main/docs/llk/l2/top_level_overview.md)
- 
[LLK Programming Model](https://github.com/tenstorrent/tt-llk/blob/main/docs/llk/l3/programming_model.md)[tt-metal](https://github.com/tenstorrent/tt-metal), providing practical guidance on how to leverage LLKs for efficient kernel development.

## Speculative Decoding[](#speculative-decoding)

Under development

## Multi Models Execution[](#multi-models-execution)

Tasks scheduler for multi models execution under development

## Virtualization[](#virtualization)

Under development

## 이미지

![BOS_Logo_4E_RGB.avif](_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
