# Eagle-N A0 Evaluation Board Documentation

This repository contains comprehensive documentation and resources for the Eagle-N A0 evaluation board.  
The documentation is organized into four main sections that guide you through understanding, setting up, and working with the Eagle-N A0 NPU evaluation board.

## Documentation Overview

### 1. [Evaluation Board Overview](./Evaluation_Board_Overview.md)
**Board specifications/configurations and boot sequence**

This document provides a comprehensive overview of the Eagle-N A0 evaluation board, including:
- **Physical Specifications**: Board dimensions, mechanical frame specifications, and power requirements
- **Connectivity Information**: Detailed connector descriptions and locations
- **Switch Settings**: Default configurations for CPU and MSHC switches
- **Operation Modes**: Boot source selection and OPMODE pin settings
- **Memory Map**: Eagle-N A0 memory layout and addressing
- **NPU Block Diagram**: Neural Processing Unit architecture and interfaces
- **Software Overview**: Component distribution across RTCore, Cortex-M7, Cortex-A53, and host AP
- **Boot Sequence**: Complete N1 booting process from power-on to Linux kernel

### 2. [How to Set up Eagle-N Eval. Board](./How_to_Setup_Board.md) 
**Physical setup and environment configuration**

This guide covers the complete setup process for the Eagle-N A0 evaluation board:
- **Physical Connection**: Using PCIe riser cable to connect board to host PC
- **System Requirements**: Host PC and Eagle-N A0 specifications
- **BIOS Configuration**: Critical BIOS settings for PCIe wake-up and BAR size support
- **Boot Process**: Step-by-step instructions for powering on and accessing the board
- **Serial Console Setup**: USB connections and terminal access configuration
- **PCIe Link Verification**: Commands to verify successful host-board connection
- **Login Credentials**: Default username and password for Linux access

### 3. [Tracy Performance Profiling](./Tracy_intro.md)
**Tracing workflow for tt-metal workloads**

This guide explains how to profile device execution with the Tracy tooling:
- **Profiler Overview**: Understand Tensix core roles (BRISC, NRISC, TRISC) and how Tracy visualizes RISC activity
- **Code Instrumentation**: Enable `DEBUG_TRACY_LOG`, include `kernel_profiler.hpp`, and wrap kernels with `DeviceZoneScopedN`
- **Build & Run Flags**: Build `tt-metal` with `--enable-profiler` and run workloads using `ENABLE_TRACY=1 TT_METAL_DEVICE_PROFILER=1`
- **Trace Collection**: Locate generated `.tracy` logs under `tt-metal/generated/profiler/.logs` and transfer them to your host
- **Trace Analysis**: Inspect matmul and softmax timelines inside the Tracy UI to diagnose stalls and pipeline balance; follow linked docs for deeper guidance

### 4. [How to Install SW Stack](./How_to_Install_SW_Stack.md)
**Software installation and environment setup**

Comprehensive software stack installation guide including:
- **Hugepage Configuration**: Linux kernel memory management setup for optimal performance
- **PCIe Driver Installation**: Kernel Mode Driver (KMD) setup using DKMS framework
- **tt-metal Framework**: Complete installation of the TensorTalk Metal framework
  - System-level dependencies and CMake 3.20 installation
  - OpenCV installation for video demos
  - Third-party submodule dependencies
  - Build process and Python virtual environment setup
- **GitHub SSH Configuration**: Required for accessing BOS Semi repositories
- **Security Settings**: UEFI Secure Boot disable requirements

### 5. [How to Fuse New Images on Eagle-N Eval. Board](./How_to_Fuse_New_Image.md)
**Firmware update and image flashing procedures**

Complete guide for updating booting images on the evaluation board:
- **Connection Setup**: USB Type-B and Type-C cable connections for image download
- **Fastboot Mode**: Entering USB fastboot mode during u-boot phase
- **Host PC Configuration**: Android platform tools installation and device recognition
- **Permission Setup**: udev rules configuration for USB device access
- **Image Flashing**: Complete process for updating firmware partitions
- **Storage Selection**: eMMC vs SD card partition switching procedures
- **Partition Management**: Understanding different image types and their purposes

## Quick Start

1. **Hardware Setup**: Follow the [board setup guide](./How_to_Setup_Board.md) to physically connect your Eagle-N A0 board
2. **Software Installation**: Install the complete software stack using the [SW installation guide](./How_to_Install_SW_Stack.md)
3. **Firmware Updates**: Use the [image flashing guide](./How_to_Fuse_New_Image.md) when you need to update board firmware
