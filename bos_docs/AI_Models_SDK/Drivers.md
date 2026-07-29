---
title: Untitled
source: https://developers.bos-semi.com/docs/AI_Models_SDK/Drivers/
---

# Drivers

The BOS AI SDK requires kernel-mode and user-mode drivers to communicate with Eagle-N hardware. This section documents both the kernel-mode driver (KMD) and user-mode driver (UMD) components, along with installation and configuration instructions.

## BOS-UMD[](#bos-umd)

This page introduces the BOS User-Mode Driver (UMD) and how to build it for supported Tenstorrent architectures. It provides core setup requirements, including Ubuntu dependencies and recommended compiler tooling. The guide explains system prerequisites such as IOMMU and hugepage behavior, with architecture-specific notes. It also outlines standard build and test flows, including optional GCC-based builds. Finally, it summarizes integration patterns and Eagle-N A0 eval. board specific implementation structure for downstream use.

## BOS AI User-Mode Driver[](#bos-ai-user-mode-driver)

### Repository[](#repository)

[https://github.com/bos-semi-release/BOS_UMD](https://github.com/bos-semi-release/BOS_UMD)

### Software Dependencies[](#software-dependencies)

BOS's UMD requires BOS's [kernel-mode driver](http://192.128.10.101:8093/tt_metal_demo/bos_kmd)

Required Ubuntu dependencies:

```
sudo apt install -y libhwloc-dev cmake ninja-build zlib1g-dev ninja-build libyaml-cpp-dev libfmt-dev libboost-all-dev
```
Suggested third-party dependency is Clang 17:

```
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
```
### IOMMU and Hugepage requirements[](#iommu-and-hugepage-requirements)

To determine whether your system requires hugepage configuration, run the provided script:

```
./scripts/iommu_detect.sh
```
##### Blackhole[](#blackhole)

If your system IOMMU is enabled, no hugepage setup is required. You can enable hugepage with the following command.

- `sudo -E env PATH=$PATH python3 ./scripts/setup_hugepages.py enable`

After the command completes, reboot your PC and run the command again.

### Build flow[](#build-flow)

To build `libdevice.so`:

```
cmake -B build -G Ninja
cmake --build build
```
Tests are built separately for each architecture.
Specify the `ARCH_NAME` environment variable as `blackhole` before building.
You also need to configure cmake to enable tests, hence the need to run cmake configuration step again.
To build tests:

```
cmake -B build -G Ninja -DTT_UMD_BUILD_TESTS=ON
cmake --build build
```
To build with GCC, set these environment variables before invoking `cmake`:

```
export CMAKE_C_COMPILER=/usr/bin/gcc
export CMAKE_CXX_COMPILER=/usr/bin/g++
```
### Add Unit Tests[](#add-unit-tests)

in tests/blackholeplus/CMakeLists.txt :

```
# Example: Add new unit tests here
# Uncomment and modify these lines to add new test files:
# add_unit_test(unit_test_example "unit_test_example.cpp")
# add_unit_test(unit_test_feature "unit_test_feature.cpp" "helper_file.cpp")
# add_unit_test(unit_test_integration "unit_test_integration.cpp")
```
### Integration[](#integration)

UMD can be consumed by downstream projects in multiple ways.

### From Source (CMake)[](#from-source-cmake)

You can link `libdevice.so` by linking against the `umd::device` target.

#### As a submodule/external project[](#as-a-submoduleexternal-project)

```
add_subdirectory(<path to umd>)
```
## BOS-KMD[](#bos-kmd)

The BOS Kernel-Mode Driver (KMD) provides low-level hardware access and device management for Eagle-N NPU. This section covers KMD installation, configuration, device file registration, and building with DKMS for streamlined driver management.

### Repository[](#repository-1)

[https://github.com/bos-semi-release/BOS_KMD](https://github.com/bos-semi-release/BOS_KMD)

### Supported hardware:[](#supported-hardware)

- BOS NN Accelerator

The driver registers device files named `/dev/bos/%d`, one for each enumerated device.

### How to Build and Install the BOS KMD Module Using DKMS[](#how-to-build-and-install-the-bos-kmd-module-using-dkms)

- You must have dkms installed

```
sudo apt-get install dkms
```
- You have to set the version of the KMD module. For now, it is set to `2.3.0`.

```
export VERSION=2.3.0
```
- Build & Install BOS KMD Module with DKMS

```
sudo dkms add .
sudo dkms build -m bos -v 2.3.0
sudo dkms install -m bos -v 2.3.0
sudo modprobe bos
```
- Remove the module from DKMS

```
sudo dkms remove bos_kmd/2.3.0 --all
```
- Insert Module

```
sudo insmod /path/dkms/bos.ko
```
- Remove Module

```
sudo rmmod bos
```
- Verify if the module is successfully inserted

```
lspci -v 
```
- You can use directly the .sh file (You have to create folder /usr/src/bos_kmd-2.3.0)

```
./set_dkms.sh
```
- If you want to change the version of KMD, edit the following files

```
dkms.conf
module.c
set_dkms.h
```

## 이미지

![BOS_Logo_4E_RGB.avif](../_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
