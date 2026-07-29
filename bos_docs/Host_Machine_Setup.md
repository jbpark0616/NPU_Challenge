---
title: Untitled
source: https://developers.bos-semi.com/docs/Host%20Machine%20Setup/
---

# Host Machine Setup

- **Bare-metal**
- **Docker**

## Bare-metal[](#bare-metal)

Comprehensive guide for setting up a bare-metal Linux host environment to run BOS AI SDK. Covers hugepage configuration, kernel-mode driver installation, system tuning, and verification procedures for direct hardware access on Eagle-N systems.

### Configure Hugepage on Linux[](#configure-hugepage-on-linux)

Open grub configuration and apply hugepage configuration

```
$ sudo nano /etc/default/grub
```
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash hugepagesz=1G hugepages=2 default_hugepagesz=1G"
```
Apply the change and reboot the host PC

```
$ sudo update-grub
Sourcing file `/etc/default/grub'
Generating grub configuration file ...
Found linux image: /boot/vmlinuz-6.14.0-35-generic
Found initrd image: /boot/initrd.img-6.14.0-35-generic
Found linux image: /boot/vmlinuz-6.14.0-34-generic
Found initrd image: /boot/initrd.img-6.14.0-34-generic
Found memtest86+ 64bit EFI image: /boot/memtest86+x64.efi
Warning: os-prober will not be executed to detect other bootable partitions.
Systems on them will not be added to the GRUB boot configuration.
Check GRUB_DISABLE_OS_PROBER documentation entry.
Adding boot menu entry for UEFI Firmware Settings ...
done
$ sudo reboot
```
After reboot, verify that the hugepage configuration is applied and that the hugepage size is `1048576 KB`.

```
$ cat /proc/meminfo | grep Huge
AnonHugePages:         0 kB
ShmemHugePages:    57344 kB
FileHugePages:         0 kB
HugePages_Total:       2
HugePages_Free:        3
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:    1048576 kB
Hugetlb:         2097152 kB
```
Mount hugepage file system

```
$ sudo mkdir -p /dev/hugepages-1G
$ sudo mount -t hugetlbfs -o pagesize=1G,mode=0777 hugetlbfs /dev/hugepages-1G
$ mount | grep hugepages
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,pagesize=1024M)
hugetlbfs on /dev/hugepages-1G type hugetlbfs (rw,relatime,mode=777,pagesize=1024M)
```
Add this "fstab" modification for permission at the end of the file

```
$ vi /etc/fstab 
none    /dev/hugepages-1G   hugetlbfs   pagesize=1G,mode=1777   0 0
```
### Installing KMD driver[](#installing-kmd-driver)

Clone the KMD Driver from GitHub repository ([https://github.com/bos-semi-release/BOS_KMD.git](https://github.com/bos-semi-release/BOS_KMD.git)) and execute installation script.

Since the main repositories and submodules are cloned over SSH, please configure SSH keys on both GitHub and your local machine before cloning:
(See the following guide in GitHub: [Connecting to GitHub with SSH - GitHub Docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) )

```
$ git clone 
$ cd BOS_KMD
$ ./set_dkms.sh 
[sudo]  password: 
Module bos-2.3.0 for kernel 6.14.0-34-generic (x86_64).
Before uninstall, this module version was ACTIVE on this kernel.
bos.ko.zst:
 - Uninstallation
   - Deleting from: /lib/modules/6.14.0-34-generic/updates/dkms/
 - Original module
   - No original module was found for this module on this kernel.
   - Use the dkms install command to reinstall any previous module version.
depmod...
Module bos-2.3.0 for kernel 6.14.0-35-generic (x86_64).
Before uninstall, this module version was ACTIVE on this kernel.
bos.ko.zst:
 - Uninstallation
   - Deleting from: /lib/modules/6.14.0-35-generic/updates/dkms/
 - Original module
   - No original module was found for this module on this kernel.
   - Use the dkms install command to reinstall any previous module version.
depmod...
Deleting module bos-2.3.0 completely from the DKMS tree.
Creating symlink /var/lib/dkms/bos/2.3.0/source -> /usr/src/bos-2.3.0
Sign command: /usr/bin/kmodsign
Signing key: /var/lib/shim-signed/mok/MOK.priv
Public certificate (MOK): /var/lib/shim-signed/mok/MOK.der
Building module:
Cleaning build area...
make -j16 KERNELRELEASE=6.14.0-35-generic all...
Signing module /var/lib/dkms/bos/2.3.0/build/bos.ko
Cleaning build area...
bos.ko.zst:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/6.14.0-35-generic/updates/dkms/
Running the post_install script:
depmod...
```
[[email protected]](/cdn-cgi/l/email-protection):bos-semi-release/BOS_KMD.gitCheck the driver module name is `bos`

```
$ lspci -v
0000:01:00.0 Co-processor: Synopsys, Inc. DWC_usb3 / PCIe bridge
	Flags: bus master, fast devsel, latency 0, IRQ 171, IOMMU group 11
	Memory at 80000000 (32-bit, prefetchable) [size=128M]
	Memory at 78000000 (32-bit, prefetchable) [size=128M]
	Memory at 88000000 (32-bit, prefetchable) [size=2M]
	Capabilities: <access denied>
	Kernel driver in use: bos
	Kernel modules: bos
```
### Installing tt-metal[](#installing-tt-metal)

#### System-level dependencies[](#system-level-dependencies)

**NOTE**: A Docker container that contains the full software stack can be used instead of installing **tt-metal**. Please refer to the 'Docker image and Dockerfile' section of the release note.

Disable ‘UEFI Secure boot’ from the host BIOS setup, if your host PC supports this feature.

Clone `tt-metal` with all submodules from the GitHub URL ([https://github.com/bos-semi-release/tt-metal](https://github.com/bos-semi-release/tt-metal)) using the branch or tag shown below. Branch and tag information can be found in the release notes.

```
$ git clone --branch=${RELEASE_BRANCH_NAME} 
Cloning into 'tt-metal' ...
```
[[email protected]](/cdn-cgi/l/email-protection):bos-semi-release/tt-metal.git --recurse-submodulesInstalling CMake 3.20

Some dependencies like unity require CMake 3.20.

```
$ apt -y update && \
$ apt install -y --no-install-recommends ca-certificates gpg wget && \
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
gpg --dearmor - | \
tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" | \
tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
rm -rf /var/lib/apt/lists/*
```
Installing OpenCV

For the video demo, OpenCV with GTK support is required on the host system

```
$ pip install opencv-python
```
#### Third-party dependencies[](#third-party-dependencies)

All of the third-party submodule dependencies below are cloned during cloning tt-metal:

- bos-metal, SHA: 32cd678a7fbb48703a67f536399529c33bc42ded
- tracy, SHA: a0668deca5495616b0ccd1bd618552f5a4cfb3e7
- tt-llk, SHA: 2c367bc0215369cb0dd5405edb17cd5bdea66f15
- umd, SHA: 972bf3330285225c905ba97d551e17203a470bc1

#### Building tt-metal[](#building-tt-metal)

```
## Configure Environment Variables
$ cd ${PATH_TO_SW_PACKAGE_SOURCE_ROOT}/tt-metal/tt-metal
$ source env_set.sh
## Build TT-Metal (in Release mode for demo)
$ ./build_metal.sh -b Release
```
#### Creating Python Virtual Environment[](#creating-python-virtual-environment)

```
## Create virtual environment
$ ./create_venv.sh
## Activate virtual environment
$ source env_set.sh
```
#### Run AI models on the Eagle-N A0 board[](#run-ai-models-on-the-eagle-n-a0-board)

To run an AI model, refer to the `README.md` file in each model directory specified in the release note.

## Docker-based[](#docker-based)

Docker image and Dockerfile could be used instead of installing the software stack on the host PC. Docker image contains all the runtime environment for AI workloads. You can also build your own Docker image using the provided Dockerfile. You can run the AI workload using either method, allowing you to choose one option.

Before using a Docker container, Hugepage and KMD should be configured on your host PC. Please refer to the 'Configure Hugepage on Linux' and 'Installing KMD driver' sections in the [Bare-metal](#bare-metal) document.

### Download docker image[](#download-docker-image)

Please refer to the Docker image download link in the release note and download the image. The downloaded file might be compressed using gzip or 7zip, so extract the Docker image using gunzip or 7zip before loading it into Docker.

- Unpack the downloaded file and load image in docker
`$ gunzip bos_host.tar.gz`
 $ docker load -i bos_host.tar
- Then you can see the docker image
`$ docker images`
 IMAGE ID DISK USAGE CONTENT SIZE EXTRA
 bos_host:latest d3ea56eb5d49 17.1GB 0B

### Build your own Docker image[](#build-your-own-docker-image)

Clone the `tt-metal` repository on your host PC. Please refer to the 'Installing `tt-metal`' section in the release note. You do not need to build `tt-metal` when you use a Docker container.

- Copy the `Dockerfile`to the`tt-metal`directory.`$ cp docker/Dockerfile ${TT_METAL_DIRECTORY_ROOT}`
 $ cd ${TT_METAL_DIRECTORY_ROOT}
 $ docker build -t bos_host .
- Then you can see the docker image.
`$ docker images`
 IMAGE ID DISK USAGE CONTENT SIZE EXTRA
 bos_host:latest d3ea56eb5d49 17.1GB 0B

### Run Docker container[](#run-docker-container)

Run the Docker container using the Docker image

```
$ docker run -dit --device=/dev/bos/0:/dev/bos/0 \
  --cap-add=ALL \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -v /dev/mem:/dev/mem \
  -v /sys/bus/pci/devices/:/sys/bus/pci/devices/ \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
  --ipc=host \
  --name bos_host \
  bos_host:latest
```
Attach the docker container

```
$ docker exec -it bos_host:latest /bin/bash
root@xxxxxxxxxxxxxx # 
```
Please refer to the [Bare-metal](#bare-metal) guide for display setup instructions.

### Run AI models on the Eagle-N A0 board[](#run-ai-models-on-the-eagle-n-a0-board-1)

To run an AI model, refer to the `README.md` file in each model directory specified in the release note.

## 이미지

![BOS_Logo_4E_RGB.avif](_assets/BOS_Logo_4E_RGB.avif)  <!-- https://developers.bos-semi.com/img/BOS_Logo_4E_RGB.avif -->
