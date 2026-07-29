# Installing Host PC Software Stack

## Configure Hugepage on Linux

Open grub configuration and apply hugepage confguration
```
$ sudo nano /etc/default/grub
```
```grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash hugepagesz=1G hugepages=4 default_hugepagesz=1G"
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

After reboot, verify the hugepage configuration are applied and the hugepage size is `1048576 KB`  
```
$ cat /proc/meminfo | grep Huge
AnonHugePages:         0 kB
ShmemHugePages:    57344 kB
FileHugePages:         0 kB
HugePages_Total:       4
HugePages_Free:        3
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:    1048576 kB
Hugetlb:         4194304 kB
```

Mound hugepage file system
```
$ sudo mkdir -p /dev/hugepages-1G
$ sudo mount -t hugetlbfs -o pagesize=1G,mode=0777 hugetlbfs /dev/hugepages-1G
$ mount | grep hugepages
hugetlbfs on /dev/hugepages type hugetlbfs (rw,relatime,pagesize=1024M)
hugetlbfs on /dev/hugepages-1G type hugetlbfs (rw,relatime,mode=777,pagesize=1024M)
```

## Installing KMD driver

Clone the KMD Driver from GitHub repository (https://github.com/bos-semi-release/BOS_KMD.git) and execute installation script.  

Since the clone for the main repositories and submodule are processed with SSH protocol, therefore, please configure the SSH Keygen on the GitHub and local machine before cloning:
( refer the following guide in GitHub: [Connecting to GitHub with SSH - GitHub Docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) )

```
$ git clone git@github.com:bos-semi-release/BOS_KMD.git

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

Check the driver module name is `bos`
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

## Installing **tt-metal**

### System-level dependencies
  
**NOTE**: Docker container that contains all the software stack could be used instead of installing the **tt-metal**. Please refer the 'Docker image and Dockefile' section of release note.  
<br>


Disable ‘UEFI Secure boot’ from the host BIOS setup, if your host PC supports this feature.

Clone the tt-metal with all all submodules from the GitHub URL (https://github.com/bos-semi-release/tt-metal) with branch/tag as below(branch/tag information can be found in release notes.) 

```
$ git clone --branch=${RELEASE_BRANCH_NAME} git@github.com:bos-semi-release/tt-metal.git --recurse-submodules
Cloning into 'tt-metal' ...
```

<br>

Installing CMake 3.20

CMake 3.16 is installed in Ubuntu 20.04(default) but some dependencies like unity require CMake 3.20.
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
<br>

### Third-party dependencies

All of the third-party submodule dependencies below are cloned during cloning tt-metal:  
- bos-metal, SHA: 32cd678a7fbb48703a67f536399529c33bc42ded
- tracy, SHA: a0668deca5495616b0ccd1bd618552f5a4cfb3e7
- tt-llk, SHA: 2c367bc0215369cb0dd5405edb17cd5bdea66f15
- umd, SHA: 972bf3330285225c905ba97d551e17203a470bc1
<br>

### Building tt-metal
```
## Configure Environment Variables
$ cd ${PATH_TO_SW_PACKAGE_SOURCE_ROOT}/tt-metal/tt-metal
$ source env_set.sh

## Build TT-Metal (in Release mode for demo)
$ ./build_metal.sh -b Release
```

### Creating Python Virtual Environment
```
## Create virtual environment
$ ./create_venv.sh

## Activate virtual environment
$ source env_set.sh
```

### Run AI models on the Eagle-N A0 board

To run an AI model, refer to the `README.md` file in each model directory specified in the release note.
